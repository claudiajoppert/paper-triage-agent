"""
AI agents. Each gets the same deterministic signals but argues a
different role — like the Bull/Bear/Risk pattern in the reference repo.

Roles:
  - defender: argues the paper's findings are credible AND correctly claimed
  - skeptic:  argues the findings are overstated, confounded, or fragile
  - methodologist: neutral evaluator focused purely on study design quality
  - referee:  reads all three, reconciles, issues final verdict

Each call goes through `llm_call()` which this module abstracts. In
production you wire that to Anthropic/OpenAI/Gemini. For tests we
inject a stub.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from typing import Callable, Protocol

from .schema import AgentOpinion, PaperMeta, StudySignals, Verdict


class LLMCaller(Protocol):
    """Pluggable LLM interface. Tests inject fake callers."""
    def __call__(self, system: str, user: str) -> str: ...


# --- shared evidence block, built once per paper ---------------------------

def _evidence_block(meta: PaperMeta, signals: StudySignals) -> str:
    """
    Deterministic evidence all agents see. No persuasion, just facts.
    Mirrors the reference repo's DATA_BLOCK idea.
    """
    return (
        "=== EVIDENCE_BLOCK (deterministic) ===\n"
        + json.dumps({
            "paper": {
                "title": meta.title,
                "venue": meta.venue,
                "year": meta.year,
                "is_preprint": meta.is_preprint,
                "doi": meta.doi,
            },
            "signals": asdict(signals),
        }, indent=2, default=str)
        + "\n=== END EVIDENCE_BLOCK ==="
    )


# --- prompt templates ------------------------------------------------------

_OUTPUT_CONTRACT = """
Respond with a SINGLE JSON object and nothing else. Schema:
{
  "verdict": "CREDIBLE" | "PROMISING" | "WEAK" | "UNRELIABLE",
  "confidence": <float 0.0-1.0>,
  "strongest_points": [<string>, ...],   // 1-4 items
  "weakest_points":   [<string>, ...],   // 1-4 items
  "unresolved_questions": [<string>, ...] // 0-3 items
}
Do not wrap in markdown fences. Do not add prose before or after.
"""


DEFENDER_SYSTEM = f"""You are a careful academic peer reviewer arguing in DEFENSE of the paper.
Your job is to find the strongest reasons the paper's conclusions are supported by its evidence.
You are NOT a cheerleader: if the evidence is weak, your defense will naturally be weak, and you
should admit that. But you should steelman the paper's case.

Base your arguments on the EVIDENCE_BLOCK. Do not invent facts that aren't there.

{_OUTPUT_CONTRACT}"""


SKEPTIC_SYSTEM = f"""You are a careful academic peer reviewer arguing SKEPTICALLY of the paper.
Your job is to find the strongest reasons the paper's conclusions OVERREACH its evidence:
confounders, small samples, multiple-comparison risk, spin language, design gaps, etc.

Do NOT reject good work reflexively — if the methodology is strong, say so even while flagging
any remaining concerns. Skepticism is a lens, not a verdict.

Base your arguments on the EVIDENCE_BLOCK. Do not invent problems that aren't there.

{_OUTPUT_CONTRACT}"""


METHODOLOGIST_SYSTEM = f"""You are a neutral methodologist. Ignore the paper's topic and CLAIMS.
Evaluate only the QUALITY OF DESIGN AND ANALYSIS as reflected in the EVIDENCE_BLOCK:
  - Is the study type appropriate for the claim?
  - Is the sample size adequate given the effect being measured?
  - Are effect sizes and confidence intervals reported, or only p-values?
  - Is there preregistration, control, blinding where relevant?
  - Are conflicts of interest disclosed?

Your verdict reflects DESIGN quality, not whether the hypothesis is interesting.

{_OUTPUT_CONTRACT}"""


REFEREE_SYSTEM = """You are the senior editor. Three reviewers have given structured opinions on
a paper. You also have the deterministic EVIDENCE_BLOCK.

Your job:
  1. Reconcile the three opinions. Weight the methodologist heavily on design quality,
     and weight defender/skeptic for whether the CLAIMS match the evidence.
  2. Issue a final verdict. Be calibrated: CREDIBLE is a high bar (strong design + claims
     proportionate to evidence). UNRELIABLE means serious methodological problems OR
     claims wildly exceeding evidence.
  3. Write a 2-3 sentence plain-English summary a non-expert could understand.
  4. List concrete caveats and follow-up questions a reader should hold in mind.

Respond with a SINGLE JSON object:
{
  "verdict": "CREDIBLE" | "PROMISING" | "WEAK" | "UNRELIABLE",
  "confidence": <float 0.0-1.0>,
  "summary": <string, 2-3 sentences>,
  "caveats": [<string>, ...],
  "follow_ups": [<string>, ...]
}
Do not wrap in markdown fences. Do not add prose before or after."""


# --- agent runners ---------------------------------------------------------

def _run_opinion_agent(
    role: str,
    system: str,
    evidence: str,
    llm_call: LLMCaller,
) -> AgentOpinion:
    """Run one opinion agent and parse to AgentOpinion."""
    raw = llm_call(system=system, user=evidence)
    data = _parse_json_safely(raw, fallback={
        "verdict": "WEAK",
        "confidence": 0.3,
        "strongest_points": [],
        "weakest_points": [f"{role} produced unparseable output"],
        "unresolved_questions": [],
    })
    try:
        verdict = Verdict(data["verdict"])
    except (KeyError, ValueError):
        verdict = Verdict.WEAK
    return AgentOpinion(
        role=role,
        verdict=verdict,
        confidence=_clamp(float(data.get("confidence", 0.5)), 0.0, 1.0),
        strongest_points=list(data.get("strongest_points", []))[:4],
        weakest_points=list(data.get("weakest_points", []))[:4],
        unresolved_questions=list(data.get("unresolved_questions", []))[:3],
    )


def run_defender(meta: PaperMeta, signals: StudySignals, llm_call: LLMCaller) -> AgentOpinion:
    return _run_opinion_agent("defender", DEFENDER_SYSTEM, _evidence_block(meta, signals), llm_call)


def run_skeptic(meta: PaperMeta, signals: StudySignals, llm_call: LLMCaller) -> AgentOpinion:
    return _run_opinion_agent("skeptic", SKEPTIC_SYSTEM, _evidence_block(meta, signals), llm_call)


def run_methodologist(meta: PaperMeta, signals: StudySignals, llm_call: LLMCaller) -> AgentOpinion:
    return _run_opinion_agent(
        "methodologist", METHODOLOGIST_SYSTEM, _evidence_block(meta, signals), llm_call
    )


def run_referee(
    meta: PaperMeta,
    signals: StudySignals,
    opinions: list[AgentOpinion],
    llm_call: LLMCaller,
) -> dict:
    """Synthesize opinions into the final verdict. Returns raw dict for post-processing."""
    user = (
        _evidence_block(meta, signals)
        + "\n\n=== REVIEWER_OPINIONS ===\n"
        + json.dumps([asdict(o) for o in opinions], indent=2, default=str)
        + "\n=== END REVIEWER_OPINIONS ==="
    )
    raw = llm_call(system=REFEREE_SYSTEM, user=user)
    return _parse_json_safely(raw, fallback={
        "verdict": "WEAK",
        "confidence": 0.3,
        "summary": "Referee output could not be parsed; defaulting to WEAK.",
        "caveats": ["Referee JSON parse failed — treat result with caution."],
        "follow_ups": [],
    })


# --- helpers ---------------------------------------------------------------

def _parse_json_safely(raw: str, fallback: dict) -> dict:
    """LLMs sometimes wrap JSON in fences or add chatter. Be forgiving."""
    if not raw:
        return fallback
    text = raw.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    # find first { and last } as a last-ditch
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        warnings.warn(f"LLM output could not be parsed as JSON; using fallback. Raw: {text[:200]!r}")
        return fallback


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
