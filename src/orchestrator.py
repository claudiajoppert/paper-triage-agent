"""
Orchestrator — the agentic workflow.

Flow:
    1. Extract deterministic signals from paper text (Python, no AI)
    2. Run hard-fail gatekeeper (Python, no AI)
       └─ if hard-fail: emit rejection report, skip debate
    3. Run three agents IN PARALLEL:
       - defender, skeptic, methodologist
    4. Run referee to reconcile
    5. Run validator to sanity-check the referee's verdict against signals
    6. Assemble final TriageResult

Steps 1, 2, and 5 are deterministic. Steps 3 and 4 are AI. The gate
(step 2) and validator (step 5) are the "checks" in the checks-and-AIs pattern.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .agents import (
    LLMCaller,
    run_defender,
    run_methodologist,
    run_referee,
    run_skeptic,
)
from .extractors import extract_signals
from .gatekeeper import check_hard_fails
from .schema import (
    AgentOpinion,
    HardFail,
    HardFailResult,
    PaperMeta,
    StudySignals,
    TriageResult,
    Verdict,
)
from .validator import validate_and_adjust


def triage_paper(
    meta: PaperMeta,
    raw_text: str,
    llm_call: LLMCaller,
    *,
    parallel: bool = True,
) -> TriageResult:
    """
    Run the full pipeline on a single paper.

    Parameters
    ----------
    meta     : PaperMeta with whatever was fetched from metadata APIs.
    raw_text : paper text (abstract at minimum, full text if available).
    llm_call : any callable with signature (system: str, user: str) -> str.
    parallel : whether to fan out the three opinion agents concurrently.

    Returns
    -------
    TriageResult — fully populated final report.
    """
    # --- step 1: deterministic signal extraction --------------------------
    signals = extract_signals(raw_text)

    # --- step 2: hard-fail gatekeeper -------------------------------------
    hard_fail = check_hard_fails(meta, signals, raw_text)
    if hard_fail.failed:
        return _rejection_report(meta, signals, hard_fail)

    # --- step 3: parallel agent fan-out -----------------------------------
    if parallel:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                "defender":      pool.submit(run_defender, meta, signals, llm_call),
                "skeptic":       pool.submit(run_skeptic, meta, signals, llm_call),
                "methodologist": pool.submit(run_methodologist, meta, signals, llm_call),
            }
            opinions = []
            for name in ("defender", "skeptic", "methodologist"):
                try:
                    opinions.append(futures[name].result())
                except Exception as exc:
                    warnings.warn(f"{name} agent raised an exception: {exc}; substituting WEAK stub opinion")
                    opinions.append(AgentOpinion(
                        role=name,
                        verdict=Verdict.WEAK,
                        confidence=0.3,
                        strongest_points=[],
                        weakest_points=[f"{name} agent failed: {exc}"],
                        unresolved_questions=[],
                    ))
    else:
        opinions = [
            run_defender(meta, signals, llm_call),
            run_skeptic(meta, signals, llm_call),
            run_methodologist(meta, signals, llm_call),
        ]

    # --- step 4: referee synthesis ----------------------------------------
    referee_out = run_referee(meta, signals, opinions, llm_call)
    try:
        raw_verdict = Verdict(referee_out.get("verdict", "WEAK"))
    except ValueError:
        raw_verdict = Verdict.WEAK
    raw_confidence = _clamp(float(referee_out.get("confidence", 0.5)), 0.0, 1.0)

    # --- step 5: deterministic validation ---------------------------------
    final_verdict, final_confidence, adjustment_notes = validate_and_adjust(
        raw_verdict, raw_confidence, signals
    )

    # --- step 6: assemble final result ------------------------------------
    caveats = list(referee_out.get("caveats", []))
    caveats.extend(adjustment_notes)

    return TriageResult(
        meta=meta,
        signals=signals,
        hard_fail=hard_fail,
        opinions=opinions,
        final_verdict=final_verdict,
        final_confidence=final_confidence,
        summary=str(referee_out.get("summary", "")),
        caveats=caveats,
        recommended_follow_ups=list(referee_out.get("follow_ups", [])),
    )


def _rejection_report(
    meta: PaperMeta, signals: StudySignals, hard_fail: HardFailResult
) -> TriageResult:
    """Build a result for hard-failed inputs without calling the LLM."""
    reason_strs = [r.value for r in hard_fail.reasons if r != HardFail.NONE]
    summary = (
        "Paper rejected by deterministic gatekeeper: "
        + ", ".join(reason_strs)
        + ". No AI review was performed."
    )
    return TriageResult(
        meta=meta,
        signals=signals,
        hard_fail=hard_fail,
        opinions=[],
        final_verdict=Verdict.UNRELIABLE,
        final_confidence=0.95,  # high confidence in the rejection itself
        summary=summary,
        caveats=hard_fail.notes,
        recommended_follow_ups=[],
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
