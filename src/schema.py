"""
Data structures passed between agents.

Everything the agents produce is typed. Downstream deterministic
code can safely parse these without string munging LLM output.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    CREDIBLE = "CREDIBLE"              # evidence is solid for the claim made
    PROMISING = "PROMISING"            # interesting but not definitive
    WEAK = "WEAK"                      # claim exceeds what the evidence supports
    UNRELIABLE = "UNRELIABLE"          # methodological problems or hard-fail trigger


class HardFail(str, Enum):
    RETRACTED = "RETRACTED"
    PREDATORY_VENUE = "PREDATORY_VENUE"
    NO_METHODS = "NO_METHODS"
    NOT_A_PAPER = "NOT_A_PAPER"        # input was e.g. a news piece, not research
    NONE = "NONE"


@dataclass
class PaperMeta:
    """Raw extracted facts about the paper. Deterministic — no AI judgment."""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    venue: str = ""                    # journal / conference / preprint server
    year: Optional[int] = None
    doi: str = ""
    url: str = ""
    is_preprint: bool = False
    is_retracted: bool = False
    retraction_notes: str = ""
    abstract: str = ""
    full_text_available: bool = False


@dataclass
class StudySignals:
    """
    Parsed from text by deterministic regex + light NLP.
    These are FACTS, not judgments. The AI reasons ABOUT them.
    """
    study_type: str = "unknown"        # rct | observational | meta-analysis | review | case-report | preprint | unknown
    sample_size: Optional[int] = None
    has_control_group: Optional[bool] = None
    has_preregistration: Optional[bool] = None
    p_values_reported: list[float] = field(default_factory=list)
    confidence_intervals_reported: int = 0
    effect_sizes_reported: int = 0
    conflicts_of_interest_declared: Optional[bool] = None
    funding_disclosed: Optional[bool] = None
    animal_or_invitro_only: bool = False  # flags "mice study" framed as human finding
    observational_causal_language: bool = False  # "X causes Y" in observational study
    spin_phrases: list[str] = field(default_factory=list)  # e.g. "breakthrough", "miracle"


@dataclass
class HardFailResult:
    failed: bool
    reasons: list[HardFail] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class AgentOpinion:
    """One AI agent's take. Structured so the referee can compare them."""
    role: str                          # "defender" | "skeptic" | "methodologist"
    verdict: Verdict
    confidence: float                  # 0.0 - 1.0
    strongest_points: list[str]
    weakest_points: list[str]
    unresolved_questions: list[str]


@dataclass
class TriageResult:
    """Final output of the whole pipeline."""
    meta: PaperMeta
    signals: StudySignals
    hard_fail: HardFailResult
    opinions: list[AgentOpinion]
    final_verdict: Verdict
    final_confidence: float
    summary: str                       # plain-english bottom line
    caveats: list[str]
    recommended_follow_ups: list[str]

    def to_dict(self) -> dict:
        return asdict(self)
