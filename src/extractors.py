"""
Deterministic extractors. No LLM calls here.

These functions read paper text and pull out checkable facts:
sample sizes, p-values, study type hints, spin language, etc.

The whole point of this layer is that its output is auditable and
reproducible. The AI agents downstream work FROM these signals,
they don't replace them.
"""
from __future__ import annotations

import re
from typing import Iterable

from .schema import StudySignals


# --- spin / hype phrases commonly flagged in the methodology literature
SPIN_PHRASES = [
    "breakthrough", "miracle", "revolutionary", "game-changer",
    "proves that", "proven to", "cure for", "unprecedented",
    "groundbreaking", "first ever to show",
]

# --- causal verbs that are suspect when paired with observational designs
CAUSAL_VERBS = [
    r"\bcauses?\b", r"\bcaused\b", r"\bleads? to\b", r"\bresults in\b",
    r"\btriggers?\b", r"\bprevents?\b",
]

STUDY_TYPE_HINTS = {
    "rct": [
        r"randomi[sz]ed controlled trial", r"\bRCT\b",
        r"double[- ]blind", r"placebo[- ]controlled",
    ],
    "meta-analysis": [
        r"meta[- ]analysis", r"systematic review and meta",
    ],
    "review": [
        r"\bsystematic review\b", r"\bnarrative review\b", r"\bscoping review\b",
    ],
    "observational": [
        r"\bcohort study\b", r"\bcase[- ]control\b", r"\bcross[- ]sectional\b",
        r"\bobservational\b", r"\bretrospective(?:ly)?\b", r"\bprospective cohort\b",
    ],
    "case-report": [
        r"\bcase report\b", r"\bcase series\b",
    ],
    "preprint": [
        r"\bpreprint\b", r"not yet peer[- ]reviewed",
    ],
}

ANIMAL_INVITRO_HINTS = [
    r"\bin mice\b", r"\bin rats\b", r"\bmurine\b", r"\brodent model\b",
    r"\bin vitro\b", r"\bcell culture\b", r"\bin zebrafish\b",
    r"\banimal model\b",
]


def _detect_study_type(text: str) -> str:
    """Pick the strongest study-type signal. RCT/meta beat weaker ones."""
    lowered = text.lower()
    # precedence matters — a paper can mention "cohort" while being an RCT
    for stype in ("rct", "meta-analysis", "review", "observational", "case-report", "preprint"):
        for pattern in STUDY_TYPE_HINTS[stype]:
            if re.search(pattern, lowered):
                return stype
    return "unknown"


def _extract_sample_size(text: str) -> int | None:
    """
    Find the largest plausible sample size mentioned.

    Matches forms like:
      "n = 1,234"  |  "N=42"  |  "1,234 participants"  |  "enrolled 220 patients"
    Returns None if nothing plausible is found.
    """
    candidates: list[int] = []

    for m in re.finditer(r"[nN]\s*=\s*([\d,]{1,10})", text):
        candidates.append(_clean_int(m.group(1)))

    for m in re.finditer(
        r"([\d,]{2,10})\s+(?:participants|patients|subjects|individuals|respondents)",
        text, re.IGNORECASE,
    ):
        candidates.append(_clean_int(m.group(1)))

    for m in re.finditer(
        r"(?:enrolled|recruited|included)\s+([\d,]{2,10})\s+(?:participants|patients|subjects)?",
        text, re.IGNORECASE,
    ):
        candidates.append(_clean_int(m.group(1)))

    plausible = [c for c in candidates if c is not None and 2 <= c <= 10_000_000]
    return max(plausible) if plausible else None


def _clean_int(raw: str) -> int | None:
    try:
        return int(raw.replace(",", ""))
    except ValueError:
        return None


def _extract_p_values(text: str) -> list[float]:
    """Pull reported p-values. Handles p<0.05, p = .003, p=1e-6, etc."""
    vals: list[float] = []
    for m in re.finditer(r"[pP]\s*[=<>]\s*(0?\.\d+|\d\.\d+e-?\d+)", text):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            continue
    return vals


def _count_pattern(text: str, pattern: str, flags: int = re.IGNORECASE) -> int:
    return len(re.findall(pattern, text, flags))


def _any_match(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _find_phrases(text: str, phrases: Iterable[str]) -> list[str]:
    lowered = text.lower()
    return [p for p in phrases if p in lowered]


def _bool_from_hints(text: str, yes_patterns: list[str], no_patterns: list[str]) -> bool | None:
    """Return True/False if we find strong hints either way, else None."""
    if _any_match(text, yes_patterns):
        return True
    if _any_match(text, no_patterns):
        return False
    return None


def extract_signals(text: str) -> StudySignals:
    """
    Main entry point. Given paper text (abstract + methods ideally),
    pull deterministic signals we can reason over.
    """
    if not text:
        return StudySignals()

    study_type = _detect_study_type(text)

    has_control = _bool_from_hints(
        text,
        yes_patterns=[r"control group", r"placebo[- ]controlled", r"vs\.?\s+placebo"],
        no_patterns=[r"no control group", r"single[- ]arm"],
    )

    has_prereg = _bool_from_hints(
        text,
        yes_patterns=[r"pre[- ]?registered", r"prospectively registered",
                      r"ClinicalTrials\.gov", r"OSF registration"],
        no_patterns=[],
    )

    coi_declared = _bool_from_hints(
        text,
        yes_patterns=[r"conflict[s]? of interest", r"competing interests",
                      r"declaration of interests", r"the authors declare"],
        no_patterns=[],
    )

    funding = _bool_from_hints(
        text,
        yes_patterns=[r"\bfunding\b", r"\bfunded by\b", r"\bgrant\b", r"\bsponsored by\b"],
        no_patterns=[],
    )

    # observational + causal language = classic over-claim pattern
    causal_count = sum(_count_pattern(text, v) for v in CAUSAL_VERBS)
    observational_causal = (
        study_type in ("observational", "unknown") and causal_count >= 2
    )

    return StudySignals(
        study_type=study_type,
        sample_size=_extract_sample_size(text),
        has_control_group=has_control,
        has_preregistration=has_prereg,
        p_values_reported=_extract_p_values(text),
        confidence_intervals_reported=_count_pattern(text, r"\b95%\s*CI\b")
                                      + _count_pattern(text, r"confidence interval"),
        effect_sizes_reported=_count_pattern(
            text, r"\b(?:odds ratio|OR|hazard ratio|HR|risk ratio|RR|Cohen'?s d|effect size)\b",
        ),
        conflicts_of_interest_declared=coi_declared,
        funding_disclosed=funding,
        animal_or_invitro_only=_any_match(text, ANIMAL_INVITRO_HINTS)
                               and not re.search(r"\bhuman\b", text, re.IGNORECASE),
        observational_causal_language=observational_causal,
        spin_phrases=_find_phrases(text, SPIN_PHRASES),
    )
