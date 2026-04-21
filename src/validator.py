"""
Verdict validator.

The referee AI issues a verdict. This module sanity-checks that verdict
against the deterministic signals. If the AI says CREDIBLE but the paper
has a sample size of 8, no control group, and the word "breakthrough"
three times, we override.

This is the final deterministic check — the AI proposes, the rules dispose.
"""
from __future__ import annotations

from .schema import StudySignals, Verdict


def validate_and_adjust(
    ai_verdict: Verdict,
    ai_confidence: float,
    signals: StudySignals,
) -> tuple[Verdict, float, list[str]]:
    """
    Returns (possibly-adjusted verdict, possibly-adjusted confidence, list of adjustment notes).

    Rules are conservative — they only DOWNGRADE, never upgrade. We'd rather
    miss a credible paper than falsely endorse a weak one.
    """
    notes: list[str] = []
    verdict = ai_verdict
    confidence = ai_confidence

    # Rule 1: credibility requires a minimum evidence floor.
    if verdict == Verdict.CREDIBLE:
        if signals.sample_size is not None and signals.sample_size < 30:
            verdict = Verdict.PROMISING
            notes.append(
                f"Downgraded CREDIBLE → PROMISING: sample size {signals.sample_size} "
                "is too small to support a strong credibility verdict"
            )
        if signals.study_type == "case-report":
            verdict = Verdict.PROMISING
            notes.append("Downgraded CREDIBLE → PROMISING: case reports generate hypotheses, not conclusions")
        if signals.animal_or_invitro_only:
            verdict = Verdict.PROMISING
            notes.append("Downgraded CREDIBLE → PROMISING: animal/in-vitro results don't credibly transfer to humans")

    # Rule 2: heavy spin language caps credibility regardless of AI take.
    if len(signals.spin_phrases) >= 3 and verdict == Verdict.CREDIBLE:
        verdict = Verdict.PROMISING
        notes.append(
            f"Downgraded CREDIBLE → PROMISING: {len(signals.spin_phrases)} spin phrases "
            f"detected ({', '.join(signals.spin_phrases[:3])}...)"
        )

    # Rule 3: observational + causal language = overclaim.
    if signals.observational_causal_language and verdict in (Verdict.CREDIBLE, Verdict.PROMISING):
        old = verdict
        verdict = Verdict.WEAK
        notes.append(
            f"Downgraded {old.value} → WEAK: observational study with causal language "
            "(classic overclaim pattern)"
        )

    # Rule 4: if the AI was highly confident but we just overrode, clip its confidence.
    if notes and confidence > 0.7:
        confidence = 0.6
        notes.append("Capped confidence at 0.6 due to deterministic override(s)")

    return verdict, confidence, notes
