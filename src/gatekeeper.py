"""
Hard-fail gatekeeper.

Runs before any AI debate. If a paper trips one of these wires, we
reject it deterministically and skip the expensive reasoning steps.

This is the pattern from the reference investment-agent repo: the
'red_flag_detector' that runs in pure Python before the bull/bear
debate fires. Saves money, saves time, and makes rejection auditable.
"""
from __future__ import annotations

from .schema import HardFail, HardFailResult, PaperMeta, StudySignals


# Known predatory / questionable publishers. Non-exhaustive — a real
# deployment would pull from Beall's list successor or Cabells.
PREDATORY_VENUE_KEYWORDS = [
    "omics publishing", "scirp", "scientific research publishing",
    "bentham open", "academic journals online",
    # generic red flags
    "international journal of advanced research in everything",
]


def check_hard_fails(meta: PaperMeta, signals: StudySignals, raw_text: str) -> HardFailResult:
    """
    Apply deterministic disqualifiers.

    Returns HardFailResult.failed=True if ANY trip. The caller should
    skip the debate step in that case and route straight to a short
    rejection report.
    """
    reasons: list[HardFail] = []
    notes: list[str] = []

    if meta.is_retracted:
        reasons.append(HardFail.RETRACTED)
        notes.append(
            f"Paper is marked retracted"
            + (f": {meta.retraction_notes}" if meta.retraction_notes else "")
        )

    venue_lower = (meta.venue or "").lower()
    if any(kw in venue_lower for kw in PREDATORY_VENUE_KEYWORDS):
        reasons.append(HardFail.PREDATORY_VENUE)
        notes.append(f"Venue '{meta.venue}' appears on a predatory-publisher watchlist")

    # "No methods section" — heuristic. If we have full text but can't
    # find any methods-like language, the artifact is probably not a
    # research paper (editorial, op-ed, press release scraped as HTML).
    if meta.full_text_available and raw_text:
        methods_present = any(
            h in raw_text.lower()
            for h in ("methods", "methodology", "materials and methods",
                      "study design", "participants", "procedure")
        )
        if not methods_present:
            reasons.append(HardFail.NO_METHODS)
            notes.append("Full text available but no methods-like section found")

    # If we have essentially no extractable signals AND no methods,
    # the input probably isn't a paper at all.
    no_stats = (
        signals.sample_size is None
        and not signals.p_values_reported
        and signals.confidence_intervals_reported == 0
        and signals.effect_sizes_reported == 0
    )
    if no_stats and signals.study_type == "unknown" and len(raw_text) < 500:
        reasons.append(HardFail.NOT_A_PAPER)
        notes.append(
            "Input too short and contains no statistical signals — "
            "likely not a research paper"
        )

    if not reasons:
        reasons.append(HardFail.NONE)

    failed = any(r != HardFail.NONE for r in reasons)
    return HardFailResult(failed=failed, reasons=reasons, notes=notes)
