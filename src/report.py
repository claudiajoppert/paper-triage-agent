"""
Markdown report renderer.

Takes a TriageResult and produces a readable, shareable report.
"""
from __future__ import annotations

from .schema import HardFail, TriageResult, Verdict


_VERDICT_EMOJI = {
    Verdict.CREDIBLE:   "✅",
    Verdict.PROMISING:  "🟡",
    Verdict.WEAK:       "⚠️",
    Verdict.UNRELIABLE: "❌",
}


def render_markdown(result: TriageResult) -> str:
    lines: list[str] = []
    m = result.meta
    s = result.signals

    title = m.title or "(untitled paper)"
    verdict_icon = _VERDICT_EMOJI.get(result.final_verdict, "")
    lines.append(f"# Paper Triage: {title}")
    lines.append("")
    lines.append(f"**Verdict:** {verdict_icon} **{result.final_verdict.value}**  "
                 f"(confidence: {result.final_confidence:.2f})")
    lines.append("")

    # metadata
    lines.append("## Source")
    lines.append(f"- **Venue:** {m.venue or 'unknown'}"
                 + (" *(preprint)*" if m.is_preprint else ""))
    if m.year:
        lines.append(f"- **Year:** {m.year}")
    if m.authors:
        lines.append(f"- **Authors:** {', '.join(m.authors[:5])}"
                     + (" et al." if len(m.authors) > 5 else ""))
    if m.doi:
        lines.append(f"- **DOI:** {m.doi}")
    if m.url:
        lines.append(f"- **URL:** {m.url}")
    lines.append("")

    # bottom line
    if result.summary:
        lines.append("## Bottom Line")
        lines.append(result.summary)
        lines.append("")

    # hard-fail short-circuit
    if result.hard_fail.failed:
        lines.append("## ❌ Rejected by Gatekeeper")
        reasons = [r.value for r in result.hard_fail.reasons if r != HardFail.NONE]
        lines.append(f"**Reasons:** {', '.join(reasons)}")
        if result.hard_fail.notes:
            lines.append("")
            for n in result.hard_fail.notes:
                lines.append(f"- {n}")
        lines.append("")
        lines.append("*No AI review was performed. These are deterministic rejection rules.*")
        return "\n".join(lines)

    # signals block
    lines.append("## Study Signals (deterministic)")
    lines.append(f"- **Study type:** {s.study_type}")
    lines.append(f"- **Sample size:** {s.sample_size if s.sample_size else 'not detected'}")
    lines.append(f"- **Control group:** {_tribool(s.has_control_group)}")
    lines.append(f"- **Preregistration:** {_tribool(s.has_preregistration)}")
    lines.append(f"- **P-values reported:** {len(s.p_values_reported)} "
                 f"(min: {min(s.p_values_reported) if s.p_values_reported else '—'})")
    lines.append(f"- **Confidence intervals:** {s.confidence_intervals_reported}")
    lines.append(f"- **Effect sizes:** {s.effect_sizes_reported}")
    lines.append(f"- **COI declared:** {_tribool(s.conflicts_of_interest_declared)}")
    if s.animal_or_invitro_only:
        lines.append("- ⚠️  **Animal / in-vitro only**")
    if s.observational_causal_language:
        lines.append("- ⚠️  **Causal language in observational design**")
    if s.spin_phrases:
        lines.append(f"- ⚠️  **Spin phrases:** {', '.join(s.spin_phrases)}")
    lines.append("")

    # agent opinions
    if result.opinions:
        lines.append("## Reviewer Opinions")
        for op in result.opinions:
            lines.append("")
            lines.append(f"### {op.role.title()} — {op.verdict.value} ({op.confidence:.2f})")
            if op.strongest_points:
                lines.append("**Strongest points:**")
                for p in op.strongest_points:
                    lines.append(f"- {p}")
            if op.weakest_points:
                lines.append("**Weakest points:**")
                for p in op.weakest_points:
                    lines.append(f"- {p}")
            if op.unresolved_questions:
                lines.append("**Unresolved:**")
                for q in op.unresolved_questions:
                    lines.append(f"- {q}")
        lines.append("")

    # caveats
    if result.caveats:
        lines.append("## Caveats")
        for c in result.caveats:
            lines.append(f"- {c}")
        lines.append("")

    # follow-ups
    if result.recommended_follow_ups:
        lines.append("## Recommended Follow-ups")
        for f in result.recommended_follow_ups:
            lines.append(f"- {f}")
        lines.append("")

    lines.append("---")
    lines.append("*This is an automated triage and does not replace expert peer review.*")
    return "\n".join(lines)


def _tribool(v: bool | None) -> str:
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return "unclear"
