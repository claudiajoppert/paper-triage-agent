"""End-to-end tests using the StubCaller — no API calls."""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import PaperMeta, StubCaller, Verdict, triage_paper, render_markdown


STRONG_PAPER_TEXT = """
Title: Effect of Intervention X on Outcome Y: A Multicenter Trial

Abstract:
This randomized controlled trial enrolled 1,240 patients across 12 sites.
Methods: We conducted a double-blind, placebo-controlled RCT. Protocol was
prospectively registered on ClinicalTrials.gov. Primary outcome analyzed by
intention-to-treat.

Results: Intervention X reduced the primary outcome (hazard ratio 0.72,
95% CI 0.61-0.84, p < 0.001). Effect size (Cohen's d) was 0.35.

Conclusion: Intervention X meaningfully reduced outcome Y in this population.
Conflicts of interest: None declared. Funding: NIH grant R01-XXXXXX.
"""


WEAK_PAPER_TEXT = """
Title: A breakthrough in weight loss

Abstract: This revolutionary, groundbreaking study proves that drinking
lemon water causes dramatic weight loss. We observed 8 patients in our clinic
retrospectively. Methods: We looked at charts. Drinking lemon water leads to
reduced appetite and triggers metabolic changes that cause fat loss.

This miracle finding represents an unprecedented breakthrough.
"""


STUB_RESPONSES_STRONG = {
    # "senior editor" must come before "SKEPTIC": REFEREE_SYSTEM mentions
    # "defender/skeptic" so the "SKEPTIC" key would otherwise match it first.
    "senior editor": json.dumps({
        "verdict": "CREDIBLE",
        "confidence": 0.83,
        "summary": "A well-designed multicenter RCT with preregistration and proper effect-size reporting. The evidence reasonably supports the stated conclusion.",
        "caveats": ["Replication in an independent cohort is still desirable."],
        "follow_ups": ["Check whether results replicated in follow-up trials."],
    }),
    "DEFENSE": json.dumps({
        "verdict": "CREDIBLE",
        "confidence": 0.85,
        "strongest_points": [
            "Large multicenter RCT with 1,240 patients",
            "Preregistered on ClinicalTrials.gov",
            "Effect sizes and CIs reported alongside p-values",
        ],
        "weakest_points": ["Single trial — replication pending"],
        "unresolved_questions": [],
    }),
    "SKEPTIC": json.dumps({
        "verdict": "PROMISING",
        "confidence": 0.75,
        "strongest_points": ["Well-designed RCT"],
        "weakest_points": [
            "Industry/NIH funding should be scrutinized for incentives",
            "Single trial risk — effect sizes often shrink in replication",
        ],
        "unresolved_questions": ["Long-term safety data?"],
    }),
    "methodologist": json.dumps({
        "verdict": "CREDIBLE",
        "confidence": 0.9,
        "strongest_points": [
            "Study design appropriate for causal inference",
            "Double-blinding reduces bias",
            "Preregistration reduces p-hacking risk",
        ],
        "weakest_points": [],
        "unresolved_questions": [],
    }),
}


STUB_RESPONSES_WEAK = {
    # "senior editor" first — same routing fix as STUB_RESPONSES_STRONG.
    "senior editor": json.dumps({
        "verdict": "UNRELIABLE",
        "confidence": 0.9,
        "summary": "Eight-patient retrospective chart review cannot support causal claims. Extensive hype language compounds the problem.",
        "caveats": ["No credible evidence for the claimed effect."],
        "follow_ups": [],
    }),
    "DEFENSE": json.dumps({
        "verdict": "WEAK",
        "confidence": 0.4,
        "strongest_points": ["Observation is at least reported honestly"],
        "weakest_points": [
            "n=8 is far too small",
            "Retrospective chart review cannot establish causation",
            "Spin language is extreme",
        ],
        "unresolved_questions": [],
    }),
    "SKEPTIC": json.dumps({
        "verdict": "UNRELIABLE",
        "confidence": 0.9,
        "strongest_points": [],
        "weakest_points": [
            "n=8 insufficient for any claim",
            "Retrospective design, no control group",
            "'Proves that', 'causes', 'breakthrough', 'miracle' — classic hype",
        ],
        "unresolved_questions": [],
    }),
    "methodologist": json.dumps({
        "verdict": "UNRELIABLE",
        "confidence": 0.95,
        "strongest_points": [],
        "weakest_points": [
            "No control, no randomization, no blinding",
            "Sample size cannot support the claim",
        ],
        "unresolved_questions": [],
    }),
}


def test_strong_paper_full_pipeline():
    caller = StubCaller(responses=STUB_RESPONSES_STRONG)
    meta = PaperMeta(
        title="Effect of Intervention X on Outcome Y",
        venue="NEJM",
        year=2024,
    )
    result = triage_paper(meta, STRONG_PAPER_TEXT, caller, parallel=False)

    assert result.final_verdict == Verdict.CREDIBLE
    assert result.final_confidence > 0.7
    assert not result.hard_fail.failed
    assert len(result.opinions) == 3
    assert result.signals.study_type == "rct"
    assert result.signals.sample_size == 1240
    # stub was called 4 times: defender, skeptic, methodologist, referee
    assert len(caller.calls) == 4


def test_weak_paper_full_pipeline():
    caller = StubCaller(responses=STUB_RESPONSES_WEAK)
    meta = PaperMeta(title="Lemon water breakthrough", venue="Some Blog")
    result = triage_paper(meta, WEAK_PAPER_TEXT, caller, parallel=False)

    assert result.final_verdict == Verdict.UNRELIABLE
    assert len(result.opinions) == 3


def test_retracted_paper_skips_ai():
    """Hard-fail path should never call the LLM."""
    caller = StubCaller(responses=STUB_RESPONSES_STRONG)
    meta = PaperMeta(title="Retracted", is_retracted=True,
                     retraction_notes="fraud", venue="NEJM")
    result = triage_paper(meta, STRONG_PAPER_TEXT, caller, parallel=False)

    assert result.hard_fail.failed
    assert result.final_verdict == Verdict.UNRELIABLE
    # critical: the stub was NEVER called
    assert len(caller.calls) == 0


def test_validator_overrides_ai_on_small_sample():
    """Even if AI says CREDIBLE, small sample should force a downgrade."""
    # Paper text with RCT language but tiny n
    text = "We ran a randomized controlled trial on N=10 participants. p < 0.01."
    overconfident_ai = {
        "DEFENSE": json.dumps({"verdict": "CREDIBLE", "confidence": 0.9,
                               "strongest_points": [], "weakest_points": [], "unresolved_questions": []}),
        "SKEPTIC": json.dumps({"verdict": "CREDIBLE", "confidence": 0.9,
                               "strongest_points": [], "weakest_points": [], "unresolved_questions": []}),
        "methodologist": json.dumps({"verdict": "CREDIBLE", "confidence": 0.9,
                                     "strongest_points": [], "weakest_points": [], "unresolved_questions": []}),
        "senior editor": json.dumps({
            "verdict": "CREDIBLE", "confidence": 0.9,
            "summary": "Looks great.", "caveats": [], "follow_ups": []
        }),
    }
    caller = StubCaller(responses=overconfident_ai)
    result = triage_paper(PaperMeta(title="tiny trial", venue="J Stuff"),
                          text, caller, parallel=False)
    # validator should have caught this
    assert result.final_verdict == Verdict.PROMISING
    assert any("sample size" in c.lower() for c in result.caveats)


def test_render_markdown_produces_output():
    caller = StubCaller(responses=STUB_RESPONSES_STRONG)
    meta = PaperMeta(title="Test paper", venue="NEJM", year=2024)
    result = triage_paper(meta, STRONG_PAPER_TEXT, caller, parallel=False)
    md = render_markdown(result)
    assert "Test paper" in md
    assert "CREDIBLE" in md
    assert "Defender" in md
    assert "Skeptic" in md
    assert "Methodologist" in md


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ✓ {t.__name__}")
        except Exception:
            failed += 1
            print(f"  ✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
