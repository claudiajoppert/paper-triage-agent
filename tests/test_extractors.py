"""Tests for the deterministic extractor layer."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractors import extract_signals


def test_detects_rct():
    text = "We conducted a double-blind, placebo-controlled randomized controlled trial (RCT)."
    s = extract_signals(text)
    assert s.study_type == "rct"


def test_detects_observational():
    text = "This was a retrospective cohort study of hospital admissions."
    s = extract_signals(text)
    assert s.study_type == "observational"


def test_extracts_sample_size_n_equals():
    text = "We enrolled participants (N = 1,234) from three clinics."
    s = extract_signals(text)
    assert s.sample_size == 1234


def test_extracts_sample_size_participants():
    text = "A total of 420 participants completed the study."
    s = extract_signals(text)
    assert s.sample_size == 420


def test_picks_largest_plausible_sample_size():
    """If multiple numbers, pick the largest plausible one (the total)."""
    text = "Of 500 patients screened, 420 were enrolled. Subgroup n=30 analyzed."
    s = extract_signals(text)
    assert s.sample_size == 500


def test_extracts_p_values():
    text = "The effect was significant (p < 0.001) and replicated (p = 0.02)."
    s = extract_signals(text)
    assert 0.001 in s.p_values_reported
    assert 0.02 in s.p_values_reported


def test_detects_spin_phrases():
    text = "This groundbreaking study represents a breakthrough in the field."
    s = extract_signals(text)
    assert "breakthrough" in s.spin_phrases
    assert "groundbreaking" in s.spin_phrases


def test_detects_observational_causal_overclaim():
    text = "This retrospective cohort shows that vitamin X causes reduced mortality. "
    text += "The intervention leads to fewer hospitalizations."
    s = extract_signals(text)
    assert s.study_type == "observational"
    assert s.observational_causal_language


def test_rct_with_causal_language_not_flagged():
    text = "This randomized controlled trial shows the drug causes symptom reduction."
    s = extract_signals(text)
    # RCTs can legitimately use causal language
    assert not s.observational_causal_language


def test_detects_animal_only():
    text = "We treated mice with compound X in a murine model."
    s = extract_signals(text)
    assert s.animal_or_invitro_only


def test_animal_human_mix_not_flagged():
    text = "After validating in mice, we tested compound X in 200 human participants."
    s = extract_signals(text)
    assert not s.animal_or_invitro_only


def test_counts_effect_sizes_and_cis():
    text = (
        "The odds ratio was 1.8 (95% CI 1.2-2.7). Secondary analysis showed a "
        "hazard ratio of 0.7 and Cohen's d of 0.4."
    )
    s = extract_signals(text)
    assert s.effect_sizes_reported >= 3
    assert s.confidence_intervals_reported >= 1


def test_detects_preregistration():
    text = "The protocol was prospectively registered on ClinicalTrials.gov."
    s = extract_signals(text)
    assert s.has_preregistration is True


def test_empty_text():
    s = extract_signals("")
    assert s.study_type == "unknown"
    assert s.sample_size is None
    assert s.p_values_reported == []


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
