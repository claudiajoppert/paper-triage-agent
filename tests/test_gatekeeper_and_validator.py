"""Tests for the gatekeeper and validator — the 'check' layers."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gatekeeper import check_hard_fails
from src.schema import HardFail, PaperMeta, StudySignals, Verdict
from src.validator import validate_and_adjust


# --- gatekeeper -----------------------------------------------------------

def test_gatekeeper_passes_normal_paper():
    meta = PaperMeta(title="A real paper", venue="NEJM")
    signals = StudySignals(study_type="rct", sample_size=500,
                           p_values_reported=[0.01])
    text = "Methods: we enrolled 500 patients..."
    result = check_hard_fails(meta, signals, text)
    assert not result.failed


def test_gatekeeper_catches_retracted():
    meta = PaperMeta(title="Retracted paper", is_retracted=True,
                     retraction_notes="data fabrication")
    signals = StudySignals(study_type="rct", sample_size=500)
    result = check_hard_fails(meta, signals, "methods: blah")
    assert result.failed
    assert HardFail.RETRACTED in result.reasons


def test_gatekeeper_catches_predatory_venue():
    meta = PaperMeta(title="x", venue="OMICS Publishing Group")
    signals = StudySignals(study_type="rct", sample_size=100)
    result = check_hard_fails(meta, signals, "methods: blah")
    assert result.failed
    assert HardFail.PREDATORY_VENUE in result.reasons


def test_gatekeeper_catches_missing_methods():
    meta = PaperMeta(title="x", venue="Nature", full_text_available=True)
    signals = StudySignals()
    text = "Our findings are amazing. We did stuff. The end."  # no methods section
    result = check_hard_fails(meta, signals, text)
    assert result.failed
    assert HardFail.NO_METHODS in result.reasons


def test_gatekeeper_catches_not_a_paper():
    meta = PaperMeta(title="Press release")
    signals = StudySignals()  # nothing extracted
    text = "Short blurb."
    result = check_hard_fails(meta, signals, text)
    assert result.failed
    assert HardFail.NOT_A_PAPER in result.reasons


# --- validator ------------------------------------------------------------

def test_validator_no_op_when_evidence_solid():
    signals = StudySignals(study_type="rct", sample_size=500,
                           p_values_reported=[0.01], confidence_intervals_reported=3,
                           effect_sizes_reported=2)
    v, c, notes = validate_and_adjust(Verdict.CREDIBLE, 0.85, signals)
    assert v == Verdict.CREDIBLE
    assert c == 0.85
    assert notes == []


def test_validator_downgrades_small_sample():
    signals = StudySignals(study_type="rct", sample_size=12)
    v, c, notes = validate_and_adjust(Verdict.CREDIBLE, 0.9, signals)
    assert v == Verdict.PROMISING
    assert c <= 0.6
    assert any("sample size" in n.lower() for n in notes)


def test_validator_downgrades_case_report():
    signals = StudySignals(study_type="case-report", sample_size=3)
    v, c, notes = validate_and_adjust(Verdict.CREDIBLE, 0.9, signals)
    assert v == Verdict.PROMISING


def test_validator_downgrades_animal_only():
    signals = StudySignals(study_type="rct", sample_size=60,
                           animal_or_invitro_only=True)
    v, c, notes = validate_and_adjust(Verdict.CREDIBLE, 0.8, signals)
    assert v == Verdict.PROMISING


def test_validator_downgrades_spin_heavy():
    signals = StudySignals(study_type="rct", sample_size=400,
                           spin_phrases=["breakthrough", "revolutionary", "miracle"])
    v, c, notes = validate_and_adjust(Verdict.CREDIBLE, 0.85, signals)
    assert v == Verdict.PROMISING
    assert any("spin" in n.lower() for n in notes)


def test_validator_downgrades_observational_causal_hard():
    signals = StudySignals(study_type="observational", sample_size=5000,
                           observational_causal_language=True)
    v, c, notes = validate_and_adjust(Verdict.CREDIBLE, 0.8, signals)
    # this is the overclaim pattern — big sample doesn't save it
    assert v == Verdict.WEAK


def test_validator_never_upgrades():
    """Validator should never turn WEAK into CREDIBLE."""
    signals = StudySignals(study_type="rct", sample_size=1000)
    v, c, notes = validate_and_adjust(Verdict.WEAK, 0.4, signals)
    assert v == Verdict.WEAK


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
