"""
Quick-start example: triage a paper with obvious methodology problems.

Watch the deterministic validator downgrade the verdict automatically —
small sample, spin phrases, observational-causal overclaim.

Requires ANTHROPIC_API_KEY in your environment.
Run from the project root:  python examples/try_weak_paper.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import PaperMeta, default_caller, triage_paper, render_markdown


meta = PaperMeta(
    title="Lemon water weight loss breakthrough",
    venue="Wellness Blog",
    year=2024,
)

text = """
This revolutionary, groundbreaking study proves that drinking lemon water
causes dramatic weight loss. We observed 8 patients in our clinic
retrospectively. Methods: We reviewed charts. Drinking lemon water leads
to reduced appetite and triggers metabolic changes that cause fat loss.
This miracle finding represents an unprecedented breakthrough.
"""

if __name__ == "__main__":
    result = triage_paper(meta, text, default_caller())
    print(render_markdown(result))
