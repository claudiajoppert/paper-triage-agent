"""
Quick-start example: triage a well-designed RCT.

Requires ANTHROPIC_API_KEY in your environment.
Run from the project root:  python examples/try_strong_paper.py
"""
import sys
import os

# make `src` importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import PaperMeta, default_caller, triage_paper, render_markdown


meta = PaperMeta(
    title="Effect of Intervention X on Outcome Y",
    venue="NEJM",
    year=2024,
)

text = """
This randomized controlled trial enrolled 1,240 patients across 12 sites.
Methods: We conducted a double-blind, placebo-controlled RCT. Protocol was
prospectively registered on ClinicalTrials.gov. Primary outcome analyzed by
intention-to-treat.

Results: Intervention X reduced the primary outcome (hazard ratio 0.72,
95% CI 0.61-0.84, p < 0.001). Effect size (Cohen's d) was 0.35.

Conclusion: Intervention X meaningfully reduced outcome Y in this population.
Conflicts of interest: None declared. Funding: NIH grant R01-XXXXXX.
"""

if __name__ == "__main__":
    result = triage_paper(meta, text, default_caller())
    print(render_markdown(result))
