# Paper Triage Agent

An agentic AI system that evaluates the credibility of research papers by combining deterministic rule-based checks with multi-agent AI debate.

## What it does

Given a paper's metadata and text, the system produces a verdict — `CREDIBLE`, `PROMISING`, `WEAK`, or `UNRELIABLE` — along with a plain-English summary, the specific reasoning from three reviewers, and concrete follow-up questions.

It's meant as a triage tool for anyone consuming research: researchers scanning preprints, journalists vetting claims, clinicians filtering new evidence, or just anyone evaluating a wellness article on the internet.

## Why this architecture

Most "AI reviewer" tools are a single prompt to a single LLM. That has two problems: the AI can hallucinate facts about the paper, and it tends toward sycophancy — saying nice things about whatever you give it. This project fixes both by wrapping the AI in deterministic layers:

- **Deterministic checks run first.** Regex pulls out study type, sample size, p-values, and hype language before any AI sees the paper. If the paper is retracted, from a predatory venue, or lacks a methods section, it's rejected without calling the AI at all.
- **Three AI agents argue from different angles.** A Defender steelmans the paper, a Skeptic attacks it, and a Methodologist evaluates design quality independently of the topic. They each see the same deterministic evidence block.
- **A Referee synthesizes, then a validator double-checks.** The referee weighs the three opinions into a final verdict. Deterministic rules can still override — if the AI says `CREDIBLE` but the sample size is 8, the validator downgrades to `PROMISING` automatically.

The AI proposes. The rules dispose.

## Architecture

```
Paper text + metadata
        │
        ▼
┌──────────────────────┐
│  Signal extractor    │  pure Python regex — study type, n, p-values, spin phrases
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Hard-fail gatekeeper│  retracted? predatory? no methods? → reject without AI
└──────────┬───────────┘
           │ pass
           ▼
    ┌──────┴──────┐
    │ parallel    │
    ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌────────────────┐
│Defender│  │Skeptic │  │ Methodologist  │   three AI agents, same evidence
└────┬───┘  └────┬───┘  └────────┬───────┘
     └───────────┼───────────────┘
                 ▼
        ┌────────────────┐
        │    Referee     │     AI synthesizes final verdict
        └────────┬───────┘
                 ▼
        ┌────────────────┐
        │   Validator    │     deterministic override if AI verdict
        └────────┬───────┘     contradicts the signals
                 ▼
          Final report
```

## Quick start

Requires Python 3.10+.

```bash
# clone and enter
git clone https://github.com/YOUR-USERNAME/paper-triage-agent.git
cd paper-triage-agent

# install Anthropic SDK (only runtime dep)
pip install anthropic

# set your API key (from console.anthropic.com)
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# run the example
python examples/try_strong_paper.py
```

You'll see a markdown report with deterministic signals, three reviewer opinions, and a final synthesized verdict.

### Run the test suite (no API key needed)

```bash
python tests/test_extractors.py
python tests/test_gatekeeper_and_validator.py
```

These exercise the entire deterministic layer — extractors, gatekeeper, validator — with zero external calls.

## Usage in your own code

```python
from src import PaperMeta, default_caller, triage_paper, render_markdown

meta = PaperMeta(
    title="Effect of Intervention X on Outcome Y",
    venue="NEJM",
    year=2024,
)
text = """
This randomized controlled trial enrolled 1,240 patients across 12 sites.
Methods: We conducted a double-blind, placebo-controlled RCT. Protocol was
prospectively registered on ClinicalTrials.gov. Results: Intervention X
reduced the primary outcome (hazard ratio 0.72, 95% CI 0.61-0.84, p < 0.001).
"""

result = triage_paper(meta, text, default_caller())
print(render_markdown(result))
```

## What each module does

| Module | Responsibility | AI? |
|---|---|---|
| `src/schema.py` | Typed dataclasses everything passes through | No |
| `src/extractors.py` | Regex-based signal extraction from paper text | No |
| `src/gatekeeper.py` | Hard-fail rules — retracted, predatory, no methods | No |
| `src/agents.py` | Defender, Skeptic, Methodologist, Referee prompts | Yes |
| `src/validator.py` | Deterministic override of the AI's verdict | No |
| `src/orchestrator.py` | Pipeline wiring — extract → gate → fan out → referee → validate | No |
| `src/llm.py` | Anthropic API adapter + scriptable stub for tests | — |
| `src/report.py` | Markdown renderer for the final result | No |

## Design choices worth highlighting

**Parallel fan-out.** The three opinion agents run concurrently via `ThreadPoolExecutor`. On Anthropic's API this cuts end-to-end time by roughly two-thirds.

**Structured JSON outputs.** Every AI response is parsed into a typed dataclass. The orchestrator never does string-munging on LLM prose. If the AI returns malformed JSON, a safe default is used and the run continues.

**The validator only downgrades.** If the AI says `WEAK`, the validator can't promote it to `CREDIBLE`. This is deliberately asymmetric — we'd rather miss a good paper than endorse a bad one.

**Hard-fails skip the AI entirely.** A retracted paper triggers a short rejection report without any API calls. Saves money, and makes the rejection reasoning fully auditable.

## Limitations

- **No paper fetching yet.** You have to supply the text and metadata yourself. A PubMed / arXiv / DOI fetcher is an obvious next step.
- **Predatory-venue list is tiny.** Real deployment would pull from a maintained source like Cabells.
- **English-language regex.** The extractors assume English paper text.
- **No ground-truth evaluation.** There's no benchmark yet that scores how often the system agrees with expert peer reviewers.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

This is a research and triage tool. It does not replace expert peer review, and its verdicts should not be treated as authoritative. Use it to generate starting points for deeper reading, not final judgments.
