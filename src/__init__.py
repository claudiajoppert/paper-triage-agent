"""Paper Triage Agent — an agentic AI system for evaluating research credibility."""
from .orchestrator import triage_paper
from .schema import (
    AgentOpinion,
    HardFail,
    HardFailResult,
    PaperMeta,
    StudySignals,
    TriageResult,
    Verdict,
)
from .llm import AnthropicCaller, StubCaller, default_caller
from .report import render_markdown

__all__ = [
    "triage_paper",
    "AgentOpinion",
    "HardFail",
    "HardFailResult",
    "PaperMeta",
    "StudySignals",
    "TriageResult",
    "Verdict",
    "AnthropicCaller",
    "StubCaller",
    "default_caller",
    "render_markdown",
]
