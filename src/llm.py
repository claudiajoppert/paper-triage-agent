"""
LLM adapters.

Two concrete implementations:
  - AnthropicCaller:  real calls to the Anthropic API (requires ANTHROPIC_API_KEY env var)
  - StubCaller:       scriptable fake for tests and examples

Both satisfy the LLMCaller protocol from agents.py.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Optional


class AnthropicCaller:
    """
    Production LLM caller. Uses the Anthropic Python SDK if installed.

    Example:
        caller = AnthropicCaller(model="claude-opus-4-7")
        result = triage_paper(meta, text, caller)
    """

    def __init__(self, model: str = "claude-opus-4-7", max_tokens: int = 1024):
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env
        self._model = model
        self._max_tokens = max_tokens

    def __call__(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # concatenate all text blocks
        return "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )


class StubCaller:
    """
    Scriptable fake LLM for tests and demos.

    You can either:
      - pass a dict of {role_hint: json_response_str} and it will match on the
        system prompt to pick the right canned response, OR
      - pass a callable router (system, user) -> str for full control.
    """

    def __init__(
        self,
        responses: Optional[dict[str, str]] = None,
        router: Optional[Callable[[str, str], str]] = None,
    ):
        self._responses = responses or {}
        self._router = router
        self.calls: list[tuple[str, str]] = []   # captured for assertions

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self._router is not None:
            return self._router(system, user)
        # match by a keyword in the system prompt
        for hint, response in self._responses.items():
            if hint.lower() in system.lower():
                return response
        # default: a valid-shaped but uninformative response
        return json.dumps({
            "verdict": "WEAK",
            "confidence": 0.5,
            "strongest_points": ["stub: no specific response configured"],
            "weakest_points": ["stub: no specific response configured"],
            "unresolved_questions": [],
            "summary": "Stub caller produced default response.",
            "caveats": [],
            "follow_ups": [],
        })


def default_caller() -> AnthropicCaller:
    """Convenience factory. Fails loudly if the API key isn't set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Either set it, or construct a StubCaller for offline use."
        )
    return AnthropicCaller()
