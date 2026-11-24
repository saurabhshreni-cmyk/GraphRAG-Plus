"""Tests for the Groq-backed reasoning verifier and backend selection.

The HTTP layer is stubbed, so these run without a key or network. What they
pin is the failure policy that makes verification safe to ship: whatever goes
wrong, the evidence-grounded draft is returned untouched with
``verified=False``.
"""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message

import pytest

from graphrag_plus.app.generation import reasoning_verifier as rv
from graphrag_plus.app.generation.reasoning_verifier import (
    GroqReasoningVerifier,
    ReasoningVerifier,
    build_reasoning_verifier,
)

EVIDENCE = [{"full_text": "Nova Dynamics acquired Orion Labs on 2024-01-15."}]
DRAFT = "Nova Dynamics acquired Orion Labs."


class _Response:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _groq_payload(content: str, reasoning: str = "") -> dict:
    return {"choices": [{"message": {"content": content, "reasoning": reasoning}}]}


@pytest.fixture
def captured(monkeypatch):
    """Patch urlopen; tests set ``captured['reply']`` to a payload or exception."""
    state: dict = {"requests": [], "reply": None}

    def fake_urlopen(request, timeout):
        state["requests"].append(json.loads(request.data.decode("utf-8")))
        reply = state["reply"]
        if isinstance(reply, Exception):
            raise reply
        return _Response(reply)

    monkeypatch.setattr(rv.urllib.request, "urlopen", fake_urlopen)
    return state


class TestGroqReasoningVerifier:
    def test_unavailable_without_a_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        verifier = GroqReasoningVerifier()
        assert not verifier.available()
        result = verifier.verify("q", DRAFT, EVIDENCE)
        assert result.final_answer == DRAFT
        assert not result.verified

    def test_confirms_a_correct_draft(self, captured):
        captured["reply"] = _groq_payload(
            f"The evidence states the acquisition.\nFINAL VERIFIED ANSWER: {DRAFT}",
            reasoning="Check the date.\nCheck the parties.\nBoth match the evidence.",
        )
        result = GroqReasoningVerifier(api_key="k").verify("Who acquired Orion?", DRAFT, EVIDENCE)
        assert result.verified
        assert not result.changed
        assert result.final_answer == DRAFT

    def test_reports_a_correction(self, captured):
        captured["reply"] = _groq_payload(
            "FINAL VERIFIED ANSWER: Nova Dynamics acquired Orion Labs on 2024-01-15."
        )
        result = GroqReasoningVerifier(api_key="k").verify("When?", DRAFT, EVIDENCE)
        assert result.verified
        assert result.changed
        assert "2024-01-15" in result.final_answer

    def test_surfaces_the_reasoning_trace(self, captured):
        # The trace is what fills the UI's "what the reasoning model checked"
        # panel; dropping it would make verification an opaque rewrite.
        captured["reply"] = _groq_payload(
            f"FINAL VERIFIED ANSWER: {DRAFT}",
            reasoning="First, find the acquirer.\nSecond, confirm the target.\nConclusion: supported.",
        )
        result = GroqReasoningVerifier(api_key="k").verify("q", DRAFT, EVIDENCE)
        assert "find the acquirer" in result.reasoning_summary
        assert "supported" in result.reasoning_summary

    def test_sends_reasoning_parameters(self, captured):
        captured["reply"] = _groq_payload(f"FINAL VERIFIED ANSWER: {DRAFT}")
        GroqReasoningVerifier(api_key="k").verify("q", DRAFT, EVIDENCE)
        body = captured["requests"][0]
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["reasoning_effort"] == "high"
        assert body["include_reasoning"] is True
        # Groq rejects include_reasoning combined with reasoning_format (400).
        assert "reasoning_format" not in body

    def test_http_error_returns_the_draft(self, captured):
        captured["reply"] = urllib.error.HTTPError(
            "u", 429, "Too Many Requests", Message(), io.BytesIO(b"rate limited")
        )
        result = GroqReasoningVerifier(api_key="k").verify("q", DRAFT, EVIDENCE)
        assert result.final_answer == DRAFT
        assert not result.verified
        assert "429" in result.reasoning_summary

    def test_timeout_returns_the_draft(self, captured):
        captured["reply"] = TimeoutError("slow")
        result = GroqReasoningVerifier(api_key="k").verify("q", DRAFT, EVIDENCE)
        assert result.final_answer == DRAFT
        assert not result.verified

    def test_missing_marker_uses_the_substantive_content(self, captured):
        captured["reply"] = _groq_payload("Nova Dynamics acquired Orion Labs in January 2024.")
        result = GroqReasoningVerifier(api_key="k").verify("q", DRAFT, EVIDENCE)
        assert result.verified
        assert "January 2024" in result.final_answer

    def test_empty_content_returns_the_draft(self, captured):
        captured["reply"] = _groq_payload("")
        result = GroqReasoningVerifier(api_key="k").verify("q", DRAFT, EVIDENCE)
        assert result.final_answer == DRAFT
        assert not result.verified

    def test_no_choices_returns_the_draft(self, captured):
        captured["reply"] = {"choices": []}
        result = GroqReasoningVerifier(api_key="k").verify("q", DRAFT, EVIDENCE)
        assert result.final_answer == DRAFT
        assert not result.verified

    def test_skips_without_evidence(self, captured):
        result = GroqReasoningVerifier(api_key="k").verify("q", DRAFT, [])
        assert not result.verified
        assert captured["requests"] == []


class TestBuildReasoningVerifier:
    def test_prefers_local_ollama(self, monkeypatch):
        monkeypatch.setattr(ReasoningVerifier, "available", lambda self: True)
        monkeypatch.setenv("GROQ_API_KEY", "k")
        assert isinstance(build_reasoning_verifier(), ReasoningVerifier)

    def test_falls_back_to_groq(self, monkeypatch):
        monkeypatch.setattr(ReasoningVerifier, "available", lambda self: False)
        monkeypatch.setenv("GROQ_API_KEY", "k")
        assert isinstance(build_reasoning_verifier(), GroqReasoningVerifier)

    def test_returns_an_unavailable_verifier_when_nothing_is_configured(self, monkeypatch):
        monkeypatch.setattr(ReasoningVerifier, "available", lambda self: False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert not build_reasoning_verifier().available()
