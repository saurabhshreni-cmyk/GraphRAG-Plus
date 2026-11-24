"""Tests for the LLM client implementations and the selection order.

Focus: the Groq cloud client (added for keyless-Ollama deployments) and the
``build_default_llm_client`` preference order. Network is always mocked.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from graphrag_plus.app.generation import llm_clients as lc


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the object returned by urlopen (context manager)."""

    status = 200

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _mock_urlopen(payload: dict[str, Any]) -> Any:
    body = json.dumps(payload).encode("utf-8")

    def _open(req: Any, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(body)

    return _open


def test_groq_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        lc.GroqClient()


def test_groq_complete_parses_openai_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lc.urllib.request,
        "urlopen",
        _mock_urlopen({"choices": [{"message": {"content": "A knowledge graph links entities."}}]}),
    )
    client = lc.GroqClient(api_key="test-key")
    out = client.complete("what is a KG?", "1) context snippet")
    assert "knowledge graph" in out.lower()


def test_groq_empty_choices_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc.urllib.request, "urlopen", _mock_urlopen({"choices": []}))
    assert lc.GroqClient(api_key="k").complete("q", "ctx") == ""


def test_selection_prefers_groq_when_no_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    # No Ollama running; GROQ key present -> GroqClient wins over Anthropic/Echo.
    monkeypatch.setattr(lc, "_ollama_available", lambda *a, **k: False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = lc.build_default_llm_client(llm_enabled=True)
    assert isinstance(client, lc.GroqClient)


def test_selection_echo_when_nothing_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "_ollama_available", lambda *a, **k: False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(lc.build_default_llm_client(llm_enabled=True), lc.EchoClient)


def test_selection_none_when_disabled() -> None:
    assert lc.build_default_llm_client(llm_enabled=False) is None
