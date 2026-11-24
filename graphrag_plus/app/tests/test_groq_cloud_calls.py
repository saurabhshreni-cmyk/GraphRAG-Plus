"""Regression tests for the two bugs that kept Groq unused on the cloud deploy.

1. Every Groq call was rejected by Cloudflare (HTTP 403, "error code: 1010")
   because urllib sent its default ``Python-urllib`` User-Agent.
2. The extraction stage only knew Ollama, so with no Ollama in the cloud the
   graph never got typed relations.

The HTTP layer is stubbed; no key or network is needed.
"""

from __future__ import annotations

import json
import sys

import pytest

from graphrag_plus.app.embeddings import remote_embedder
from graphrag_plus.app.extraction import extractor, groq_extraction
from graphrag_plus.app.extraction.groq_extraction import (
    GroqExtractionClient,
    build_groq_extraction_client,
)
from graphrag_plus.app.generation import llm_clients, reasoning_verifier
from graphrag_plus.app.generation.llm_clients import GroqClient
from graphrag_plus.app.generation.reasoning_verifier import GroqReasoningVerifier
from graphrag_plus.app.utils.http_utils import USER_AGENT, json_headers


class _Response:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _chat_reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _recorder(monkeypatch, module, reply: dict) -> list:
    """Patch ``module.urllib.request.urlopen``; return the captured requests."""
    seen: list = []

    def fake_urlopen(request, timeout):
        seen.append(request)
        return _Response(reply)

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    return seen


def _user_agent(request) -> str:
    # urllib normalises header names with str.capitalize().
    return request.get_header("User-agent") or ""


class TestUserAgent:
    def test_json_headers_carry_a_non_default_user_agent(self):
        headers = json_headers(authorization="Bearer k")
        assert headers["user-agent"] == USER_AGENT
        assert "python-urllib" not in USER_AGENT.lower()
        assert headers["authorization"] == "Bearer k"

    def test_groq_answer_client_sends_it(self, monkeypatch):
        seen = _recorder(monkeypatch, llm_clients, _chat_reply("Priya Raman founded it."))
        GroqClient(api_key="k").complete("Who founded it?", "[1] Priya Raman founded it.")
        assert _user_agent(seen[0]) == USER_AGENT

    def test_groq_verifier_sends_it(self, monkeypatch):
        seen = _recorder(monkeypatch, reasoning_verifier, _chat_reply("VERDICT: SUPPORTED"))
        GroqReasoningVerifier(api_key="k").verify("q?", "draft", [{"full_text": "evidence"}])
        assert _user_agent(seen[0]) == USER_AGENT

    def test_remote_embedder_sends_it(self, monkeypatch):
        seen = _recorder(monkeypatch, remote_embedder, {"ok": True})
        remote_embedder._post_json("https://example.invalid/embed", {}, {"x-api-key": "k"}, 5.0)
        assert _user_agent(seen[0]) == USER_AGENT
        assert seen[0].get_header("X-api-key") == "k"


class TestGroqExtractionClient:
    def test_requests_json_and_returns_the_ollama_shape(self, monkeypatch):
        content = '{"entities": [], "relationships": []}'
        seen = _recorder(monkeypatch, groq_extraction, _chat_reply(content))
        reply = GroqExtractionClient("k", model="m").chat(
            model="ignored", messages=[{"role": "user", "content": "text"}], format="json", think=False
        )
        assert reply == {"message": {"content": content}}
        body = json.loads(seen[0].data.decode("utf-8"))
        assert body["model"] == "m"
        assert body["response_format"] == {"type": "json_object"}
        assert _user_agent(seen[0]) == USER_AGENT

    def test_no_choices_raises_so_the_extractor_retries(self, monkeypatch):
        _recorder(monkeypatch, groq_extraction, {"choices": []})
        with pytest.raises(ValueError):
            GroqExtractionClient("k").chat(messages=[])

    def test_model_comes_from_env_when_not_given(self, monkeypatch):
        monkeypatch.delenv("GROQ_EXTRACTION_MODEL", raising=False)
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
        assert GroqExtractionClient("k").model == "openai/gpt-oss-120b"

    def test_builder_needs_a_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert build_groq_extraction_client() is None
        monkeypatch.setenv("GROQ_API_KEY", "k")
        assert isinstance(build_groq_extraction_client(), GroqExtractionClient)


@pytest.fixture
def no_ollama(monkeypatch):
    """Simulate the cloud: the ollama SDK is not installed."""
    monkeypatch.setitem(sys.modules, "ollama", None)
    monkeypatch.setattr(extractor, "_ollama_client", None)


class TestExtractorBackendSelection:
    def test_falls_back_to_groq_without_ollama(self, no_ollama, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "k")
        assert isinstance(extractor._get_ollama_client(), GroqExtractionClient)

    def test_no_backend_without_ollama_or_key(self, no_ollama, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert extractor._get_ollama_client() is None

    def test_groq_relations_reach_the_extraction_result(self, no_ollama, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "k")
        payload = {
            "entities": [
                {"name": "Nova Dynamics", "type": "ORG", "confidence": 0.95},
                {"name": "Helix Sensors", "type": "ORG", "confidence": 0.95},
            ],
            "relationships": [
                {
                    "source": "Nova Dynamics",
                    "target": "Helix Sensors",
                    "relation": "acquired",
                    "confidence": 0.9,
                }
            ],
        }
        _recorder(monkeypatch, groq_extraction, _chat_reply(json.dumps(payload)))
        result = extractor.extract("Nova Dynamics acquired Helix Sensors in 2021.", "c1", use_llm=True)
        relations = {(r.source, r.relation, r.target) for r in result.relationships}
        assert ("Nova Dynamics", "acquired", "Helix Sensors") in relations


class TestGroqAnswerBudget:
    """gpt-oss spends hidden reasoning tokens out of max_tokens. With 400 and
    default effort the answer was often empty, silently forcing extractive."""

    def test_requests_low_effort_and_a_roomy_budget(self, monkeypatch):
        seen = _recorder(monkeypatch, llm_clients, _chat_reply("Priya Raman founded it."))
        GroqClient(api_key="k").complete("Who founded it?", "[1] Priya Raman founded it.")
        body = json.loads(seen[0].data.decode("utf-8"))
        assert body["reasoning_effort"] == "low"
        assert body["include_reasoning"] is False
        assert body["max_tokens"] >= 1024

    def test_empty_content_is_logged_not_silent(self, monkeypatch, caplog):
        reply = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        _recorder(monkeypatch, llm_clients, reply)
        with caplog.at_level("WARNING"):
            assert GroqClient(api_key="k").complete("q?", "[1] ctx") == ""
        assert any(
            "groq.empty_content" in r.getMessage() and "length" in r.getMessage() for r in caplog.records
        )

    def test_verifier_budget_leaves_room_for_high_effort(self, monkeypatch):
        seen = _recorder(monkeypatch, reasoning_verifier, _chat_reply("VERDICT: SUPPORTED"))
        GroqReasoningVerifier(api_key="k").verify("q?", "draft", [{"full_text": "evidence"}])
        assert json.loads(seen[0].data.decode("utf-8"))["max_tokens"] >= 4096
