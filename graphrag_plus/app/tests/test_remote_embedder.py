"""Tests for the hosted embedding backends.

Every test stubs the HTTP layer - nothing here touches the network or needs an
API key, so CI runs them unconditionally. The behaviour being pinned is the
part that matters in production: a provider failure must degrade retrieval to
BM25 + TF-IDF rather than propagate, and a malformed response must never reach
the FAISS index.
"""

from __future__ import annotations

import json
import urllib.error
from email.message import Message

import pytest

from graphrag_plus.app.embeddings import remote_embedder as re_mod
from graphrag_plus.app.embeddings.remote_embedder import (
    REMOTE_EMBEDDING_DIM,
    GeminiEmbedder,
    JinaEmbedder,
    RemoteEmbedder,
    build_remote_embedder,
)

_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "JINA_API_KEY")


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch):
    """Start every test with no provider configured."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def _vector(fill: float = 0.1) -> list[float]:
    return [fill] * REMOTE_EMBEDDING_DIM


class _Recorder:
    """Stand-in for ``_post_json`` that records calls and replays responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append({"url": url, "payload": payload, "headers": headers})
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _gemini_response(count: int):
    return {"embeddings": [{"values": _vector()} for _ in range(count)]}


def _jina_response(count: int, *, shuffle: bool = False):
    rows = [{"index": i, "embedding": _vector(float(i) / 100)} for i in range(count)]
    if shuffle:
        rows.reverse()
    return {"data": rows}


class TestGeminiEmbedder:
    def test_requires_a_key(self):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiEmbedder()

    def test_accepts_google_api_key_as_an_alias(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        assert GeminiEmbedder().dimension == REMOTE_EMBEDDING_DIM

    def test_sends_one_content_per_text(self, monkeypatch):
        recorder = _Recorder([_gemini_response(3)])
        monkeypatch.setattr(re_mod, "_post_json", recorder)
        embedder = GeminiEmbedder(api_key="k")

        vectors = embedder.embed_batch(["a", "b", "c"])

        assert len(vectors) == 3
        requests = recorder.calls[0]["payload"]["requests"]
        # One Content per text. Batching several parts into one Content would
        # return a single aggregated vector and silently corrupt the index.
        assert len(requests) == 3
        assert [r["content"]["parts"][0]["text"] for r in requests] == ["a", "b", "c"]

    def test_requests_the_target_dimension(self, monkeypatch):
        recorder = _Recorder([_gemini_response(1)])
        monkeypatch.setattr(re_mod, "_post_json", recorder)

        GeminiEmbedder(api_key="k").embed_batch(["x"])

        assert recorder.calls[0]["payload"]["requests"][0]["outputDimensionality"] == REMOTE_EMBEDDING_DIM

    def test_uses_an_asymmetric_task_type(self, monkeypatch):
        recorder = _Recorder([_gemini_response(1), _gemini_response(1)])
        monkeypatch.setattr(re_mod, "_post_json", recorder)
        embedder = GeminiEmbedder(api_key="k")

        embedder.embed_batch(["doc"], is_query=False)
        embedder.embed_batch(["q"], is_query=True)

        assert recorder.calls[0]["payload"]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
        assert recorder.calls[1]["payload"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"

    def test_rejects_a_count_mismatch(self, monkeypatch):
        monkeypatch.setattr(re_mod, "_post_json", _Recorder([_gemini_response(2)]))
        with pytest.raises(ValueError, match="different number of embeddings"):
            GeminiEmbedder(api_key="k").embed_batch(["a", "b", "c"])

    def test_splits_large_input_into_several_requests(self, monkeypatch):
        texts = [f"t{i}" for i in range(re_mod._MAX_BATCH + 5)]
        recorder = _Recorder([_gemini_response(re_mod._MAX_BATCH), _gemini_response(5)])
        monkeypatch.setattr(re_mod, "_post_json", recorder)

        vectors = GeminiEmbedder(api_key="k").embed_batch(texts)

        assert len(vectors) == len(texts)
        assert len(recorder.calls) == 2

    def test_empty_input_makes_no_request(self, monkeypatch):
        recorder = _Recorder([])
        monkeypatch.setattr(re_mod, "_post_json", recorder)
        assert GeminiEmbedder(api_key="k").embed_batch([]) == []
        assert recorder.calls == []


class TestJinaEmbedder:
    def test_requires_a_key(self):
        with pytest.raises(ValueError, match="JINA_API_KEY"):
            JinaEmbedder()

    def test_sends_a_bearer_token_and_the_whole_batch(self, monkeypatch):
        recorder = _Recorder([_jina_response(2)])
        monkeypatch.setattr(re_mod, "_post_json", recorder)

        JinaEmbedder(api_key="k").embed_batch(["a", "b"])

        assert recorder.calls[0]["headers"]["authorization"] == "Bearer k"
        assert recorder.calls[0]["payload"]["input"] == ["a", "b"]
        assert recorder.calls[0]["payload"]["dimensions"] == REMOTE_EMBEDDING_DIM

    def test_reorders_rows_by_index(self, monkeypatch):
        # The API may return rows out of order; `index` is authoritative, and
        # trusting arrival order would pair vectors with the wrong chunks.
        monkeypatch.setattr(re_mod, "_post_json", _Recorder([_jina_response(3, shuffle=True)]))

        vectors = JinaEmbedder(api_key="k").embed_batch(["a", "b", "c"])

        assert vectors[0][0] == pytest.approx(0.0)
        assert vectors[1][0] == pytest.approx(0.01)
        assert vectors[2][0] == pytest.approx(0.02)

    def test_uses_an_asymmetric_task_hint(self, monkeypatch):
        recorder = _Recorder([_jina_response(1), _jina_response(1)])
        monkeypatch.setattr(re_mod, "_post_json", recorder)
        embedder = JinaEmbedder(api_key="k")

        embedder.embed_batch(["doc"], is_query=False)
        embedder.embed_batch(["q"], is_query=True)

        assert recorder.calls[0]["payload"]["task"] == "retrieval.passage"
        assert recorder.calls[1]["payload"]["task"] == "retrieval.query"

    def test_rejects_a_count_mismatch(self, monkeypatch):
        monkeypatch.setattr(re_mod, "_post_json", _Recorder([_jina_response(1)]))
        with pytest.raises(ValueError, match="different number of embeddings"):
            JinaEmbedder(api_key="k").embed_batch(["a", "b"])


class _FakeProvider:
    """Minimal provider for exercising RemoteEmbedder's own policy."""

    name = "fake:test"
    dimension = REMOTE_EMBEDDING_DIM

    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.call_count = 0

    def embed_batch(self, texts, *, is_query=False):
        self.call_count += 1
        if isinstance(self._behaviour, Exception):
            raise self._behaviour
        return self._behaviour(texts, is_query)


class TestRemoteEmbedder:
    def test_reports_the_provider_identity(self):
        embedder = RemoteEmbedder(_FakeProvider(lambda t, q: [_vector() for _ in t]))
        assert embedder.model_name == "fake:test"
        assert embedder.available()
        assert embedder.dimension == REMOTE_EMBEDDING_DIM

    def test_embeds_text_and_query(self):
        seen: list[bool] = []

        def behaviour(texts, is_query):
            seen.append(is_query)
            return [_vector() for _ in texts]

        embedder = RemoteEmbedder(_FakeProvider(behaviour))
        assert len(embedder.embed_text("doc")) == REMOTE_EMBEDDING_DIM
        assert len(embedder.embed_query("q")) == REMOTE_EMBEDDING_DIM
        assert seen == [False, True]

    def test_a_provider_error_degrades_instead_of_raising(self):
        embedder = RemoteEmbedder(_FakeProvider(urllib.error.URLError("boom")))
        assert embedder.embed_batch(["a"]) == []
        assert not embedder.available()
        assert embedder.dimension == 0

    def test_failure_disables_the_provider_for_the_process(self):
        provider = _FakeProvider(urllib.error.URLError("boom"))
        embedder = RemoteEmbedder(provider)

        for _ in range(5):
            assert embedder.embed_batch(["a"]) == []

        # One timeout total, not one per query: a dead key or an exhausted
        # quota must not add a network round trip to every request.
        assert provider.call_count == 1

    def test_mismatched_vector_count_disables_the_provider(self):
        embedder = RemoteEmbedder(_FakeProvider(lambda t, q: [_vector()]))
        assert embedder.embed_batch(["a", "b"]) == []
        assert not embedder.available()

    def test_empty_vectors_disable_the_provider(self):
        embedder = RemoteEmbedder(_FakeProvider(lambda t, q: [[] for _ in t]))
        assert embedder.embed_batch(["a"]) == []
        assert not embedder.available()

    def test_empty_input_short_circuits(self):
        provider = _FakeProvider(lambda t, q: [])
        embedder = RemoteEmbedder(provider)
        assert embedder.embed_batch([]) == []
        assert provider.call_count == 0
        assert embedder.available()


class TestBuildRemoteEmbedder:
    def test_returns_none_without_any_key(self):
        assert build_remote_embedder() is None

    def test_prefers_gemini(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g")
        monkeypatch.setenv("JINA_API_KEY", "j")
        embedder = build_remote_embedder()
        assert embedder is not None
        assert embedder.model_name.startswith("gemini:")

    def test_falls_back_to_jina(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "j")
        embedder = build_remote_embedder()
        assert embedder is not None
        assert embedder.model_name.startswith("jina:")


class TestPostJson:
    def test_http_error_propagates(self, monkeypatch):
        def raise_http(*_args, **_kwargs):
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", Message(), None)

        monkeypatch.setattr(re_mod.urllib.request, "urlopen", raise_http)
        with pytest.raises(urllib.error.HTTPError):
            re_mod._post_json("https://example.invalid", {}, {}, 1.0)

    def test_decodes_a_successful_body(self, monkeypatch):
        class _Response:
            def read(self):
                return json.dumps({"ok": True}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        monkeypatch.setattr(re_mod.urllib.request, "urlopen", lambda *a, **k: _Response())
        assert re_mod._post_json("https://example.invalid", {}, {}, 1.0) == {"ok": True}
