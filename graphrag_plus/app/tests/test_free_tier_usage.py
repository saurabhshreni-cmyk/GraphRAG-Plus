"""Keep embedding-API calls low on a free-tier key.

1. Every new serverless instance used to re-embed each corpus it touched,
   because only meta/graph/chunks were stored durably. The FAISS index now
   travels with the corpus, so a second instance reuses it.
2. One transient error (a 429 rate limit) switched the remote embedder off
   for the rest of the process; now only a rejected key does that.
3. Repeated questions re-embedded the same query text every time.
"""

from __future__ import annotations

import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from graphrag_plus.app.corpus.blob_store import InMemoryBlobStore
from graphrag_plus.app.corpus.manager import CorpusManager
from graphrag_plus.app.embeddings.remote_embedder import RemoteEmbedder
from graphrag_plus.app.ingestion.models import Chunk
from graphrag_plus.app.retrieval import service as retrieval_service

_DIM = 8


class _CountingEmbedder:
    """Deterministic fake embedder that counts passage-embedding calls."""

    model_name = "fake:counting"
    dimension = _DIM

    def __init__(self) -> None:
        self.passages_embedded = 0

    def available(self) -> bool:
        return True

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.passages_embedded += len(texts)
        return [[float((len(t) + i) % 7 + 1) for i in range(_DIM)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


def _chunks() -> list[Chunk]:
    texts = ["Nova Dynamics acquired Helix Sensors.", "Helix Sensors is based in Boulder."]
    return [
        Chunk(chunk_id=f"c{i}", doc_id="d", text=t, start=0, end=len(t), timestamp=None)
        for i, t in enumerate(texts)
    ]


class TestVectorsTravelWithTheCorpus:
    def test_a_second_instance_reuses_the_stored_index(self, tmp_path: Path, monkeypatch):
        embedder = _CountingEmbedder()
        monkeypatch.setattr(retrieval_service, "get_embedder", lambda: embedder)
        store = InMemoryBlobStore()

        # Instance A ingests: embeds both chunks once and flushes to the store.
        instance_a = CorpusManager(tmp_path / "a", blob_store=store)
        bundle = instance_a.create(name="Nova", domain="general")
        bundle.retrieval.build_indexes(_chunks())
        instance_a.flush(bundle.meta.corpus_id)
        assert embedder.passages_embedded == 2
        stored = store.load(bundle.meta.corpus_id)
        assert stored is not None and stored.vectors is not None

        # Instance B is fresh: hydrating must not call the embedding API again.
        instance_b = CorpusManager(tmp_path / "b", blob_store=store)
        restored = instance_b.get(bundle.meta.corpus_id)
        assert embedder.passages_embedded == 2
        assert restored.retrieval.faiss_store is not None
        assert len(restored.retrieval.faiss_store) == 2

    def test_an_index_from_another_model_is_rebuilt_and_saved_back(self, tmp_path: Path, monkeypatch):
        old = _CountingEmbedder()
        monkeypatch.setattr(retrieval_service, "get_embedder", lambda: old)
        store = InMemoryBlobStore()
        instance_a = CorpusManager(tmp_path / "a", blob_store=store)
        bundle = instance_a.create(name="Nova", domain="general")
        bundle.retrieval.build_indexes(_chunks())
        instance_a.flush(bundle.meta.corpus_id)

        new = _CountingEmbedder()
        new.model_name = "fake:other-model"
        monkeypatch.setattr(retrieval_service, "get_embedder", lambda: new)
        CorpusManager(tmp_path / "b", blob_store=store).get(bundle.meta.corpus_id)
        assert new.passages_embedded == 2  # stale vectors were not trusted
        reloaded = store.load(bundle.meta.corpus_id)
        assert reloaded is not None and reloaded.vectors is not None
        assert reloaded.vectors["meta"]["model"] == "fake:other-model"

        # A third instance on the new model now reuses what B stored.
        CorpusManager(tmp_path / "c", blob_store=store).get(bundle.meta.corpus_id)
        assert new.passages_embedded == 2

    def test_corrupt_stored_vectors_fall_back_to_embedding(self, tmp_path: Path, monkeypatch):
        embedder = _CountingEmbedder()
        monkeypatch.setattr(retrieval_service, "get_embedder", lambda: embedder)
        store = InMemoryBlobStore()
        instance_a = CorpusManager(tmp_path / "a", blob_store=store)
        bundle = instance_a.create(name="Nova", domain="general")
        bundle.retrieval.build_indexes(_chunks())
        instance_a.flush(bundle.meta.corpus_id)
        store.save_vectors(bundle.meta.corpus_id, {"meta": {}, "index_b64": "not base64!"})

        restored = CorpusManager(tmp_path / "b", blob_store=store).get(bundle.meta.corpus_id)
        assert restored.retrieval.faiss_store is not None
        assert embedder.passages_embedded == 4


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _ScriptedProvider:
    name = "fake:scripted"
    dimension = _DIM

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def embed_batch(self, texts, *, is_query=False):
        self.calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return [[1.0] * _DIM for _ in texts]


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid", code, "error", Message(), None)


class TestTransientErrorsDoNotDisableForGood:
    def test_a_rate_limit_pauses_then_recovers(self):
        clock = _Clock()
        provider = _ScriptedProvider([_http_error(429)])
        embedder = RemoteEmbedder(provider, clock=clock)

        assert embedder.embed_batch(["a"]) == []
        assert not embedder.available()
        assert embedder.embed_batch(["a"]) == []  # paused: no extra API call
        assert provider.calls == 1

        clock.now += RemoteEmbedder._COOLDOWN_S + 1
        assert embedder.available()
        assert len(embedder.embed_batch(["a"])[0]) == _DIM

    @pytest.mark.parametrize("code", [400, 401, 403])
    def test_a_rejected_key_stays_disabled(self, code):
        clock = _Clock()
        embedder = RemoteEmbedder(_ScriptedProvider([_http_error(code)]), clock=clock)
        embedder.embed_batch(["a"])
        clock.now += 10 * RemoteEmbedder._COOLDOWN_S
        assert not embedder.available()


class TestQueryCache:
    def test_a_repeated_question_costs_one_api_call(self):
        provider = _ScriptedProvider([])
        embedder = RemoteEmbedder(provider)
        first = embedder.embed_query("Who founded Nova Dynamics?")
        second = embedder.embed_query("Who founded Nova Dynamics?")
        assert first == second
        assert provider.calls == 1

    def test_failures_are_not_cached(self):
        clock = _Clock()
        provider = _ScriptedProvider([_http_error(503)])
        embedder = RemoteEmbedder(provider, clock=clock)
        assert embedder.embed_query("q") == []
        clock.now += RemoteEmbedder._COOLDOWN_S + 1
        assert embedder.embed_query("q")
        assert provider.calls == 2
