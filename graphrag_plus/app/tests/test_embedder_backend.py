"""Tests for embedding-backend selection and vector-space identity.

Two behaviours are pinned here:

* :func:`get_embedder` picks local, then remote, then nothing - and honours
  ``GRAPHRAG_EMBEDDING_BACKEND`` so the deployed path can be exercised from a
  machine that also has the local model installed.
* A persisted FAISS index is only reused when the SAME backend asks for it.
  Dimension alone is not enough: two providers can both emit 1024-dim vectors
  that mean entirely different things.
"""

from __future__ import annotations

import pytest

from graphrag_plus.app.embeddings import embedder as embedder_mod
from graphrag_plus.app.embeddings.embedder import (
    EmbedderLike,
    NullEmbedder,
    get_embedder,
    reset_embedder,
)
from graphrag_plus.app.embeddings.faiss_store import FAISSStore


@pytest.fixture(autouse=True)
def _isolate_backend(monkeypatch):
    """Reset the process-wide singleton and clear provider keys."""
    reset_embedder()
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "JINA_API_KEY", "GRAPHRAG_EMBEDDING_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    yield
    reset_embedder()


class _StubLocal:
    """Stands in for a loaded sentence-transformers model."""

    model_name = "stub-local"

    def __init__(self, *, is_available: bool = True):
        self._is_available = is_available

    def available(self) -> bool:
        return self._is_available

    @property
    def dimension(self) -> int:
        return 1024 if self._is_available else 0

    def embed_text(self, text: str) -> list[float]:
        return [0.5] * 1024

    def embed_query(self, text: str) -> list[float]:
        return [0.5] * 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * 1024 for _ in texts]


class TestProtocolConformance:
    def test_null_embedder_satisfies_the_protocol(self):
        assert isinstance(NullEmbedder(), EmbedderLike)

    def test_null_embedder_is_inert(self):
        null = NullEmbedder()
        assert not null.available()
        assert null.dimension == 0
        assert null.model_name == "none"
        assert null.embed_text("x") == []
        assert null.embed_query("x") == []
        assert null.embed_batch(["x", "y"]) == []


class TestBackendSelection:
    def test_prefers_an_available_local_model(self, monkeypatch):
        monkeypatch.setattr(embedder_mod, "Embedder", _StubLocal)
        assert get_embedder().model_name == "stub-local"

    def test_falls_back_to_remote_when_local_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(embedder_mod, "Embedder", lambda: _StubLocal(is_available=False))
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        assert get_embedder().model_name.startswith("gemini:")

    def test_returns_null_when_nothing_is_configured(self, monkeypatch):
        monkeypatch.setattr(embedder_mod, "Embedder", lambda: _StubLocal(is_available=False))
        backend = get_embedder()
        assert not backend.available()
        assert backend.model_name == "none"

    def test_remote_can_be_forced_over_an_available_local_model(self, monkeypatch):
        # The reason this override exists: the cloud path has to be testable
        # from a laptop that also has the local model installed.
        monkeypatch.setattr(embedder_mod, "Embedder", _StubLocal)
        monkeypatch.setenv("GRAPHRAG_EMBEDDING_BACKEND", "remote")
        monkeypatch.setenv("JINA_API_KEY", "k")
        assert get_embedder().model_name.startswith("jina:")

    def test_forced_remote_without_a_key_is_null_not_local(self, monkeypatch):
        monkeypatch.setattr(embedder_mod, "Embedder", _StubLocal)
        monkeypatch.setenv("GRAPHRAG_EMBEDDING_BACKEND", "remote")
        assert get_embedder().model_name == "none"

    def test_forced_local_without_a_model_does_not_reach_remote(self, monkeypatch):
        monkeypatch.setattr(embedder_mod, "Embedder", lambda: _StubLocal(is_available=False))
        monkeypatch.setenv("GRAPHRAG_EMBEDDING_BACKEND", "local")
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        assert get_embedder().model_name == "none"

    def test_backend_is_cached_across_calls(self, monkeypatch):
        calls = []

        def _factory():
            calls.append(1)
            return _StubLocal()

        monkeypatch.setattr(embedder_mod, "Embedder", _factory)
        get_embedder()
        get_embedder()
        assert len(calls) == 1


class TestFaissVectorSpaceIdentity:
    """A saved index must not be reused across embedding backends."""

    @staticmethod
    def _save(tmp_path, model_name: str):
        pytest.importorskip("faiss", reason="faiss-cpu is an optional dependency")
        store = FAISSStore(dimension=8, model_name=model_name)
        store.add_batch(["c1", "c2"], [[0.1] * 8, [0.2] * 8])
        path = tmp_path / "faiss.index"
        store.save(str(path))
        return path

    def test_model_name_round_trips(self, tmp_path):
        path = self._save(tmp_path, "gemini:gemini-embedding-001")
        restored = FAISSStore(dimension=8)
        assert restored.load(str(path))
        assert restored.model_name == "gemini:gemini-embedding-001"
        assert restored.dimension == 8

    def test_a_different_backend_is_detectable_at_the_same_dimension(self, tmp_path):
        # Both providers emit 1024-dim vectors, so the dimension check passes
        # and only the identity check catches the mismatch.
        path = self._save(tmp_path, "gemini:gemini-embedding-001")
        restored = FAISSStore(dimension=8)
        restored.load(str(path))
        assert restored.dimension == 8
        assert restored.model_name != "jina:jina-embeddings-v3"

    def test_a_legacy_index_without_the_field_fails_closed(self, tmp_path):
        import json

        path = self._save(tmp_path, "whatever")
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["model"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        restored = FAISSStore(dimension=8)
        assert restored.load(str(path))
        # Empty string never equals a real backend name, so the caller rebuilds
        # rather than trusting vectors of unknown provenance.
        assert restored.model_name == ""
