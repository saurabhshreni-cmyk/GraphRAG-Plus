"""Embedding service: a local model when present, a hosted API otherwise.

:class:`Embedder` wraps a ``sentence-transformers`` model (default
``BAAI/bge-large-en-v1.5``, 1024-dim) behind a lazy singleton: loaded on first
use and shared process-wide.

:func:`get_embedder` picks a backend at first call:

1. ``sentence-transformers`` when importable - the local-first path, used on a
   dev machine where the model is already cached.
2. A hosted API (Gemini, then Jina) when an API key is set. This is what the
   deployed backend uses: PyTorch alone exceeds what any free, card-free host
   will run, while the API returns the same 1024-dim vectors for nothing.
3. Nothing - ``available() == False``, and callers fall back to BM25 + TF-IDF.

``GRAPHRAG_EMBEDDING_BACKEND`` forces the choice (``local``, ``remote`` or
``auto``; default ``auto``), which is how the cloud path gets exercised from a
laptop that also has the local model installed.

No backend ever hard-fails the pipeline: every path degrades to ``[]`` and
retrieval loses the semantic signal rather than the request.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Protocol, runtime_checkable

from dotenv import load_dotenv

from graphrag_plus.app.embeddings.remote_embedder import build_remote_embedder
from graphrag_plus.app.utils.logging_utils import get_logger

load_dotenv()  # ensure EMBEDDING_MODEL from the project .env is visible

logger = get_logger(__name__)

_FALLBACK_MODEL = "BAAI/bge-large-en-v1.5"

# BGE-family models are trained with an instruction prefix on the QUERY side
# (passages are embedded bare). Applying it lifts retrieval quality by
# several MTEB points; other model families ignore the concept entirely.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Backend selection. "auto" prefers a local model and falls back to a hosted
# API; "local"/"remote" pin one backend (useful for testing the deployed path
# from a machine that has the local model installed).
_BACKEND_ENV = "GRAPHRAG_EMBEDDING_BACKEND"


@runtime_checkable
class EmbedderLike(Protocol):
    """The contract every embedding backend satisfies.

    Local and remote backends are interchangeable through this protocol, so
    the retrieval service never learns which one it is talking to.
    """

    @property
    def model_name(self) -> str:
        """Identity of the vector space.

        For example ``BAAI/bge-large-en-v1.5`` or
        ``gemini:gemini-embedding-001``. Persisted alongside a FAISS index so
        an index built by one backend is never reused against another
        backend's query vectors - two backends can share a dimension and still
        be entirely incomparable.

        Declared read-only so implementations are free to expose it as a
        plain attribute or a property.
        """
        ...

    def available(self) -> bool:
        """True when this backend can currently produce vectors."""
        ...

    @property
    def dimension(self) -> int:
        """Vector width, or 0 when unavailable."""
        ...

    def embed_text(self, text: str) -> list[float]:
        """Embed one passage. ``[]`` when unavailable."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query. ``[]`` when unavailable."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many passages. ``[]`` when unavailable."""
        ...


class NullEmbedder:
    """Backend of last resort: always unavailable, never raises.

    Lets callers treat "no embedding backend configured" the same as "the
    model failed to load", instead of handling ``None``.
    """

    model_name = "none"

    def available(self) -> bool:
        return False

    @property
    def dimension(self) -> int:
        return 0

    def embed_text(self, text: str) -> list[float]:
        return []

    def embed_query(self, text: str) -> list[float]:
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return []


def _default_model_name() -> str:
    """Resolve the model name from the environment at call time."""
    return os.environ.get("EMBEDDING_MODEL", _FALLBACK_MODEL)


def _dimension_of(model: Any) -> int:
    """Embedding dim across sentence-transformers versions (method renamed in v5)."""
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(getter())


class Embedder:
    """Lazy, thread-safe wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or _default_model_name()
        self._is_bge = "bge" in self.model_name.lower()
        self._model: Any = None
        self._load_failed = False
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is not None or self._load_failed:
            return self._model
        with self._lock:
            if self._model is not None or self._load_failed:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("embedder.loading model=%s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("embedder.loaded model=%s dim=%d", self.model_name, _dimension_of(self._model))
            except Exception as exc:
                self._load_failed = True
                logger.warning("embedder.load_failed error=%s — semantic search disabled", str(exc)[:200])
        return self._model

    def available(self) -> bool:
        return self._get_model() is not None

    @property
    def dimension(self) -> int:
        model = self._get_model()
        return _dimension_of(model) if model else 0

    def embed_text(self, text: str) -> list[float]:
        """Embed one string (passage-side). [] when the model is unavailable."""
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        BGE models expect their instruction prefix on queries only; for
        other model families this is identical to :meth:`embed_text`.
        """
        if self._is_bge:
            text = _BGE_QUERY_INSTRUCTION + text
        return self.embed_text(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings in one forward pass. [] when unavailable."""
        if not texts:
            return []
        model = self._get_model()
        if model is None:
            return []
        try:
            vectors = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # FAISS store normalizes
            )
            return [vector.tolist() for vector in vectors]
        except Exception as exc:
            logger.warning("embedder.encode_failed error=%s", str(exc)[:200])
            return []


# Shared instance — a local model weighs hundreds of MB in RAM and a remote
# client holds a disable flag that must be process-wide, so one instance per
# process is the right granularity either way.
_shared_embedder: EmbedderLike | None = None
_shared_lock = threading.Lock()


def _select_backend() -> EmbedderLike:
    """Choose an embedding backend. Never raises; worst case NullEmbedder."""
    backend = os.environ.get(_BACKEND_ENV, "auto").strip().lower()

    if backend in {"auto", "local"}:
        local = Embedder()
        if local.available():
            logger.info("embedder.backend=local model=%s dim=%d", local.model_name, local.dimension)
            return local
        if backend == "local":
            logger.warning("embedder.backend=local requested but sentence-transformers is unavailable")
            return NullEmbedder()

    if backend in {"auto", "remote"}:
        remote = build_remote_embedder()
        if remote is not None:
            logger.info("embedder.backend=remote model=%s dim=%d", remote.model_name, remote.dimension)
            return remote
        if backend == "remote":
            logger.warning("embedder.backend=remote requested but no provider API key is set")
            return NullEmbedder()

    logger.warning(
        "embedder.backend=none — no local model and no API key; "
        "semantic retrieval is disabled (BM25 + TF-IDF still active)"
    )
    return NullEmbedder()


def get_embedder() -> EmbedderLike:
    """Return the process-wide embedding backend, selecting it on first call."""
    global _shared_embedder
    if _shared_embedder is None:
        with _shared_lock:
            if _shared_embedder is None:
                _shared_embedder = _select_backend()
    return _shared_embedder


def reset_embedder() -> None:
    """Drop the cached backend so the next call re-selects one.

    Only for tests and for reacting to a configuration change at runtime.
    """
    global _shared_embedder
    with _shared_lock:
        _shared_embedder = None


__all__ = [
    "Embedder",
    "EmbedderLike",
    "NullEmbedder",
    "get_embedder",
    "reset_embedder",
]
