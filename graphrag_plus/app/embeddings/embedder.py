"""Sentence-transformers embedding service.

Wraps ``all-MiniLM-L6-v2`` (384-dim) behind a lazy singleton: the model is
only loaded on first use (~90 MB download on the very first run, then cached
by huggingface hub), and one instance is shared process-wide.

If sentence-transformers or torch is unavailable the embedder reports
``available() == False`` and callers fall back to TF-IDF — the pipeline
never hard-fails on a missing model.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def _dimension_of(model: Any) -> int:
    """Embedding dim across sentence-transformers versions (method renamed in v5)."""
    getter = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension"
    )
    return int(getter())


class Embedder:
    """Lazy, thread-safe wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
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
        """Embed one string. Returns [] when the model is unavailable."""
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

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


# Shared instance — the underlying model weighs ~90 MB in RAM, so one per
# process is the right granularity.
_shared_embedder: Embedder | None = None
_shared_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _shared_embedder
    if _shared_embedder is None:
        with _shared_lock:
            if _shared_embedder is None:
                _shared_embedder = Embedder()
    return _shared_embedder
