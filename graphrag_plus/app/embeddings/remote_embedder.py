"""Hosted embedding APIs behind the local :class:`Embedder` interface.

The local path (``sentence-transformers`` + PyTorch) needs ~1.5 GB of wheels
and a model download, which no free, card-free host will give us. These
clients get the same 1024-dimensional vectors over HTTPS for nothing, so the
deployed backend keeps real semantic retrieval while the bundle stays small.

Two providers, both with a free tier:

* :class:`GeminiEmbedder` - Google AI Studio, ``gemini-embedding-001``.
  Free of charge on the free tier. Quota renews daily, which suits a demo
  that sits idle for weeks and then gets opened.
* :class:`JinaEmbedder` - Jina AI, ``jina-embeddings-v3``. A one-off free
  token allowance; used as the fallback when no Gemini key is present.

Both are selected by :func:`build_remote_embedder` purely from which API key
is in the environment, mirroring how ``build_default_llm_client`` picks an LLM.
Any failure returns ``[]`` rather than raising, so retrieval degrades to
BM25 + TF-IDF instead of the request failing - the same contract the local
embedder already had.

Dimension note: both providers are asked for **1024** dimensions so vectors
are interchangeable with the ``BAAI/bge-large-en-v1.5`` vectors this project
used locally, and so a persisted FAISS index stays loadable across providers.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any

from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Target dimensionality for every remote provider. Matches bge-large-en-v1.5,
# the model the local path used, so indexes and measured relevance floors stay
# comparable across backends.
REMOTE_EMBEDDING_DIM = 1024

# Gemini supports 128-3072 via output_dimensionality (Matryoshka truncation).
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL_DEFAULT = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Jina's v3 supports Matryoshka truncation through the `dimensions` field.
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL_DEFAULT = os.environ.get("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")

# Request ceiling. Providers cap batch size and payload bytes; chunking keeps
# us well inside both and bounds the blast radius of one failed call.
_MAX_BATCH = 64

# Per-request timeout. Ingestion embeds in batches and Vercel allows 300 s
# total, so a generous per-call timeout is safe and avoids spurious retries.
_TIMEOUT_S = 30.0


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    """POST JSON and decode the JSON response. Raises on any failure."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"content-type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")[:300]
        except Exception:  # pragma: no cover - defensive
            detail = ""
        logger.warning("remote_embedder.http_error status=%s detail=%s", exc.code, detail)
        raise
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("remote_embedder.request_failed error=%s", str(exc)[:200])
        raise
    return json.loads(body)


def _batched(texts: list[str], size: int) -> list[list[str]]:
    """Split ``texts`` into consecutive chunks of at most ``size``."""
    return [texts[i : i + size] for i in range(0, len(texts), size)]


class GeminiEmbedder:
    """Google AI Studio embeddings (free tier).

    Uses ``:batchEmbedContents`` so one HTTP call covers a batch. Each text
    becomes its own ``Content`` - passing several parts in a single request
    returns one aggregated vector instead of one per text, which would
    silently corrupt the index.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = GEMINI_MODEL_DEFAULT,
        dimension: int = REMOTE_EMBEDDING_DIM,
        timeout_s: float = _TIMEOUT_S,
    ) -> None:
        resolved = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved:
            raise ValueError("GeminiEmbedder requires an api_key or GEMINI_API_KEY/GOOGLE_API_KEY.")
        self._api_key: str = resolved
        self.model = model
        self._dimension = dimension
        self.timeout_s = timeout_s

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_batch(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Embed ``texts``; returns one vector per input, or [] on failure."""
        if not texts:
            return []
        # task_type steers the embedding toward retrieval use, asymmetrically:
        # queries and documents are embedded for different roles.
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []
        for batch in _batched(texts, _MAX_BATCH):
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": text}]},
                        "taskType": task_type,
                        "outputDimensionality": self._dimension,
                    }
                    for text in batch
                ]
            }
            url = f"{GEMINI_API_ROOT}/{self.model}:batchEmbedContents?key={self._api_key}"
            parsed = _post_json(url, payload, {}, self.timeout_s)
            embeddings = parsed.get("embeddings") or []
            if len(embeddings) != len(batch):
                logger.warning(
                    "remote_embedder.gemini_count_mismatch sent=%d got=%d",
                    len(batch),
                    len(embeddings),
                )
                raise ValueError("Gemini returned a different number of embeddings than inputs")
            vectors.extend([float(x) for x in item.get("values", [])] for item in embeddings)
        return vectors


class JinaEmbedder:
    """Jina AI embeddings (free token allowance).

    OpenAI-compatible request shape, so the whole batch goes in one call.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = JINA_MODEL_DEFAULT,
        dimension: int = REMOTE_EMBEDDING_DIM,
        timeout_s: float = _TIMEOUT_S,
    ) -> None:
        resolved = api_key or os.environ.get("JINA_API_KEY")
        if not resolved:
            raise ValueError("JinaEmbedder requires an api_key argument or JINA_API_KEY env var.")
        self._api_key: str = resolved
        self.model = model
        self._dimension = dimension
        self.timeout_s = timeout_s

    @property
    def name(self) -> str:
        return f"jina:{self.model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_batch(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Embed ``texts``; returns one vector per input, or [] on failure."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in _batched(texts, _MAX_BATCH):
            payload = {
                "model": self.model,
                "task": "retrieval.query" if is_query else "retrieval.passage",
                "dimensions": self._dimension,
                "input": list(batch),
            }
            parsed = _post_json(
                JINA_API_URL,
                payload,
                {"authorization": f"Bearer {self._api_key}"},
                self.timeout_s,
            )
            data = parsed.get("data") or []
            if len(data) != len(batch):
                logger.warning("remote_embedder.jina_count_mismatch sent=%d got=%d", len(batch), len(data))
                raise ValueError("Jina returned a different number of embeddings than inputs")
            # The API may return rows out of order; `index` is authoritative.
            ordered = sorted(data, key=lambda row: int(row.get("index", 0)))
            vectors.extend([float(x) for x in row.get("embedding", [])] for row in ordered)
        return vectors


# A provider is anything with this shape; kept structural so tests can inject
# a fake without importing either concrete client.
RemoteProvider = GeminiEmbedder | JinaEmbedder


class RemoteEmbedder:
    """Drop-in replacement for :class:`~.embedder.Embedder` backed by an API.

    Mirrors the local embedder's contract exactly: ``available()``,
    ``dimension``, ``embed_text``, ``embed_query`` and ``embed_batch``, with
    every failure degrading to ``[]`` instead of raising. One failed call
    disables the provider for the process so a dead key or an exhausted quota
    costs one timeout rather than one per query.
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._disabled = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ status
    @property
    def model_name(self) -> str:
        return str(getattr(self._provider, "name", "remote"))

    def available(self) -> bool:
        """True while the provider has not failed in this process."""
        return not self._disabled

    @property
    def dimension(self) -> int:
        return 0 if self._disabled else int(self._provider.dimension)

    def _disable(self, exc: Exception) -> None:
        with self._lock:
            if not self._disabled:
                self._disabled = True
                logger.warning(
                    "remote_embedder.disabled provider=%s error=%s - "
                    "semantic retrieval falls back to BM25 + TF-IDF",
                    self.model_name,
                    str(exc)[:200],
                )

    # ----------------------------------------------------------------- embedding
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed passages. [] when unavailable or on any provider error."""
        return self._embed(texts, is_query=False)

    def embed_text(self, text: str) -> list[float]:
        """Embed one passage. [] when unavailable."""
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Both providers take an explicit retrieval task hint, so the query/
        passage asymmetry is handled by the API rather than by prepending an
        instruction string the way BGE models require.
        """
        vectors = self._embed([text], is_query=True)
        return vectors[0] if vectors else []

    def _embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        if not texts or self._disabled:
            return []
        try:
            vectors = self._provider.embed_batch(texts, is_query=is_query)
        except Exception as exc:
            self._disable(exc)
            return []
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            self._disable(ValueError("provider returned empty or mismatched vectors"))
            return []
        return vectors


def build_remote_embedder() -> RemoteEmbedder | None:
    """Return a configured remote embedder, or ``None`` when no key is set.

    Preference order mirrors ``build_default_llm_client``: Gemini first (its
    free quota renews daily), then Jina (a one-off allowance). ``None`` means
    no remote provider is configured, which is not an error - the caller then
    tries the local model and finally BM25 + TF-IDF.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        try:
            provider: Any = GeminiEmbedder()
            logger.info("remote_embedder.selected provider=gemini model=%s", provider.model)
            return RemoteEmbedder(provider)
        except ValueError as exc:  # pragma: no cover - guarded by the check above
            logger.warning("remote_embedder.gemini_init_failed error=%s", exc)
    if os.environ.get("JINA_API_KEY"):
        try:
            provider = JinaEmbedder()
            logger.info("remote_embedder.selected provider=jina model=%s", provider.model)
            return RemoteEmbedder(provider)
        except ValueError as exc:  # pragma: no cover - guarded by the check above
            logger.warning("remote_embedder.jina_init_failed error=%s", exc)
    return None


__all__ = [
    "REMOTE_EMBEDDING_DIM",
    "GeminiEmbedder",
    "JinaEmbedder",
    "RemoteEmbedder",
    "build_remote_embedder",
]
