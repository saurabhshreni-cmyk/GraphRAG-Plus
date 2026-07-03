"""FAISS vector store (cosine similarity via normalized inner product).

Uses ``IndexFlatIP`` — exact search, no training step, ideal for corpora up
to a few hundred thousand chunks. Vectors are L2-normalized before insertion
and before querying, so inner product == cosine similarity.

Persistence writes two files next to each other:

* ``<path>``           — the FAISS index binary
* ``<path>.meta.json`` — the ordered chunk-id list (row i ↔ chunk_ids[i])

Chunk-id de-duplication: re-adding an existing chunk_id replaces its vector
logically (the old row is masked out of search results) so re-ingesting a
document doesn't return duplicate hits.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class FAISSStore:
    """Exact cosine-similarity index over chunk embeddings."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._lock = threading.Lock()
        self._index = None  # lazy: faiss import deferred until first use
        self.chunk_ids: list[str] = []
        # chunk_id -> latest row; superseded rows are skipped at search time.
        self._latest_row: dict[str, int] = {}

    # ------------------------------------------------------------------ internals
    def _get_index(self):
        if self._index is None:
            import faiss

            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    # ------------------------------------------------------------------ mutation
    def add_chunk(self, chunk_id: str, text: str, embedding: list[float]) -> None:
        """Add (or logically replace) one chunk vector.

        ``text`` is accepted for interface completeness but not stored —
        chunk text lives in the retrieval service / Neo4j; the FAISS store
        maps vectors to chunk_ids only.
        """
        _ = text
        if not embedding:
            return
        self.add_batch([chunk_id], [embedding])

    def add_batch(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        """Vectorized insert — one FAISS call for the whole batch."""
        if not chunk_ids or not embeddings or len(chunk_ids) != len(embeddings):
            return
        matrix = _normalize(np.asarray(embeddings, dtype=np.float32))
        with self._lock:
            index = self._get_index()
            start_row = index.ntotal
            index.add(matrix)
            for offset, chunk_id in enumerate(chunk_ids):
                self.chunk_ids.append(chunk_id)
                self._latest_row[chunk_id] = start_row + offset

    # ------------------------------------------------------------------- search
    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``top_k`` (chunk_id, cosine_score) pairs, best first."""
        if not query_embedding:
            return []
        with self._lock:
            index = self._get_index()
            if index.ntotal == 0:
                return []
            query = _normalize(np.asarray([query_embedding], dtype=np.float32))
            # Over-fetch to survive masked (superseded) rows, capped at ntotal.
            fetch = min(index.ntotal, max(top_k * 2, top_k + 8))
            scores, rows = index.search(query, fetch)
            results: list[tuple[str, float]] = []
            for score, row in zip(scores[0], rows[0], strict=True):
                if row < 0:
                    continue
                chunk_id = self.chunk_ids[row]
                if self._latest_row.get(chunk_id) != row:
                    continue  # superseded by a later re-ingest
                results.append((chunk_id, float(score)))
                if len(results) >= top_k:
                    break
            return results

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Write the index + chunk-id metadata to ``path`` (+ .meta.json)."""
        import faiss

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self._get_index(), str(target))
            meta = {"dimension": self.dimension, "chunk_ids": self.chunk_ids}
            target.with_suffix(target.suffix + ".meta.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
        logger.info("faiss.saved path=%s vectors=%d", path, len(self.chunk_ids))

    def load(self, path: str) -> bool:
        """Restore a saved index. Returns False (and stays empty) on failure."""
        import faiss

        target = Path(path)
        meta_path = target.with_suffix(target.suffix + ".meta.json")
        if not target.exists() or not meta_path.exists():
            return False
        try:
            index = faiss.read_index(str(target))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            chunk_ids = list(meta.get("chunk_ids", []))
            if index.ntotal != len(chunk_ids):
                logger.warning(
                    "faiss.load_mismatch vectors=%d ids=%d — ignoring saved index",
                    index.ntotal,
                    len(chunk_ids),
                )
                return False
            with self._lock:
                self._index = index
                self.dimension = index.d
                self.chunk_ids = chunk_ids
                self._latest_row = {cid: row for row, cid in enumerate(chunk_ids)}
            logger.info("faiss.loaded path=%s vectors=%d", path, len(chunk_ids))
            return True
        except Exception as exc:
            logger.warning("faiss.load_failed path=%s error=%s", path, str(exc)[:200])
            return False

    def clear(self) -> None:
        with self._lock:
            self._index = None
            self.chunk_ids = []
            self._latest_row = {}

    def __len__(self) -> int:
        return len(self._latest_row)
