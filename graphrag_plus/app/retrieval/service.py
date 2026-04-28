"""Hybrid retrieval service."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from graphrag_plus.app.graph.store import GraphStore
from graphrag_plus.app.ingestion.models import Chunk
from graphrag_plus.app.utils.io_utils import dump_json, load_json
from graphrag_plus.app.utils.logging_utils import get_logger
from graphrag_plus.app.utils.math_utils import safe_entropy

# Use sklearn's built-in English stopword list so BM25 + tokenizer agree with
# the TF-IDF vocabulary and we don't accidentally let "about", "me", "tell"
# etc. drive matches for off-topic queries.
_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
# Minimum normalized lexical+semantic score required for a candidate to be
# considered relevant. Below this threshold the retrieval service abstains so
# that genuinely off-topic questions return NO_EVIDENCE instead of a confident
# wrong answer.
_MIN_RELEVANCE = 0.05

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords, drop ultra-short tokens.

    The same tokenizer is used at index time and query time so BM25 sees a
    consistent vocabulary.
    """
    tokens = [tok.lower() for tok in _TOKEN_RE.findall(text or "")]
    return [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 1]


@dataclass
class RetrievalCandidate:
    """Candidate from retrieval stack."""

    id: str
    source_id: str
    snippet: str
    semantic_score: float
    graph_score: float
    confidence_score: float
    trust_score: float
    uncertainty_penalty: float


class RetrievalService:
    """Vector + BM25 + graph retrieval.

    Indexes are persisted to ``chunks_path`` on every successful ``build_indexes``
    call and restored automatically when the service is constructed. This means
    a backend restart no longer wipes retrieval state and silently turns every
    query into ``NO_EVIDENCE``.
    """

    def __init__(self, graph_store: GraphStore, chunks_path: Path | None = None):
        self.graph_store = graph_store
        self.chunks_path = chunks_path
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(_STOPWORDS),
            token_pattern=r"(?u)\b[A-Za-z0-9_]{2,}\b",
        )
        self.chunk_matrix = None
        self.bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

        # Restore from disk if a previous session persisted chunks.
        if chunks_path is not None:
            persisted = load_json(chunks_path, default=None)
            if persisted:
                try:
                    restored = [Chunk(**row) for row in persisted]
                    if restored:
                        self._build(restored, persist=False)
                        logger.info(
                            "retrieval.indexes_restored count=%d path=%s",
                            len(restored),
                            chunks_path,
                        )
                except Exception as exc:  # malformed file -> skip
                    logger.warning("retrieval.restore_failed path=%s error=%s", chunks_path, exc)

    # ------------------------------------------------------------------ helpers
    def _build(self, chunks: list[Chunk], *, persist: bool) -> None:
        """Build (or rebuild) BM25 + TF-IDF indexes from a list of chunks."""
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        if not texts:
            self.chunk_matrix = None
            self.bm25 = None
            self._tokenized = []
            return
        # TF-IDF with shared tokenization rules.
        self.chunk_matrix = self.vectorizer.fit_transform(texts)
        self._tokenized = [_tokenize(text) for text in texts]
        # rank-bm25 chokes on empty token lists; backfill so every chunk has at
        # least its own id as a token.
        backfilled = [
            tokens or [chunk.chunk_id] for tokens, chunk in zip(self._tokenized, chunks, strict=True)
        ]
        self.bm25 = BM25Okapi(backfilled)
        if persist and self.chunks_path is not None:
            try:
                dump_json(self.chunks_path, [asdict(chunk) for chunk in chunks])
            except Exception as exc:  # non-fatal: in-memory indexes still work
                logger.warning("retrieval.persist_failed path=%s error=%s", self.chunks_path, exc)

    # ----------------------------------------------------------------- public API
    def build_indexes(self, chunks: list[Chunk]) -> None:
        """Build in-memory indexes and persist for restart resilience.

        Chunks accumulate across ingestion calls so that re-ingesting one file
        doesn't wipe earlier ones. Identical ``chunk_id`` values are de-duped
        with the most recent text winning.
        """
        merged: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in self.chunks}
        for chunk in chunks:
            merged[chunk.chunk_id] = chunk
        self._build(list(merged.values()), persist=True)

    def query(self, question: str, top_k: int, trust_lookup: dict[str, float]) -> list[dict[str, float]]:
        """Retrieve candidates with base scores."""
        if not self.chunks or self.chunk_matrix is None or self.bm25 is None:
            logger.info(
                "retrieval.empty_index question=%r chunks=%d",
                question[:60],
                len(self.chunks),
            )
            return []

        question_vec = self.vectorizer.transform([question])
        cosine = (self.chunk_matrix @ question_vec.T).toarray().ravel()
        question_tokens = _tokenize(question) or [question.lower()]
        bm25_scores = np.array(self.bm25.get_scores(question_tokens))
        graph_hits = self._graph_hit_scores(question)

        # rank-bm25 returns negative scores for tiny corpora (its IDF term goes
        # negative when most documents contain the query term). Clamp at 0 so
        # negative BM25 doesn't cancel a strong cosine match.
        bm25_scores = np.clip(bm25_scores, 0.0, None)
        bm25_max = float(bm25_scores.max()) if bm25_scores.size else 0.0
        bm25_norm = bm25_scores / bm25_max if bm25_max > 0 else bm25_scores

        rows: list[dict[str, float]] = []
        for idx, chunk in enumerate(self.chunks):
            source_id = chunk.doc_id
            semantic = float(cosine[idx])
            keyword = float(bm25_norm[idx])
            graph_score = graph_hits.get(chunk.chunk_id, 0.0)
            # Combined lexical+semantic score in [0, 1]-ish range.
            blended = 0.6 * semantic + 0.4 * keyword
            base_confidence = 0.5 + min(0.5, max(0.0, blended))
            trust_score = trust_lookup.get(source_id, 0.5)
            uncertainty = safe_entropy(base_confidence)
            rows.append(
                {
                    "id": chunk.chunk_id,
                    "source_id": source_id,
                    "snippet": chunk.text[:300],
                    "semantic_score": blended,
                    "graph_score": graph_score,
                    "confidence_score": base_confidence,
                    "trust_score": trust_score,
                    "uncertainty_penalty": uncertainty,
                    # Preserve a copy of the raw lexical+semantic score so that
                    # post-normalization downstream doesn't erase scale info.
                    "raw_relevance": blended,
                }
            )

        # Drop chunks below the minimum-relevance bar so that off-topic
        # questions return NO_EVIDENCE rather than a confident wrong answer.
        # A graph-hit alone is enough to keep a candidate (it indicates an
        # entity-level match even if surface tokens differ).
        scored = [row for row in rows if row["semantic_score"] >= _MIN_RELEVANCE or row["graph_score"] > 0]
        if not scored:
            return []
        scored.sort(key=lambda item: item["semantic_score"], reverse=True)
        return scored[: max(top_k * 3, 10)]

    def _graph_hit_scores(self, question: str) -> dict[str, float]:
        keywords = set(_tokenize(question))
        if not keywords:
            return {}
        scores: dict[str, float] = {}
        for node_id, attrs in self.graph_store.graph.nodes(data=True):
            label = str(attrs.get("label", "")).lower()
            if not label:
                continue
            label_tokens = set(_tokenize(label))
            overlap = len(keywords & label_tokens)
            if overlap <= 0:
                continue
            for pred in self.graph_store.graph.predecessors(node_id):
                if pred.startswith("doc_") or "_ch_" in pred:
                    scores[pred] = scores.get(pred, 0.0) + float(overlap)
        return scores
