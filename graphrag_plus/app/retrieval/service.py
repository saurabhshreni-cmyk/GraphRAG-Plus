"""Hybrid retrieval service: BM25 + FAISS semantic + Neo4j graph.

Signal stack (each optional except BM25):

* **BM25** (rank-bm25) — lexical keyword recall. Always available.
* **Semantic** — sentence-transformers ``all-MiniLM-L6-v2`` embeddings in a
  FAISS ``IndexFlatIP`` (cosine). Falls back to TF-IDF cosine when the
  embedding model is unavailable so retrieval never goes dark.
* **Graph** — spaCy entities from the question traverse the Neo4j corpus
  graph (``get_related_chunks``); the legacy NetworkX label-overlap signal
  is kept as an additive local component.

Blend weights (hybrid mode): BM25=0.35, semantic=0.40, graph=0.25.
Legacy TF-IDF mode keeps the original 0.7 cosine / 0.3 BM25 blend.

Indexes are persisted to ``chunks_path`` (JSON) and ``faiss.index`` (binary,
same directory) on every successful ``build_indexes`` call and restored on
construction, so a backend restart doesn't wipe retrieval state.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from graphrag_plus.app.embeddings.embedder import get_embedder
from graphrag_plus.app.embeddings.faiss_store import FAISSStore
from graphrag_plus.app.graph.store import GraphStore
from graphrag_plus.app.ingestion.models import Chunk
from graphrag_plus.app.utils.io_utils import dump_json, load_json
from graphrag_plus.app.utils.logging_utils import get_logger, log_event
from graphrag_plus.app.utils.math_utils import safe_entropy

# Use sklearn's built-in English stopword list so BM25 + tokenizer agree with
# the TF-IDF vocabulary and we don't accidentally let "about", "me", "tell"
# etc. drive matches for off-topic queries.
_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# --- relevance gates --------------------------------------------------------
# Either signal alone is enough to keep a candidate:
#   - blended score >= ``_MIN_BLEND``, OR
#   - semantic alone >= ``_MIN_COSINE``, OR
#   - graph hit (entity-level connection to the question).
# In addition, chunks must share at least one non-stopword token with the
# question OR have semantic >= ``_STRONG_COSINE``.
_MIN_BLEND = 0.20
_MIN_COSINE = 0.20
_STRONG_COSINE = 0.30

# Hybrid blend weights (FAISS semantic available).
_W_BM25_HYBRID = 0.35
_W_SEMANTIC_HYBRID = 0.40
_W_GRAPH_HYBRID = 0.25

# Legacy blend weights (TF-IDF fallback — no reliable graph normalization).
_W_COSINE = 0.7
_W_BM25 = 0.3

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords, drop ultra-short tokens.

    The same tokenizer is used at index time and query time so BM25 sees a
    consistent vocabulary.
    """
    tokens = [tok.lower() for tok in _TOKEN_RE.findall(text or "")]
    return [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 1]


# Patterns that suggest a chunk contains an enumeration. Used for LIST-intent
# boosting only — never for gating.
_LIST_MARKER_RE = re.compile(
    r"(?:"
    r"\b(?:types?|kinds?|examples?|methods?|categories|forms?|varieties|classes|approaches|styles)\s+(?:of|include|are)\b|"
    r"\bsuch\s+as\b|"
    r"\bfollowing\s+(?:types|kinds|examples|methods)\b|"
    r"\bincludes?\s+the\s+following\b"
    r")",
    re.IGNORECASE,
)
_BULLET_LINE_RE = re.compile(r"(?m)^\s*(?:[-*•·]|\d+[.)]|[a-z][.)])\s+\S")


def _has_list_markers(text_lower: str) -> bool:
    """True if ``text_lower`` looks like it enumerates things."""
    if _LIST_MARKER_RE.search(text_lower):
        return True
    return bool(_BULLET_LINE_RE.search(text_lower))


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
    """BM25 + FAISS semantic + Neo4j graph retrieval with graceful fallback."""

    def __init__(self, graph_store: GraphStore, chunks_path: Path | None = None):
        self.graph_store = graph_store
        self.chunks_path = chunks_path
        # Corpus directories are named corpus_<id>; the FAISS index and the
        # Neo4j node partition share that id so all three stores stay aligned.
        self.corpus_id = chunks_path.parent.name if chunks_path is not None else "default"
        self.faiss_path = chunks_path.parent / "faiss.index" if chunks_path is not None else None
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(_STOPWORDS),
            token_pattern=r"(?u)\b[A-Za-z0-9_]{2,}\b",
        )
        self.chunk_matrix = None
        self.bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []
        self.faiss_store: FAISSStore | None = None

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
        """Build (or rebuild) BM25 + TF-IDF + FAISS indexes from chunks."""
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        if not texts:
            self.chunk_matrix = None
            self.bm25 = None
            self._tokenized = []
            self.faiss_store = None
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
        self._build_faiss(chunks, persist=persist)
        if persist and self.chunks_path is not None:
            try:
                dump_json(self.chunks_path, [asdict(chunk) for chunk in chunks])
            except Exception as exc:  # non-fatal: in-memory indexes still work
                logger.warning("retrieval.persist_failed path=%s error=%s", self.chunks_path, exc)

    def _build_faiss(self, chunks: list[Chunk], *, persist: bool) -> None:
        """Load or (re)build the FAISS semantic index for these chunks.

        Restore path (persist=False): reuse the saved index when its ids match
        the current chunk set, avoiding a re-embedding pass on every restart.
        Ingest path (persist=True): always rebuild from scratch and save.
        Any failure leaves ``faiss_store = None`` → TF-IDF cosine fallback.
        """
        self.faiss_store = None
        try:
            embedder = get_embedder()
            if not embedder.available():
                return
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            if not persist and self.faiss_path is not None:
                candidate = FAISSStore(dimension=embedder.dimension)
                if candidate.load(str(self.faiss_path)) and sorted(candidate.chunk_ids) == sorted(
                    chunk_ids
                ):
                    self.faiss_store = candidate
                    return
            store = FAISSStore(dimension=embedder.dimension)
            embeddings = embedder.embed_batch([chunk.text for chunk in chunks])
            if len(embeddings) != len(chunks):
                logger.warning("retrieval.faiss_embed_incomplete got=%d want=%d", len(embeddings), len(chunks))
                return
            store.add_batch(chunk_ids, embeddings)
            self.faiss_store = store
            if self.faiss_path is not None:
                store.save(str(self.faiss_path))
            logger.info("retrieval.faiss_built corpus=%s vectors=%d", self.corpus_id, len(chunk_ids))
        except Exception as exc:
            self.faiss_store = None
            logger.warning("retrieval.faiss_build_failed error=%s — TF-IDF fallback", str(exc)[:200])

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

    # ------------------------------------------------------------ signal helpers
    def _semantic_scores(self, question: str) -> tuple[dict[str, float], bool]:
        """FAISS cosine per chunk_id. Returns ({}, False) when unavailable."""
        if self.faiss_store is None or len(self.faiss_store) == 0:
            return {}, False
        try:
            embedder = get_embedder()
            query_vec = embedder.embed_text(question)
            if not query_vec:
                return {}, False
            hits = self.faiss_store.search(query_vec, top_k=min(len(self.chunks), 50))
            return {chunk_id: max(0.0, score) for chunk_id, score in hits}, True
        except Exception as exc:
            logger.warning("retrieval.semantic_failed error=%s", str(exc)[:200])
            return {}, False

    def _neo4j_graph_scores(self, question: str) -> dict[str, float]:
        """Traverse the Neo4j corpus graph from the question's entities.

        spaCy-only extraction (no LLM — must stay fast at query time). Any
        failure returns {} so the query proceeds on the other signals.
        """
        try:
            from graphrag_plus.app.extraction.extractor import extract
            from graphrag_plus.app.graph.neo4j_store import get_neo4j_store

            result = extract(question, use_llm=False)
            entity_names = [entity.name for entity in result.entities]
            # Content tokens as a fallback when NER finds nothing in short
            # questions ("what is backpropagation?").
            if not entity_names:
                entity_names = _tokenize(question)[:5]
            if not entity_names:
                return {}
            rows = get_neo4j_store().get_related_chunks(entity_names, self.corpus_id, max_hops=2)
            scores: dict[str, float] = {}
            for row in rows:
                chunk_id = str(row.get("chunk_id", ""))
                weight = float(row.get("weight", 0.0))
                if chunk_id:
                    scores[chunk_id] = max(scores.get(chunk_id, 0.0), weight)
            return scores
        except Exception as exc:
            logger.warning("retrieval.neo4j_signal_failed error=%s", str(exc)[:200])
            return {}

    def query(
        self,
        question: str,
        top_k: int,
        trust_lookup: dict[str, float],
        *,
        intent: str | None = None,
        comparison_terms: tuple[str, str] | None = None,
        loose: bool = False,
    ) -> list[dict[str, float]]:
        """Retrieve candidates with base scores.

        Pipeline:
            1. Score every chunk on BM25 + semantic (FAISS, TF-IDF fallback)
               + graph (Neo4j traversal + local label overlap).
            2. Blend: hybrid 0.35/0.40/0.25 (FAISS mode) or 0.7/0.3 (legacy).
            3. Apply intent-aware boosts (list markers, comparison gates).
            4. Reject chunks that share no query term and have weak semantic.
            5. Reject chunks below the relevance floor.
            6. Sort by blended score and return top-k * 3.

        ``loose`` relaxes the relevance floor by 50% and skips the
        term-overlap gate — used as a last-resort fallback retrieval pass
        from the pipeline when strict mode returned nothing.
        """
        if not self.chunks or self.chunk_matrix is None or self.bm25 is None:
            log_event(
                logger,
                "retrieval.empty_index",
                {"question": question[:80], "chunks": len(self.chunks)},
            )
            return []

        question_vec = self.vectorizer.transform([question])
        tfidf_cosine = (self.chunk_matrix @ question_vec.T).toarray().ravel()
        question_tokens = _tokenize(question) or [question.lower()]
        question_token_set = set(question_tokens)
        bm25_scores = np.array(self.bm25.get_scores(question_tokens))

        # Semantic signal: FAISS cosine when available, TF-IDF cosine otherwise.
        semantic_by_id, semantic_is_faiss = self._semantic_scores(question)

        # Graph signal: Neo4j traversal + legacy NetworkX label overlap.
        neo4j_scores = self._neo4j_graph_scores(question)
        local_graph_hits = self._graph_hit_scores(question)
        local_max = max(local_graph_hits.values()) if local_graph_hits else 0.0

        # Clamp negative BM25 and normalize against corpus max.
        bm25_scores = np.clip(bm25_scores, 0.0, None)
        bm25_max = float(bm25_scores.max()) if bm25_scores.size else 0.0
        bm25_norm = bm25_scores / bm25_max if bm25_max > 0 else bm25_scores

        # Pre-compute lowered comparison terms for the comparison gate.
        cmp_a, cmp_b = (None, None)
        if intent == "comparison" and comparison_terms is not None:
            cmp_a, cmp_b = (
                comparison_terms[0].lower().strip(),
                comparison_terms[1].lower().strip(),
            )

        # Loose-mode floors for the fallback pass.
        min_blend = _MIN_BLEND * (0.5 if loose else 1.0)
        min_cosine = _MIN_COSINE * (0.5 if loose else 1.0)

        rows: list[dict[str, float]] = []
        rejected: list[dict[str, object]] = []
        for idx, chunk in enumerate(self.chunks):
            source_id = chunk.doc_id
            keyword = float(bm25_norm[idx])
            semantic = (
                semantic_by_id.get(chunk.chunk_id, 0.0) if semantic_is_faiss else float(tfidf_cosine[idx])
            )

            # Graph: Neo4j weight (already 0..1) blended with normalized
            # local overlap; take the max so either source can carry it.
            local_component = (
                local_graph_hits.get(chunk.chunk_id, 0.0) / local_max if local_max > 0 else 0.0
            )
            graph_signal = max(neo4j_scores.get(chunk.chunk_id, 0.0), local_component)

            if semantic_is_faiss:
                blended = (
                    _W_BM25_HYBRID * keyword
                    + _W_SEMANTIC_HYBRID * semantic
                    + _W_GRAPH_HYBRID * graph_signal
                )
            else:
                blended = _W_COSINE * semantic + _W_BM25 * keyword

            chunk_text_lower = (chunk.text or "").lower()

            # ---- Intent-aware boosts (additive, not gating) ---------------
            list_bonus = 0.0
            if intent == "list" and _has_list_markers(chunk_text_lower):
                list_bonus = 0.10
                blended += list_bonus

            # ---- Comparison gate -----------------------------------------
            comparison_pass = True
            if cmp_a and cmp_b:
                has_a = cmp_a in chunk_text_lower
                has_b = cmp_b in chunk_text_lower
                # Pass if BOTH terms present, OR semantic strong enough.
                comparison_pass = (has_a and has_b) or semantic >= _STRONG_COSINE

            chunk_tokens = self._tokenized[idx] if idx < len(self._tokenized) else _tokenize(chunk.text)
            shares_term = bool(question_token_set.intersection(chunk_tokens))
            semantic_strong = semantic >= _STRONG_COSINE

            blend_strong = blended >= min_blend
            cos_above_floor = semantic >= min_cosine
            graph_strong = graph_signal > 0

            base_row = {
                "id": chunk.chunk_id,
                "source_id": source_id,
                # Preserve the full chunk text so the generator can extract
                # complete sentences instead of snippet-boundary fragments.
                # The 300-char ``snippet`` field stays for legacy callers /
                # API responses that don't want the full payload.
                "snippet": chunk.text[:300],
                "full_text": chunk.text,
                "semantic_score": blended,
                "graph_score": graph_signal,
                "confidence_score": 0.5 + min(0.5, max(0.0, blended)),
                "trust_score": trust_lookup.get(source_id, 0.5),
                "uncertainty_penalty": safe_entropy(0.5 + min(0.5, max(0.0, blended))),
                "raw_relevance": blended,
                "raw_cosine": semantic,
                "raw_bm25": keyword,
                "raw_graph": graph_signal,
                "semantic_backend": "faiss" if semantic_is_faiss else "tfidf",
                "list_bonus": list_bonus,
            }

            if not comparison_pass:
                rejected.append({**base_row, "reason": "missing_comparison_term"})
                continue
            if not loose and not (shares_term or semantic_strong):
                rejected.append({**base_row, "reason": "no_term_overlap"})
                continue
            if not (blend_strong or cos_above_floor or graph_strong):
                rejected.append({**base_row, "reason": "below_threshold"})
                continue
            rows.append(base_row)

        rows.sort(key=lambda item: item["semantic_score"], reverse=True)
        kept = rows[: max(top_k * 3, 10)]

        # Structured per-query log: kept and rejected chunks with their raw
        # signals, plus which retrieval backends contributed.
        log_event(
            logger,
            "retrieval.query",
            {
                "question": question[:120],
                "tokens": question_tokens,
                "semantic_backend": "faiss" if semantic_is_faiss else "tfidf",
                "neo4j_hits": len(neo4j_scores),
                "kept": [
                    {
                        "id": r["id"],
                        "sem": round(r["raw_cosine"], 3),
                        "bm25": round(r["raw_bm25"], 3),
                        "graph": round(r["raw_graph"], 3),
                        "blend": round(r["raw_relevance"], 3),
                    }
                    for r in kept
                ],
                "rejected": [
                    {
                        "id": r["id"],
                        "sem": round(float(r["raw_cosine"]), 3),
                        "bm25": round(float(r["raw_bm25"]), 3),
                        "reason": r["reason"],
                    }
                    for r in rejected[:5]
                ],
            },
        )
        return kept

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
