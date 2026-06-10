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
from graphrag_plus.app.utils.logging_utils import get_logger, log_event
from graphrag_plus.app.utils.math_utils import safe_entropy

# Use sklearn's built-in English stopword list so BM25 + tokenizer agree with
# the TF-IDF vocabulary and we don't accidentally let "about", "me", "tell"
# etc. drive matches for off-topic queries.
_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# --- relevance gates --------------------------------------------------------
# Loosely-related chunks were leaking through and diluting answers (e.g.
# "what is graph data structure?" pulled in adjacency-matrix text alongside
# the correct chunk). The defaults below were tuned against the demo corpus
# so that:
#   * top-1 wins decisively, runner-ups only join if they're genuinely close,
#   * off-topic questions return NO_EVIDENCE instead of a confident wrong
#     answer.
#
# Either signal alone is enough to keep a candidate:
#   - blended (cosine + BM25) >= ``_MIN_BLEND``, OR
#   - cosine alone >= ``_MIN_COSINE`` (semantic similarity is the dominant
#     signal so a strong cosine wins even if BM25 is noisy / zero), OR
#   - graph hit (entity-level overlap with the question).
#
# In addition, chunks must share at least one non-stopword token with the
# question OR have cosine >= ``_STRONG_COSINE``. That blocks chunks that
# share zero query terms but happen to register a tiny cosine residual.
_MIN_BLEND = 0.20
_MIN_COSINE = 0.20
_STRONG_COSINE = 0.30
# Blend ratio: semantic dominates so on-topic cosine matches outrank lexical
# noise. BM25 still contributes for keyword recall.
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
    """True if ``text_lower`` looks like it enumerates things.

    Triggers on either lexical cues ("types of", "such as") or visible
    bullet/numbered lines. Cheap regex check — runs once per chunk.
    """
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
            1. Score every chunk on cosine + BM25 + graph hits.
            2. Apply intent-aware boosts:
               * ``list``       — boost chunks containing list markers
                 (``"types of"``, bullets, numbered lines).
               * ``comparison`` — require BOTH compared terms to appear,
                 unless cosine is strong enough to trust semantic match.
            3. Reject chunks that share no query term and have weak cosine.
            4. Reject chunks below the relevance floor.
            5. Sort by blended score and return top-k * 3.

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
        cosine = (self.chunk_matrix @ question_vec.T).toarray().ravel()
        question_tokens = _tokenize(question) or [question.lower()]
        question_token_set = set(question_tokens)
        bm25_scores = np.array(self.bm25.get_scores(question_tokens))
        graph_hits = self._graph_hit_scores(question)

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
            semantic_cos = float(cosine[idx])
            keyword = float(bm25_norm[idx])
            graph_score = graph_hits.get(chunk.chunk_id, 0.0)
            blended = _W_COSINE * semantic_cos + _W_BM25 * keyword

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
                # Pass if BOTH terms present, OR cosine strong (semantic match).
                comparison_pass = (has_a and has_b) or semantic_cos >= _STRONG_COSINE

            chunk_tokens = self._tokenized[idx] if idx < len(self._tokenized) else _tokenize(chunk.text)
            shares_term = bool(question_token_set.intersection(chunk_tokens))
            cosine_strong = semantic_cos >= _STRONG_COSINE

            blend_strong = blended >= min_blend
            cos_above_floor = semantic_cos >= min_cosine
            graph_strong = graph_score > 0

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
                "graph_score": graph_score,
                "confidence_score": 0.5 + min(0.5, max(0.0, blended)),
                "trust_score": trust_lookup.get(source_id, 0.5),
                "uncertainty_penalty": safe_entropy(0.5 + min(0.5, max(0.0, blended))),
                "raw_relevance": blended,
                "raw_cosine": semantic_cos,
                "raw_bm25": keyword,
                "list_bonus": list_bonus,
            }

            if not comparison_pass:
                rejected.append({**base_row, "reason": "missing_comparison_term"})
                continue
            if not loose and not (shares_term or cosine_strong):
                rejected.append({**base_row, "reason": "no_term_overlap"})
                continue
            if not (blend_strong or cos_above_floor or graph_strong):
                rejected.append({**base_row, "reason": "below_threshold"})
                continue
            rows.append(base_row)

        rows.sort(key=lambda item: item["semantic_score"], reverse=True)
        kept = rows[: max(top_k * 3, 10)]

        # Structured per-query log: kept and rejected chunks with their raw
        # signals. Use INFO so it shows up in standard backend logs.
        log_event(
            logger,
            "retrieval.query",
            {
                "question": question[:120],
                "tokens": question_tokens,
                "kept": [
                    {
                        "id": r["id"],
                        "cos": round(r["raw_cosine"], 3),
                        "bm25": round(r["raw_bm25"], 3),
                        "blend": round(r["raw_relevance"], 3),
                        "graph": round(r["graph_score"], 3),
                    }
                    for r in kept
                ],
                "rejected": [
                    {
                        "id": r["id"],
                        "cos": round(float(r["raw_cosine"]), 3),
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
