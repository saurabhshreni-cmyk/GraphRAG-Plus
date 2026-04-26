"""Hybrid retrieval service."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from graphrag_plus.app.graph.store import GraphStore
from graphrag_plus.app.ingestion.models import Chunk
from graphrag_plus.app.utils.math_utils import safe_entropy


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
    """Vector + BM25 + graph retrieval."""

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer()
        self.chunk_matrix = None
        self.bm25: BM25Okapi | None = None

    def build_indexes(self, chunks: list[Chunk]) -> None:
        """Build in-memory indexes."""
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        if not texts:
            self.chunk_matrix = None
            self.bm25 = None
            return
        self.chunk_matrix = self.vectorizer.fit_transform(texts)
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, question: str, top_k: int, trust_lookup: dict[str, float]) -> list[dict[str, float]]:
        """Retrieve candidates with base scores."""
        if not self.chunks or self.chunk_matrix is None or self.bm25 is None:
            return []

        question_vec = self.vectorizer.transform([question])
        cosine = (self.chunk_matrix @ question_vec.T).toarray().ravel()
        bm25_scores = np.array(self.bm25.get_scores(question.lower().split()))
        graph_hits = self._graph_hit_scores(question)

        rows: list[dict[str, float]] = []
        for idx, chunk in enumerate(self.chunks):
            source_id = chunk.doc_id
            semantic = float(cosine[idx])
            keyword = float(bm25_scores[idx])
            graph_score = graph_hits.get(chunk.chunk_id, 0.0)
            base_confidence = 0.5 + min(0.5, max(0.0, semantic))
            trust_score = trust_lookup.get(source_id, 0.5)
            uncertainty = safe_entropy(base_confidence)
            rows.append(
                {
                    "id": chunk.chunk_id,
                    "source_id": source_id,
                    "snippet": chunk.text[:300],
                    "semantic_score": 0.6 * semantic + 0.4 * keyword,
                    "graph_score": graph_score,
                    "confidence_score": base_confidence,
                    "trust_score": trust_score,
                    "uncertainty_penalty": uncertainty,
                }
            )
        rows.sort(key=lambda item: item["semantic_score"], reverse=True)
        return rows[: max(top_k * 3, 10)]

    def _graph_hit_scores(self, question: str) -> dict[str, float]:
        keywords = {token.lower() for token in question.split() if len(token) > 2}
        scores: dict[str, float] = {}
        for node_id, attrs in self.graph_store.graph.nodes(data=True):
            label = str(attrs.get("label", "")).lower()
            if not label:
                continue
            overlap = len(keywords.intersection(set(label.split())))
            if overlap <= 0:
                continue
            for pred in self.graph_store.graph.predecessors(node_id):
                if pred.startswith("doc_") or "_ch_" in pred:
                    scores[pred] = scores.get(pred, 0.0) + float(overlap)
        return scores
