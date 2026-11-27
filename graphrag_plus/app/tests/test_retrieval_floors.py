"""Relevance floors in FAISS (dense) mode, per embedding backend.

Regression for the cloud deploy: with Gemini embeddings calibrated to
MIN 0.58 / STRONG 0.64, the off-topic question "What is the capital of
France?" still admitted chunks scoring cosine 0.45-0.47 through the
loose fallback, because the blend-score floor was a fixed 0.20 sized for
bge-large. The answer degraded from a clean NO_EVIDENCE refusal to
"Partial answer: I cannot answer...".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphrag_plus.app.graph.store import GraphStore
from graphrag_plus.app.ingestion.models import Chunk
from graphrag_plus.app.retrieval import service as svc
from graphrag_plus.app.retrieval.service import RetrievalService, hybrid_blend_floor

_TEXTS = [
    "Nova Dynamics is a robotics company headquartered in Austin, Texas.",
    "In 2021 Nova Dynamics acquired Helix Sensors, a lidar startup based in Boulder.",
]


@pytest.fixture
def service(tmp_path: Path, monkeypatch) -> RetrievalService:
    """A two-chunk corpus with no graph signal and stubbed dense scores."""
    monkeypatch.setattr(RetrievalService, "_build_faiss", lambda self, chunks, *, persist: None)
    retrieval = RetrievalService(GraphStore(tmp_path / "graph.json"))
    chunks = [
        Chunk(chunk_id=f"c{i}", doc_id="doc", text=text, start=0, end=len(text), timestamp=None)
        for i, text in enumerate(_TEXTS)
    ]
    retrieval._build(chunks, persist=False)
    monkeypatch.setattr(retrieval, "_neo4j_graph_scores", lambda question: {})
    monkeypatch.setattr(retrieval, "_graph_hit_scores", lambda question: {})
    return retrieval


def _with_dense_scores(monkeypatch, retrieval: RetrievalService, cosine: float) -> None:
    scores = {chunk.chunk_id: cosine for chunk in retrieval.chunks}
    monkeypatch.setattr(retrieval, "_semantic_scores", lambda question: (scores, True))


def _gemini_floors(monkeypatch) -> None:
    monkeypatch.setenv("GRAPHRAG_MIN_COSINE_SEMANTIC", "0.58")
    monkeypatch.setenv("GRAPHRAG_STRONG_COSINE_SEMANTIC", "0.64")


class TestHybridBlendFloor:
    def test_follows_a_higher_calibrated_cosine_floor(self):
        assert hybrid_blend_floor(0.58) == pytest.approx(svc._W_SEMANTIC_HYBRID * 0.58)

    def test_never_drops_below_the_original_floor(self):
        # bge-large's floor (0.45) keeps the original 0.20: local behaviour is unchanged.
        assert hybrid_blend_floor(0.45) == svc._MIN_BLEND


class TestOffTopicStaysOut:
    def test_gemini_off_topic_scores_are_rejected_even_in_the_loose_pass(self, service, monkeypatch):
        _gemini_floors(monkeypatch)
        _with_dense_scores(monkeypatch, service, 0.473)  # observed live for "capital of France"
        rows = service.query("What is the capital of France?", 5, {}, loose=True)
        assert rows == []

    def test_gemini_on_topic_paraphrase_still_gets_through(self, service, monkeypatch):
        _gemini_floors(monkeypatch)
        _with_dense_scores(monkeypatch, service, 0.69)  # observed live, zero keyword overlap
        rows = service.query("Who runs the robotics firm?", 5, {})
        assert rows

    def test_bge_defaults_are_unchanged(self, service, monkeypatch):
        monkeypatch.delenv("GRAPHRAG_MIN_COSINE_SEMANTIC", raising=False)
        monkeypatch.delenv("GRAPHRAG_STRONG_COSINE_SEMANTIC", raising=False)
        _with_dense_scores(monkeypatch, service, 0.36)  # bge off-topic ceiling
        assert service.query("What is the capital of France?", 5, {}, loose=True) == []
        _with_dense_scores(monkeypatch, service, 0.70)
        assert service.query("Who runs the robotics firm?", 5, {})
