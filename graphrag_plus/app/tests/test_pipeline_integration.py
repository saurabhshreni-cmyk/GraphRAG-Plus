"""Pipeline integration test."""

from pathlib import Path

from graphrag_plus.app.config.settings import Settings
from graphrag_plus.app.pipeline import GraphRAGPipeline
from graphrag_plus.app.schemas.models import QueryRequest


def test_pipeline_ingest_query_roundtrip(tmp_path: Path) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "Nova Dynamics supports Orion Labs strategy on 2024-01-15. "
        "Another report contradicts cancellation of Project Helios.",
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "data" / "reports",
        graph_path=tmp_path / "data" / "graph.json",
        graph_versions_dir=tmp_path / "data" / "graph_versions",
        trust_state_path=tmp_path / "data" / "trust.json",
        calibration_state_path=tmp_path / "data" / "calibration.json",
        review_queue_path=tmp_path / "data" / "queue.jsonl",
        answers_log_path=tmp_path / "data" / "answers.jsonl",
    )
    pipeline = GraphRAGPipeline(settings)
    ingest_res = pipeline.ingest([str(doc)], [])
    assert ingest_res.documents == 1

    query_res = pipeline.query(QueryRequest(question="What contradicts the cancellation claim?", analyst_mode=True))
    assert query_res.answer
    assert query_res.raw_confidence >= 0.0
    assert query_res.calibrated_confidence >= 0.0
    assert query_res.failure_type is not None or query_res.evidence is not None

