"""Pipeline integration test."""

from pathlib import Path

from graphrag_plus.app.config.settings import Settings
from graphrag_plus.app.pipeline import GraphRAGPipeline
from graphrag_plus.app.schemas.models import QueryRequest


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "data" / "reports",
        graph_path=tmp_path / "data" / "graph.json",
        chunks_path=tmp_path / "data" / "chunks.json",
        corpora_dir=tmp_path / "data" / "corpora",
        graph_versions_dir=tmp_path / "data" / "graph_versions",
        trust_state_path=tmp_path / "data" / "trust.json",
        calibration_state_path=tmp_path / "data" / "calibration.json",
        review_queue_path=tmp_path / "data" / "queue.jsonl",
        answers_log_path=tmp_path / "data" / "answers.jsonl",
        run_logs_path=tmp_path / "data" / "run_logs.jsonl",
        outputs_dir=tmp_path / "data" / "outputs",
        cache_dir=tmp_path / ".cache",
        temp_dir=tmp_path / ".cache" / "tmp",
    )


def test_pipeline_ingest_query_roundtrip(tmp_path: Path) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "Nova Dynamics supports Orion Labs strategy on 2024-01-15. "
        "Another report contradicts cancellation of Project Helios.",
        encoding="utf-8",
    )
    pipeline = GraphRAGPipeline(_make_settings(tmp_path))
    ingest_res = pipeline.ingest([str(doc)], [])
    assert ingest_res.documents == 1

    query_res = pipeline.query(
        QueryRequest(question="What contradicts the cancellation claim?", analyst_mode=True)
    )
    assert query_res.answer
    assert query_res.raw_confidence >= 0.0
    assert query_res.calibrated_confidence >= 0.0
    assert query_res.failure_type is not None or query_res.evidence is not None


def test_ingest_honors_explicit_corpus_name(tmp_path: Path) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("Nova Dynamics acquired Orion Labs on 2024-01-15.", encoding="utf-8")
    pipeline = GraphRAGPipeline(_make_settings(tmp_path))

    ingest_res = pipeline.ingest([str(doc)], [], corpus_name="My Research Corpus")

    assert ingest_res.corpus_name == "My Research Corpus"
    metas = pipeline.corpus_manager.list()
    assert any(m.name == "My Research Corpus" for m in metas)


def test_ingest_derives_corpus_name_when_not_given(tmp_path: Path) -> None:
    doc = tmp_path / "quarterly_report.txt"
    doc.write_text("Nova Dynamics acquired Orion Labs on 2024-01-15.", encoding="utf-8")
    pipeline = GraphRAGPipeline(_make_settings(tmp_path))

    ingest_res = pipeline.ingest([str(doc)], [])

    assert ingest_res.corpus_name
    assert "quarterly" in ingest_res.corpus_name.lower()
