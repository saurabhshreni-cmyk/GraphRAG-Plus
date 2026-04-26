"""Ingestion tests."""

from pathlib import Path

from graphrag_plus.app.ingestion.chunker import chunk_documents
from graphrag_plus.app.ingestion.loader import load_documents


def test_text_ingestion_and_chunking(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Alpha supports Beta on 2024-01-01.", encoding="utf-8")
    docs = load_documents([str(file_path)], [])
    chunks = chunk_documents(docs, chunk_size=15, chunk_overlap=3)
    assert len(docs) == 1
    assert len(chunks) >= 2

