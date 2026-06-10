"""Extraction should not promote bare temporal tokens to graph nodes."""

from graphrag_plus.app.extraction.extractor import extract_from_chunks
from graphrag_plus.app.ingestion.models import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="doc_x_ch_0",
        doc_id="doc_x",
        text=text,
        start=0,
        end=len(text),
        timestamp=None,
    )


def test_bare_months_and_years_are_not_entities() -> None:
    text = (
        "In January 2024, DeepSeek released a model. In April, a revision shipped, "
        "and by December 2024 the V3 base was available on Monday."
    )
    entities, _ = extract_from_chunks([_chunk(text)])
    labels = {e.text.lower() for e in entities}
    for noise in ("january", "april", "december", "monday", "2024"):
        assert noise not in labels, f"temporal token leaked as entity: {noise}"


def test_real_named_entities_still_survive() -> None:
    text = (
        "On 2024-01-15, Nova Dynamics acquired Orion Labs. "
        "Report Delta supports continuation under Project Helios."
    )
    entities, _ = extract_from_chunks([_chunk(text)])
    labels = {e.text.lower() for e in entities}
    assert any("nova dynamics" in label for label in labels)
    assert any("orion labs" in label for label in labels)
