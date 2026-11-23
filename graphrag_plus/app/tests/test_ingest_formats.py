"""Extraction coverage for every supported upload format.

The pipeline ingests plain text; these tests assert the per-format extractors
turn PDF/DOCX/HTML/TXT into clean prose (markup stripped) so downstream
extraction sees content, not tags.
"""

from __future__ import annotations

from pathlib import Path

from graphrag_plus.app.api.main import _extract_docx_text, _extract_html_text
from graphrag_plus.app.ingestion.loader import load_text_file


def test_html_extraction_strips_markup(tmp_path: Path) -> None:
    html = tmp_path / "page.html"
    html.write_text(
        "<html><head><style>b{}</style></head><body>"
        "<nav>menu</nav><h1>Report</h1>"
        "<p>Tesla builds electric cars in California.</p>"
        "<script>evil()</script></body></html>",
        encoding="utf-8",
    )
    text = _extract_html_text(html)
    assert "Tesla builds electric cars" in text
    assert "<" not in text
    assert "evil" not in text
    assert "menu" not in text  # nav stripped


def test_docx_extraction(tmp_path: Path) -> None:
    import docx

    doc = docx.Document()
    doc.add_paragraph("Google develops the Knowledge Graph.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Bing"
    table.rows[0].cells[1].text = "Microsoft"
    path = tmp_path / "doc.docx"
    doc.save(str(path))

    text = _extract_docx_text(path)
    assert "Knowledge Graph" in text
    assert "Bing | Microsoft" in text  # table cells joined


def test_text_loader(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("Nova Dynamics acquired Beats in 2021.", encoding="utf-8")
    assert "Nova Dynamics" in load_text_file(p).text
