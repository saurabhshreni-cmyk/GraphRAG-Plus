"""Chunking logic."""

from __future__ import annotations

import re

from graphrag_plus.app.ingestion.models import Chunk, Document

TIMESTAMP_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split documents with overlap and timestamp hints."""
    chunks: list[Chunk] = []
    for doc in documents:
        text = doc.text
        if not text:
            continue
        start = 0
        idx = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            timestamp_match = TIMESTAMP_PATTERN.search(chunk_text)
            timestamp = timestamp_match.group(1) if timestamp_match else None
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_ch_{idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start=start,
                    end=end,
                    timestamp=timestamp,
                )
            )
            if end == len(text):
                break
            start = max(0, end - chunk_overlap)
            idx += 1
    return chunks
