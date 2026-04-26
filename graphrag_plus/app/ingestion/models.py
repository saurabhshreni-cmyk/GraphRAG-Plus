"""Ingestion domain models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    """Loaded document."""

    doc_id: str
    source: str
    text: str
    metadata: dict[str, str]


@dataclass
class Chunk:
    """Chunked document text."""

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    timestamp: str | None
