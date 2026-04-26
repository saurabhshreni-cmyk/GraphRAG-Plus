"""Ingestion domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Document:
    """Loaded document."""

    doc_id: str
    source: str
    text: str
    metadata: Dict[str, str]


@dataclass
class Chunk:
    """Chunked document text."""

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    timestamp: Optional[str]

