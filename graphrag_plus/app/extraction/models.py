"""Extraction models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class Entity:
    """Extracted entity."""

    text: str
    entity_type: str
    confidence: float
    method: str
    source_chunk_id: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class Relation:
    """Extracted relation."""

    subject: str
    predicate: str
    obj: str
    stance: str
    confidence: float
    method: str
    source_chunk_id: str
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

