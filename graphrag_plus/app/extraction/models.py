"""Extraction models."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Entity:
    """Extracted entity."""

    text: str
    entity_type: str
    confidence: float
    method: str
    source_chunk_id: str

    def to_dict(self) -> dict[str, object]:
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
    timestamp: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
