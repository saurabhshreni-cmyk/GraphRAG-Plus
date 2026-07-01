"""Pydantic schemas for the upgraded GraphRAG pipeline.

These models are the structured contract between pipeline stages:

* extraction  → :class:`ExtractionResult` (entities + relationships)
* retrieval   → :class:`RetrievalResult` (per-chunk hybrid scores)
* generation  → :class:`AnswerResult` (final answer + provenance)

They are distinct from the legacy dataclasses in
``graphrag_plus.app.extraction.models`` (which the NetworkX store still
consumes) — conversion helpers live in the extractor.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Closed vocabulary of entity types the extractor may emit."""

    PERSON = "PERSON"
    ORG = "ORG"
    LOCATION = "LOCATION"
    DATE = "DATE"
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECHNOLOGY"
    OTHER = "OTHER"


class Entity(BaseModel):
    """A single extracted entity."""

    name: str = Field(description="The entity name exactly as it appears")
    type: EntityType = Field(description="The entity type")
    description: str | None = Field(default=None, description="Brief description")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Relationship(BaseModel):
    """A directed relationship between two extracted entities."""

    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: str = Field(description="Relationship type e.g. works_at, acquired, is_a")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Everything extracted from one chunk of text."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    chunk_id: str = Field(default="")


class RetrievalResult(BaseModel):
    """One retrieved chunk with its per-signal scores."""

    chunk_id: str
    text: str
    score: float
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    graph_score: float = 0.0
    source: str = ""


class AnswerResult(BaseModel):
    """Final generated answer with provenance."""

    answer: str
    confidence: float
    sources: list[str] = Field(default_factory=list)
    entities_used: list[str] = Field(default_factory=list)
    reasoning: str = ""
