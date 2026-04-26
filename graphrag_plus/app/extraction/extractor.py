"""Rule-based extraction with adaptive fallback hooks."""

from __future__ import annotations

import re
from typing import List, Tuple

from graphrag_plus.app.extraction.models import Entity, Relation
from graphrag_plus.app.ingestion.models import Chunk


ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b")
REL_PATTERN = re.compile(
    r"\b(?P<subj>[A-Z][a-zA-Z0-9_ ]+?)\s+(?P<pred>acquired|supports|contradicts|causes|follows|precedes)\s+(?P<obj>[A-Z][a-zA-Z0-9_ ]+)\b",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def extract_from_chunks(chunks: List[Chunk]) -> Tuple[List[Entity], List[Relation]]:
    """Extract entities and relations from chunks."""
    entities: List[Entity] = []
    relations: List[Relation] = []
    for chunk in chunks:
        chunk_dates = DATE_PATTERN.findall(chunk.text)
        for match in ENTITY_PATTERN.finditer(chunk.text):
            text = match.group(1).strip()
            if len(text) < 3:
                continue
            entities.append(
                Entity(
                    text=text,
                    entity_type="Entity",
                    confidence=0.75,
                    method="regex_ner",
                    source_chunk_id=chunk.chunk_id,
                )
            )
        for rel_match in REL_PATTERN.finditer(chunk.text):
            predicate = rel_match.group("pred").lower()
            stance = "neutral"
            if predicate in {"supports"}:
                stance = "supports"
            if predicate in {"contradicts"}:
                stance = "contradicts"
            relations.append(
                Relation(
                    subject=rel_match.group("subj").strip(),
                    predicate=predicate,
                    obj=rel_match.group("obj").strip(),
                    stance=stance,
                    confidence=0.7 if stance == "neutral" else 0.8,
                    method="regex_relation",
                    source_chunk_id=chunk.chunk_id,
                    timestamp=chunk_dates[0] if chunk_dates else chunk.timestamp,
                )
            )
    return entities, relations


def should_trigger_fallback(confidence: float, threshold: float) -> bool:
    """Adaptive fallback gate for potential LLM enrichment."""
    return confidence < threshold

