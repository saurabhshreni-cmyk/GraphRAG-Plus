"""Rule-based extraction with adaptive fallback hooks.

The extractor combines three complementary signals so that technical text
("graph is a data structure consisting of nodes and edges") yields meaningful
entities even though none of the key terms are capitalized:

1. **Capitalized noun phrases** — proper-noun style spans (e.g. "GraphRAG",
   "Trust Manager"). Useful for product / proper noun text.
2. **Domain keyword whitelist** — single-token technical terms ("graph",
   "node", "edge", "vertex", "matrix", "algorithm", ...). This is what makes
   the graph populate for CS / data-structure content where surface
   capitalization is absent.
3. **Salient noun phrases** — sequences of two+ alphabetic tokens that aren't
   stopwords and that appear at non-trivial frequency in the chunk. This
   catches multi-word concepts ("data structure", "edge weights").

Relations are extracted with the existing predicate vocabulary plus a
copular pattern ("X is a Y", "X has Y") that lets us build "is_a" / "has"
edges from declarative sentences — enough for the graph to show structure
without an LLM.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import pairwise

from graphrag_plus.app.extraction.models import Entity, Relation
from graphrag_plus.app.ingestion.models import Chunk

# --- regexes -----------------------------------------------------------------

# Two+ alphabetic chars, optional internal hyphen/digit. Whole words only.
_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b")

# Capitalized phrases (Proper Noun, possibly multi-word).
_CAPITAL_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b")

# Predicate vocabulary for typed relations.
_REL_RE = re.compile(
    r"\b(?P<subj>[A-Za-z][A-Za-z0-9_ \-]{1,40}?)\s+"
    r"(?P<pred>acquired|supports|contradicts|causes|follows|precedes|implements|extends)\s+"
    r"(?P<obj>[A-Za-z][A-Za-z0-9_ \-]{1,40})\b",
    re.IGNORECASE,
)

# "X is a Y" / "X is the Y" — copular "is_a" relation.
_IS_A_RE = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9_\- ]{1,40}|[a-z][a-z0-9\-]{2,})\s+is\s+(?:an?|the)\s+"
    r"(?P<obj>[a-zA-Z][a-zA-Z0-9_\- ]{2,60})\b"
)

# "X has Y" / "X contains Y" / "X consists of Y".
_HAS_RE = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9_\- ]{1,40}|[a-z][a-z0-9\-]{2,})\s+"
    r"(?P<pred>has|contains|consists\s+of|includes)\s+"
    r"(?P<obj>[a-zA-Z][a-zA-Z0-9_\- ]{2,60})\b",
    re.IGNORECASE,
)

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# --- vocabularies ------------------------------------------------------------

# Technical / domain keywords. Lower-case lookup. Add categories to drive
# entity_type tagging which gives the graph viz nicer color-coding.
_DOMAIN_KEYWORDS: dict[str, str] = {
    # graph theory
    "graph": "Concept",
    "node": "Concept",
    "nodes": "Concept",
    "vertex": "Concept",
    "vertices": "Concept",
    "edge": "Concept",
    "edges": "Concept",
    "tree": "Concept",
    "matrix": "Concept",
    "adjacency": "Concept",
    "adjacency-matrix": "Concept",
    "weighted": "Concept",
    "directed": "Concept",
    "undirected": "Concept",
    "path": "Concept",
    "cycle": "Concept",
    "subgraph": "Concept",
    # data / cs general
    "data": "Concept",
    "structure": "Concept",
    "data-structure": "Concept",
    "algorithm": "Concept",
    "complexity": "Concept",
    "vector": "Concept",
    "embedding": "Concept",
    "index": "Concept",
    "retrieval": "Concept",
    "tokenization": "Concept",
    "stopwords": "Concept",
    "ranking": "Concept",
    "similarity": "Concept",
    "cosine": "Concept",
    "bm25": "Concept",
    # graph-rag specific
    "rag": "Concept",
    "graph-rag": "Concept",
    "graphrag": "Concept",
    "trust": "Concept",
    "calibration": "Concept",
    "contradiction": "Concept",
    "confidence": "Concept",
    "evidence": "Concept",
    "chunk": "Concept",
    "document": "Concept",
}

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "he",
        "she",
        "his",
        "her",
        "if",
        "then",
        "else",
        "than",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "where",
        "when",
        "why",
        "how",
        "not",
        "no",
        "nor",
        "only",
        "also",
        "such",
        "very",
        "much",
        "many",
        "more",
        "most",
        "some",
        "any",
        "all",
        "each",
        "other",
        "another",
        "same",
        "so",
        "too",
        "either",
        "neither",
    }
)

# Minimum frequency for a multi-word noun phrase to count.
_MIN_PHRASE_COUNT = 1


# --- helpers -----------------------------------------------------------------


def _normalize(text: str) -> str:
    """Collapse whitespace and strip surrounding punctuation."""
    return re.sub(r"\s+", " ", text).strip(" .,;:()[]{}\"'")


def _is_meaningful(text: str) -> bool:
    if len(text) < 3:
        return False
    return text.lower() not in _STOPWORDS


def _add_entity(
    bag: dict[tuple[str, str], Entity],
    text: str,
    entity_type: str,
    chunk_id: str,
    confidence: float,
    method: str,
) -> None:
    """De-dupe entities per (lowercased text, chunk_id) so the graph stays
    compact even when a term repeats inside a chunk."""
    text = _normalize(text)
    if not _is_meaningful(text):
        return
    key = (text.lower(), chunk_id)
    if key in bag:
        # Boost confidence on repeat sightings, capped at 0.95.
        existing = bag[key]
        bag[key] = Entity(
            text=existing.text,
            entity_type=existing.entity_type,
            confidence=min(0.95, existing.confidence + 0.05),
            method=existing.method,
            source_chunk_id=existing.source_chunk_id,
        )
        return
    bag[key] = Entity(
        text=text,
        entity_type=entity_type,
        confidence=confidence,
        method=method,
        source_chunk_id=chunk_id,
    )


# --- public ------------------------------------------------------------------


def extract_from_chunks(chunks: list[Chunk]) -> tuple[list[Entity], list[Relation]]:
    """Extract entities and relations from chunks.

    Returns a tuple of (entities, relations). Entities are de-duped per
    (text_lower, chunk_id). Relations are not de-duped because contradiction
    detection benefits from seeing the raw frequency of each (subj, pred, obj).
    """
    entities: dict[tuple[str, str], Entity] = {}
    relations: list[Relation] = []

    for chunk in chunks:
        if not chunk.text or not chunk.text.strip():
            continue
        chunk_dates = _DATE_RE.findall(chunk.text)

        # 1. Capitalized phrases (proper-noun style).
        for match in _CAPITAL_PHRASE_RE.finditer(chunk.text):
            text = match.group(1)
            _add_entity(entities, text, "Entity", chunk.chunk_id, 0.75, "regex_capitalized")

        # 2. Domain-keyword whitelist (technical terms).
        tokens = _TOKEN_RE.findall(chunk.text)
        token_counter = Counter(tok.lower() for tok in tokens)
        for token in tokens:
            lowered = token.lower()
            if lowered in _DOMAIN_KEYWORDS:
                _add_entity(
                    entities,
                    token,
                    _DOMAIN_KEYWORDS[lowered],
                    chunk.chunk_id,
                    0.7,
                    "domain_keyword",
                )

        # 3. Salient bigrams (e.g. "data structure", "adjacency matrix").
        # Skip bigrams that are mostly stopwords.
        for left, right in pairwise(tokens):
            if left.lower() in _STOPWORDS or right.lower() in _STOPWORDS:
                continue
            if len(left) < 3 or len(right) < 3:
                continue
            phrase = f"{left} {right}"
            phrase_lower = phrase.lower()
            # Surface bigrams whose at least one half is a domain keyword OR
            # which appear more than once in the chunk.
            in_domain = left.lower() in _DOMAIN_KEYWORDS or right.lower() in _DOMAIN_KEYWORDS
            count = sum(1 for lhs, rhs in pairwise(tokens) if f"{lhs} {rhs}".lower() == phrase_lower)
            if not (in_domain or count >= _MIN_PHRASE_COUNT * 2):
                continue
            _add_entity(
                entities,
                phrase,
                "Phrase",
                chunk.chunk_id,
                0.65,
                "salient_phrase",
            )

        # 4. Typed relations (acquired / supports / contradicts / ...).
        for rel_match in _REL_RE.finditer(chunk.text):
            predicate = rel_match.group("pred").lower()
            stance = "neutral"
            if predicate == "supports":
                stance = "supports"
            elif predicate == "contradicts":
                stance = "contradicts"
            relations.append(
                Relation(
                    subject=_normalize(rel_match.group("subj")),
                    predicate=predicate,
                    obj=_normalize(rel_match.group("obj")),
                    stance=stance,
                    confidence=0.7 if stance == "neutral" else 0.8,
                    method="regex_relation",
                    source_chunk_id=chunk.chunk_id,
                    timestamp=chunk_dates[0] if chunk_dates else chunk.timestamp,
                )
            )

        # 5. Copular "is_a" relation.
        for is_match in _IS_A_RE.finditer(chunk.text):
            subj = _normalize(is_match.group("subj"))
            obj = _normalize(is_match.group("obj"))
            if not (_is_meaningful(subj) and _is_meaningful(obj)):
                continue
            # Trim object at the first conjunction / preposition / clause
            # boundary so we don't capture half a paragraph.
            obj = re.split(
                r"\b(?:and|or|but|that|which|where|when|consisting|containing|with|of|in|for)\b",
                obj,
                maxsplit=1,
            )[0]
            obj = _normalize(obj)
            # Cap object at ~5 tokens so the graph node label stays scannable.
            obj_tokens = obj.split()
            if len(obj_tokens) > 5:
                obj = " ".join(obj_tokens[:5])
            if not _is_meaningful(obj):
                continue
            relations.append(
                Relation(
                    subject=subj,
                    predicate="is_a",
                    obj=obj,
                    stance="neutral",
                    confidence=0.7,
                    method="copular_is_a",
                    source_chunk_id=chunk.chunk_id,
                    timestamp=chunk_dates[0] if chunk_dates else chunk.timestamp,
                )
            )

        # 6. Has / contains / consists-of relations.
        for has_match in _HAS_RE.finditer(chunk.text):
            subj = _normalize(has_match.group("subj"))
            obj = _normalize(has_match.group("obj"))
            if not (_is_meaningful(subj) and _is_meaningful(obj)):
                continue
            obj = re.split(r"\b(?:and|or|but|that|which|where|when)\b", obj, maxsplit=1)[0]
            obj = _normalize(obj)
            if not _is_meaningful(obj):
                continue
            predicate = has_match.group("pred").lower().replace(" ", "_")
            relations.append(
                Relation(
                    subject=subj,
                    predicate=predicate,
                    obj=obj,
                    stance="neutral",
                    confidence=0.65,
                    method="copular_has",
                    source_chunk_id=chunk.chunk_id,
                    timestamp=chunk_dates[0] if chunk_dates else chunk.timestamp,
                )
            )

        # Use token_counter to slightly boost entities that recurred.
        for (text_lower, cid), entity in list(entities.items()):
            if cid != chunk.chunk_id:
                continue
            if token_counter.get(text_lower, 0) >= 2:
                entities[(text_lower, cid)] = Entity(
                    text=entity.text,
                    entity_type=entity.entity_type,
                    confidence=min(0.95, entity.confidence + 0.05),
                    method=entity.method,
                    source_chunk_id=entity.source_chunk_id,
                )

    return list(entities.values()), relations


def should_trigger_fallback(confidence: float, threshold: float) -> bool:
    """Adaptive fallback gate for potential LLM enrichment."""
    return confidence < threshold
