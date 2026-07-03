"""Two-stage entity/relation extraction: spaCy NER + Ollama LLM.

Stage 1 — **spaCy** (``en_core_web_sm``): fast statistical NER for
PERSON / ORG / GPE / LOC / DATE / PRODUCT spans. Always runs.

Stage 2 — **Ollama** (``qwen2.5:3b`` by default): the chunk text plus the
spaCy entities are handed to a local LLM that returns strictly-validated
JSON (:class:`~graphrag_plus.app.models.schemas.ExtractionResult`). The LLM
adds concept/technology entities and, crucially, typed relationships that
statistical NER cannot produce. Up to ``_LLM_MAX_ATTEMPTS`` attempts; on
failure the spaCy-only result is returned so ingestion never blocks on the
LLM.

The legacy regex extractor is preserved in
:mod:`graphrag_plus.app.extraction.legacy_extractor` and used as the final
fallback when the spaCy model is unavailable.

Environment knobs (all optional):

* ``OLLAMA_BASE_URL``                    — Ollama daemon (default localhost:11434)
* ``OLLAMA_MODEL``                       — model tag (default qwen2.5:3b)
* ``GRAPHRAG_LLM_EXTRACTION``            — "0" disables stage 2 entirely
* ``GRAPHRAG_LLM_EXTRACTION_MAX_CHUNKS`` — LLM budget per ingest (default 8);
  chunks beyond the budget get spaCy-only extraction so large documents
  ingest in bounded time.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from graphrag_plus.app.extraction.models import Entity as LegacyEntity
from graphrag_plus.app.extraction.models import Relation as LegacyRelation
from graphrag_plus.app.ingestion.models import Chunk
from graphrag_plus.app.models.schemas import (
    Entity,
    EntityType,
    ExtractionResult,
    Relationship,
)
from graphrag_plus.app.utils.logging_utils import get_logger

load_dotenv()  # OLLAMA_* from the project .env must be visible at import time

logger = get_logger(__name__)

# --- configuration -----------------------------------------------------------

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
# 120s default: the first call after idle pays the model cold-load (~30-60s
# for a 4B model on 16GB RAM) on top of generation. Warm calls run in
# 10-25s; the per-ingest chunk budget keeps total ingest time bounded.
_OLLAMA_TIMEOUT_S = float(os.environ.get("GRAPHRAG_LLM_EXTRACTION_TIMEOUT_S", "120"))
_LLM_MAX_ATTEMPTS = 2

_SPACY_MODEL = "en_core_web_sm"
# spaCy label → schema EntityType. Labels outside this map are ignored at the
# spaCy stage (the LLM stage may still surface them as CONCEPT/OTHER).
_SPACY_LABEL_MAP: dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORG,
    "GPE": EntityType.LOCATION,
    "LOC": EntityType.LOCATION,
    "DATE": EntityType.DATE,
    "PRODUCT": EntityType.TECHNOLOGY,
}

_SYSTEM_PROMPT = """You are an expert at extracting structured information from text.
Given text and pre-identified entities, extract:
1. All entities with their types and brief descriptions
2. All relationships between entities

You MUST respond with ONLY valid JSON in this exact format, nothing else:
{
  "entities": [
    {"name": "entity name", "type": "PERSON|ORG|LOCATION|DATE|CONCEPT|TECHNOLOGY|OTHER", "description": "brief description", "confidence": 0.95}
  ],
  "relationships": [
    {"source": "entity1", "target": "entity2", "relation": "relation_type", "confidence": 0.9}
  ]
}"""

# Bare temporal tokens that are noise as standalone graph nodes ("Monday",
# "January", "2024"). Multi-word DATE spans like "January 2024" survive.
_MONTHS = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    }
)  # fmt: skip
_WEEKDAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
_TEMPORAL_WORDS = _MONTHS | _WEEKDAYS

_VALID_TYPES = {member.value for member in EntityType}

# --- lazy singletons ----------------------------------------------------------

_nlp: Any = None
_nlp_failed = False
_ollama_client: Any = None


def _get_nlp() -> Any:
    """Load the spaCy pipeline once. Returns None when unavailable."""
    global _nlp, _nlp_failed
    if _nlp is not None or _nlp_failed:
        return _nlp
    try:
        import spacy

        _nlp = spacy.load(_SPACY_MODEL, disable=["lemmatizer"])
        logger.info("extractor.spacy_loaded model=%s", _SPACY_MODEL)
    except Exception as exc:
        _nlp_failed = True
        logger.warning("extractor.spacy_unavailable error=%s — legacy regex fallback active", exc)
    return _nlp


def _get_ollama_client() -> Any:
    """Build the Ollama client once. Returns None when the SDK is missing."""
    global _ollama_client
    if _ollama_client is not None:
        return _ollama_client
    try:
        import ollama

        _ollama_client = ollama.Client(host=_OLLAMA_BASE_URL, timeout=_OLLAMA_TIMEOUT_S)
    except Exception as exc:
        logger.warning("extractor.ollama_client_failed error=%s", exc)
    return _ollama_client


def _llm_extraction_enabled() -> bool:
    return os.environ.get("GRAPHRAG_LLM_EXTRACTION", "1").strip().lower() not in {"0", "false", "no"}


def _llm_chunk_budget() -> int:
    try:
        return max(0, int(os.environ.get("GRAPHRAG_LLM_EXTRACTION_MAX_CHUNKS", "8")))
    except ValueError:
        return 8


# --- helpers -------------------------------------------------------------------


def _is_temporal_noise(text: str) -> bool:
    """True for bare month/weekday/year/number tokens ("Monday", "2024")."""
    if " " in text or "-" in text:
        return False
    lowered = text.lower()
    return lowered in _TEMPORAL_WORDS or lowered.isdigit()


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name or "").strip(" .,;:()[]{}\"'")


def _spacy_extract(text: str, chunk_id: str) -> ExtractionResult:
    """Stage 1: statistical NER. Returns an empty result if spaCy is missing."""
    nlp = _get_nlp()
    if nlp is None:
        return ExtractionResult(chunk_id=chunk_id)
    doc = nlp(text)
    seen: set[str] = set()
    entities: list[Entity] = []
    for ent in doc.ents:
        entity_type = _SPACY_LABEL_MAP.get(ent.label_)
        if entity_type is None:
            continue
        name = _clean_name(ent.text)
        if len(name) < 2 or len(name) > 80 or _is_temporal_noise(name):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            Entity(
                name=name,
                type=entity_type,
                description=f"spaCy {ent.label_} entity",
                confidence=0.85,
            )
        )
    return ExtractionResult(entities=entities, chunk_id=chunk_id)


def _extract_json_block(raw: str) -> str:
    """Pull the outermost JSON object out of an LLM response.

    Local models occasionally wrap the payload in markdown fences or prose;
    slicing from the first ``{`` to the last ``}`` recovers it.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in LLM response")
    return cleaned[start : end + 1]


def _coerce_payload(payload: dict[str, Any], chunk_id: str) -> ExtractionResult:
    """Validate the LLM payload strictly, dropping only malformed rows.

    Unknown entity types are coerced to OTHER rather than failing the whole
    chunk — qwen-class models sometimes invent labels like "FIELD".
    """
    entities: list[Entity] = []
    for row in payload.get("entities", []) or []:
        if not isinstance(row, dict):
            continue
        name = _clean_name(str(row.get("name", "")))
        if not name or len(name) > 80 or _is_temporal_noise(name):
            continue
        raw_type = str(row.get("type", "OTHER")).upper().strip()
        row_type = raw_type if raw_type in _VALID_TYPES else "OTHER"
        try:
            confidence = float(row.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        entities.append(
            Entity(
                name=name,
                type=EntityType(row_type),
                description=(str(row.get("description")) or None) if row.get("description") else None,
                confidence=min(1.0, max(0.0, confidence)),
            )
        )

    relationships: list[Relationship] = []
    for row in payload.get("relationships", []) or []:
        if not isinstance(row, dict):
            continue
        source = _clean_name(str(row.get("source", "")))
        target = _clean_name(str(row.get("target", "")))
        relation = re.sub(r"\s+", "_", str(row.get("relation", "related_to")).strip().lower())
        if not source or not target or source.lower() == target.lower():
            continue
        # Endpoints become graph nodes (backfilled below), so the same
        # temporal-noise gate that guards entities must guard them too —
        # otherwise bare "2024" / "January" nodes leak in via relationships.
        if _is_temporal_noise(source) or _is_temporal_noise(target):
            continue
        try:
            confidence = float(row.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        relationships.append(
            Relationship(
                source=source,
                target=target,
                relation=relation or "related_to",
                confidence=min(1.0, max(0.0, confidence)),
            )
        )

    # Every relationship endpoint must exist as an entity so the graph never
    # gets dangling edges. Backfill missing endpoints as OTHER.
    known = {e.name.lower() for e in entities}
    for rel in relationships:
        for endpoint in (rel.source, rel.target):
            if endpoint.lower() not in known:
                known.add(endpoint.lower())
                entities.append(
                    Entity(name=endpoint, type=EntityType.OTHER, confidence=rel.confidence)
                )

    return ExtractionResult(entities=entities, relationships=relationships, chunk_id=chunk_id)


def _llm_extract(text: str, chunk_id: str, spacy_entities: list[Entity]) -> ExtractionResult | None:
    """Stage 2: LLM extraction with retries. None means "use spaCy result"."""
    client = _get_ollama_client()
    if client is None:
        return None

    hints = ", ".join(f"{e.name} ({e.type.value})" for e in spacy_entities[:25]) or "none"
    user_prompt = f"Pre-identified entities: {hints}\n\nText:\n{text}"

    for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
        try:
            response = client.chat(
                model=_OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                # think=False: reasoning-capable models (qwen3.5, deepseek-r1)
                # would otherwise emit long hidden reasoning before the JSON,
                # blowing the timeout. Non-thinking models ignore the flag.
                think=False,
                options={"temperature": 0.0, "num_predict": 1200},
            )
            raw = response["message"]["content"]
            payload = json.loads(_extract_json_block(raw))
            result = _coerce_payload(payload, chunk_id)
            # A valid-but-empty payload on attempt 1 is usually a truncated
            # generation — retry once before accepting it.
            if not result.entities and attempt < _LLM_MAX_ATTEMPTS:
                continue
            return result
        except Exception as exc:
            logger.warning(
                "extractor.llm_attempt_failed chunk=%s attempt=%d/%d error=%s",
                chunk_id,
                attempt,
                _LLM_MAX_ATTEMPTS,
                str(exc)[:200],
            )
    return None


def _merge_results(spacy_result: ExtractionResult, llm_result: ExtractionResult) -> ExtractionResult:
    """Union of both stages, LLM rows winning on name collisions.

    The LLM sees the spaCy entities in its prompt and usually re-emits them
    with better types/descriptions; keeping its version avoids duplicates
    while spaCy-only spans the LLM dropped are preserved.
    """
    merged: dict[str, Entity] = {e.name.lower(): e for e in spacy_result.entities}
    for entity in llm_result.entities:
        merged[entity.name.lower()] = entity
    return ExtractionResult(
        entities=list(merged.values()),
        relationships=llm_result.relationships,
        chunk_id=llm_result.chunk_id or spacy_result.chunk_id,
    )


# --- public API ------------------------------------------------------------------


def extract(text: str, chunk_id: str = "", *, use_llm: bool | None = None) -> ExtractionResult:
    """Extract entities and relationships from one chunk of text.

    ``use_llm`` overrides the ``GRAPHRAG_LLM_EXTRACTION`` env default —
    the batch wrapper uses it to enforce a per-ingest LLM budget.
    """
    if not text or not text.strip():
        return ExtractionResult(chunk_id=chunk_id)

    spacy_result = _spacy_extract(text, chunk_id)

    llm_allowed = _llm_extraction_enabled() if use_llm is None else use_llm
    if llm_allowed:
        llm_result = _llm_extract(text, chunk_id, spacy_result.entities)
        if llm_result is not None:
            result = _merge_results(spacy_result, llm_result)
            logger.info(
                "extractor.chunk_done chunk=%s entities=%d relations=%d method=spacy+llm",
                chunk_id,
                len(result.entities),
                len(result.relationships),
            )
            return result

    logger.info(
        "extractor.chunk_done chunk=%s entities=%d relations=%d method=spacy",
        chunk_id,
        len(spacy_result.entities),
        len(spacy_result.relationships),
    )
    return spacy_result


# --- legacy-model bridge (pipeline / NetworkX store compatibility) ----------------


def _to_legacy(result: ExtractionResult, chunk: Chunk) -> tuple[list[LegacyEntity], list[LegacyRelation]]:
    entities = [
        LegacyEntity(
            text=entity.name,
            entity_type=entity.type.value.capitalize(),
            confidence=entity.confidence,
            method="spacy_llm",
            source_chunk_id=chunk.chunk_id,
        )
        for entity in result.entities
    ]
    relations = []
    for rel in result.relationships:
        stance = "neutral"
        if rel.relation == "supports":
            stance = "supports"
        elif rel.relation == "contradicts":
            stance = "contradicts"
        relations.append(
            LegacyRelation(
                subject=rel.source,
                predicate=rel.relation,
                obj=rel.target,
                stance=stance,
                confidence=rel.confidence,
                method="spacy_llm",
                source_chunk_id=chunk.chunk_id,
                timestamp=chunk.timestamp,
            )
        )
    return entities, relations


def extract_from_chunks(chunks: list[Chunk]) -> tuple[list[LegacyEntity], list[LegacyRelation]]:
    """Batch extraction over chunks, returning legacy dataclasses.

    Drop-in replacement for the old regex extractor's entry point — the
    pipeline and NetworkX graph store keep working unchanged.

    * spaCy runs on every chunk.
    * The LLM runs on the first ``GRAPHRAG_LLM_EXTRACTION_MAX_CHUNKS``
      chunks (default 8) so ingesting a large document stays bounded;
      remaining chunks get spaCy-only extraction.
    * If spaCy itself is unavailable, the legacy regex extractor handles
      the whole batch.
    """
    if _get_nlp() is None:
        from graphrag_plus.app.extraction import legacy_extractor

        logger.warning("extractor.using_legacy_regex chunks=%d", len(chunks))
        return legacy_extractor.extract_from_chunks(chunks)

    llm_budget = _llm_chunk_budget() if _llm_extraction_enabled() else 0
    all_entities: list[LegacyEntity] = []
    all_relations: list[LegacyRelation] = []
    llm_used = 0
    for chunk in chunks:
        if not chunk.text or not chunk.text.strip():
            continue
        use_llm = llm_used < llm_budget
        result = extract(chunk.text, chunk.chunk_id, use_llm=use_llm)
        if use_llm:
            llm_used += 1
        entities, relations = _to_legacy(result, chunk)
        all_entities.extend(entities)
        all_relations.extend(relations)

    logger.info(
        "extractor.batch_done chunks=%d llm_chunks=%d entities=%d relations=%d",
        len(chunks),
        llm_used,
        len(all_entities),
        len(all_relations),
    )
    return all_entities, all_relations


def should_trigger_fallback(confidence: float, threshold: float) -> bool:
    """Adaptive fallback gate for potential LLM enrichment (legacy API)."""
    return confidence < threshold
