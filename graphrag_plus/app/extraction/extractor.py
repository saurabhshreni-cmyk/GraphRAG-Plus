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

Quality controls (added to combat noisy 200+ node graphs):

* **Generic-term blacklist** — words like "system", "method", "data", "model",
  "way", "case" are filtered from capitalized phrases and bigrams (still
  allowed as parts of longer multi-word concepts).
* **Verb / connective stopwords** — "consisting", "represent", "used", "can"
  no longer leak into bigrams.
* **Alias normalization** — "LSTM" ↔ "Long Short-Term Memory" collapse to a
  single canonical entity so the graph shows one node, not two.
* **Global frequency threshold** — entities that appear only once and aren't
  in the domain whitelist or aren't high-confidence are dropped before
  returning, so the noise floor stays low.

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
    # ML / neural networks
    "lstm": "Concept",
    "rnn": "Concept",
    "rnns": "Concept",
    "cnn": "Concept",
    "gru": "Concept",
    "transformer": "Concept",
    "attention": "Concept",
    "neuron": "Concept",
    "neurons": "Concept",
    "network": "Concept",
    "perceptron": "Concept",
    "backpropagation": "Concept",
    "gradient": "Concept",
    "loss": "Concept",
    "regression": "Concept",
    "classification": "Concept",
    "clustering": "Concept",
    "training": "Concept",
    "inference": "Concept",
    "sequence": "Concept",
    "memory": "Concept",
    "gate": "Concept",
    "cell": "Concept",
    "softmax": "Concept",
    "sigmoid": "Concept",
    "tanh": "Concept",
    "relu": "Concept",
}

# Canonical alias map: alias.lower() → canonical text. After normalization,
# any matching surface form is collapsed onto one node so the graph shows
# "long short-term memory" once instead of LSTM, LSTMs, Long Short-Term Memory.
_ALIAS_MAP: dict[str, str] = {
    "lstm": "long short-term memory",
    "lstms": "long short-term memory",
    "long short term memory": "long short-term memory",
    "long short-term memories": "long short-term memory",
    "rnn": "recurrent neural network",
    "rnns": "recurrent neural network",
    "recurrent neural networks": "recurrent neural network",
    "cnn": "convolutional neural network",
    "cnns": "convolutional neural network",
    "convolutional neural networks": "convolutional neural network",
    "gru": "gated recurrent unit",
    "grus": "gated recurrent unit",
    "gan": "generative adversarial network",
    "gans": "generative adversarial network",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "nn": "neural network",
    "nns": "neural network",
    "neural networks": "neural network",
    "memory cells": "memory cell",
    "hidden states": "hidden state",
    # graph theory pluralization
    "nodes": "node",
    "vertices": "vertex",
    "edges": "edge",
    "neurons": "neuron",
}

# Standard grammatical stopwords.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must",
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "we", "us", "our", "you", "your", "i", "me", "my", "he",
        "she", "his", "her", "if", "then", "else", "than", "which", "who",
        "whom", "whose", "what", "where", "when", "why", "how", "not", "no",
        "nor", "only", "also", "such", "very", "much", "many", "more",
        "most", "some", "any", "all", "each", "other", "another", "same",
        "so", "too", "either", "neither", "into", "onto", "upon", "about",
        "between", "among", "through", "during", "before", "after", "above",
        "below", "off", "over", "under", "again", "further",
    }
)  # fmt: skip

# Generic / vague nouns and verbs that shouldn't become standalone entities
# even when they're capitalized at the start of a sentence ("System failed").
# These are still allowed as PARTS of multi-word phrases when paired with a
# domain keyword (e.g. "memory cell" passes because "cell" is in the
# domain whitelist but "memory" is generic alone).
_GENERIC_TERMS = frozenset(
    {
        # generic CS / academic nouns
        "system", "systems", "method", "methods", "model", "models",
        "approach", "approaches", "process", "processes", "result", "results",
        "value", "values", "case", "cases", "type", "types", "kind", "kinds",
        "form", "forms", "part", "parts", "set", "sets", "way", "ways",
        "work", "works", "study", "studies", "research", "paper", "papers",
        "section", "sections", "figure", "figures", "table", "tables",
        "example", "examples", "term", "terms", "field", "fields",
        "problem", "problems", "issue", "issues", "task", "tasks",
        "function", "functions", "variable", "variables", "parameter",
        "parameters", "step", "steps", "stage", "stages", "phase", "phases",
        "level", "levels", "factor", "factors", "feature", "features",
        "thing", "things", "item", "items",
        # information-theoretic-but-too-vague-alone
        "data", "information", "input", "output", "size", "length", "number",
        "amount", "rate", "ratio", "scale", "range", "quality",
        # time / generic frame
        "time", "times", "year", "years", "day", "days", "moment",
        "today", "tomorrow", "yesterday",
        # generic actor nouns
        "person", "people", "user", "users", "group", "groups", "team",
        "name", "names", "author", "authors",
        # connectives / verby noise that leaked into bigrams
        "represent", "represents", "consist", "consists", "consisting",
        "make", "makes", "made", "use", "uses", "used", "using",
        "take", "takes", "took", "give", "gives", "gave", "given",
        "see", "seen", "saw", "say", "says", "said", "show", "shows",
        "shown", "called", "find", "found", "let", "lets", "means",
        "include", "includes", "included", "including",
        "describe", "describes", "described", "provide", "provides",
        "provided", "based", "perform", "performs", "performed",
        "can", "could", "may", "might", "must", "should",
        # additional verbs / participles that left "X verb" bigrams
        "aimed", "tried", "tries", "create", "created", "creates",
        "build", "built", "builds", "develop", "developed", "develops",
        "introduce", "introduced", "introduces", "propose", "proposed",
        "proposes", "designed", "designs", "design", "trained", "train",
        "trains", "applied", "applies", "apply", "implemented", "implement",
        "running", "runs", "ran", "evaluate", "evaluated", "evaluates",
        "compare", "compared", "compares", "consider", "considered",
        "considers", "improve", "improves", "improved", "achieve",
        "achieves", "achieved", "obtain", "obtains", "obtained",
        # stray adjectives commonly preceding/following nouns in prose
        "wide", "narrow", "deep", "shallow", "fast", "slow", "right", "left",
        "above", "below", "inside", "outside", "first", "last", "next",
        "previous", "single", "multiple", "several", "few", "every",
        # vague modifiers
        "general", "specific", "various", "different", "common", "similar",
        "important", "main", "key", "primary", "secondary", "good", "bad",
        "best", "better", "worst", "worse", "high", "low", "large", "small",
        "new", "old", "simple", "complex", "modern", "recent", "current",
        "popular", "standard", "basic", "advanced",
        # adjective-y / aspect words that became cap-phrases
        "however", "although", "since", "because", "therefore", "thus",
        "moreover", "furthermore",
    }
)  # fmt: skip

# Minimum global occurrences for an entity to survive the post-extraction
# filter. Entities below this threshold are kept only if they're high
# confidence or in the domain whitelist.
_MIN_GLOBAL_FREQUENCY = 2
_HIGH_CONFIDENCE_FLOOR = 0.85


# --- helpers -----------------------------------------------------------------


def _normalize(text: str) -> str:
    """Collapse whitespace and strip surrounding punctuation."""
    return re.sub(r"\s+", " ", text).strip(" .,;:()[]{}\"'")


def _canonicalize(text: str) -> str:
    """Apply alias map. Returns the canonical surface form (already _normalize'd).

    Lookup is case-insensitive on the lowercased + stripped text. If an alias
    is found, its canonical replaces the input; otherwise the input is returned
    unchanged (still normalized).
    """
    lowered = text.lower().strip()
    return _ALIAS_MAP.get(lowered, text)


def _is_meaningful(text: str) -> bool:
    """Reject tokens that are too short, too long, too many words, or stopwords.

    The 40-char / 3-token caps catch noisy clauses like
    ``"wide applications in classification"`` or
    ``"The figure on the right"`` that slip past per-token filters.
    """
    if len(text) < 3 or len(text) > 40:
        return False
    if len(text.split()) > 3:
        return False
    return text.lower() not in _STOPWORDS


def _is_generic_single_word(text: str) -> bool:
    """True if ``text`` is a single generic word that shouldn't stand alone.

    Multi-word phrases pass through (so "memory cell" survives even if "cell"
    is generic). Domain-keyword tokens always pass (the whitelist beats the
    blacklist).
    """
    if " " in text or "-" in text:
        return False
    lowered = text.lower()
    if lowered in _DOMAIN_KEYWORDS:
        return False
    return lowered in _GENERIC_TERMS


def _add_entity(
    bag: dict[tuple[str, str], Entity],
    text: str,
    entity_type: str,
    chunk_id: str,
    confidence: float,
    method: str,
) -> None:
    """De-dupe entities per (canonical text, chunk_id) so the graph stays
    compact even when a term repeats inside a chunk.

    Applies normalization + alias map BEFORE de-dup so "LSTM" and "long
    short-term memory" both end up under the same canonical key.
    """
    text = _normalize(text)
    if not _is_meaningful(text):
        return
    # Drop generic single-word noise unless it's a known domain keyword.
    if _is_generic_single_word(text):
        return
    text = _canonicalize(text)
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
    (canonical_text, chunk_id) and then frequency-filtered globally so
    one-off noise tokens don't pollute the graph.
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
        # Skip bigrams where either half is a stopword or a generic noise
        # term — that filters "matrix used", "graph can", "RNN aimed",
        # "memory LSTM", and similar artifacts in one shot.
        for left, right in pairwise(tokens):
            llow = left.lower()
            rlow = right.lower()
            if llow in _STOPWORDS or rlow in _STOPWORDS:
                continue
            if llow in _GENERIC_TERMS or rlow in _GENERIC_TERMS:
                continue
            if len(left) < 3 or len(right) < 3:
                continue
            phrase = f"{left} {right}"
            phrase_lower = phrase.lower()
            in_domain = llow in _DOMAIN_KEYWORDS or rlow in _DOMAIN_KEYWORDS
            count = sum(1 for lhs, rhs in pairwise(tokens) if f"{lhs} {rhs}".lower() == phrase_lower)
            if not (in_domain or count >= 2):
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
                    subject=_canonicalize(_normalize(rel_match.group("subj"))),
                    predicate=predicate,
                    obj=_canonicalize(_normalize(rel_match.group("obj"))),
                    stance=stance,
                    confidence=0.7 if stance == "neutral" else 0.8,
                    method="regex_relation",
                    source_chunk_id=chunk.chunk_id,
                    timestamp=chunk_dates[0] if chunk_dates else chunk.timestamp,
                )
            )

        # 5. Copular "is_a" relation.
        for is_match in _IS_A_RE.finditer(chunk.text):
            subj = _canonicalize(_normalize(is_match.group("subj")))
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
            obj = _canonicalize(_normalize(obj))
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
            subj = _canonicalize(_normalize(has_match.group("subj")))
            obj = _normalize(has_match.group("obj"))
            if not (_is_meaningful(subj) and _is_meaningful(obj)):
                continue
            obj = re.split(r"\b(?:and|or|but|that|which|where|when)\b", obj, maxsplit=1)[0]
            obj = _canonicalize(_normalize(obj))
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

        # Use token_counter to slightly boost entities that recurred within chunk.
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

    # ---------------------- global frequency filter for generic mentions
    filtered = _frequency_filter(list(entities.values()))

    # ---------------------- guarantee: at least SOME entities per chunk -----
    # When the strict filters wipe out a chunk's entities entirely, fall back
    # to a simple noun extractor so downstream graph building always has
    # something to link to. Prevents the "0 entities, no graph" failure mode
    # for unusual / non-technical text.
    chunk_ids_with_entities = {e.source_chunk_id for e in filtered}
    fallback_added: list[Entity] = []
    for chunk in chunks:
        if not chunk.text or chunk.chunk_id in chunk_ids_with_entities:
            continue
        for noun in _fallback_noun_extract(chunk.text):
            fallback_added.append(
                Entity(
                    text=_canonicalize(noun),
                    entity_type="Phrase",
                    confidence=0.55,
                    method="fallback_noun",
                    source_chunk_id=chunk.chunk_id,
                )
            )

    if fallback_added:
        filtered.extend(fallback_added)
    return filtered, relations


# --- fallback noun extraction ------------------------------------------------

# A small set of word-shape heuristics that approximate a noun without a real
# POS tagger: words that are long, mostly alphabetic, and not stopwords or
# generic terms. Used only when the rule-based pass produced zero entities
# for a chunk — its job is "ensure entities > 0", not "be perfect".
_NOUN_CANDIDATE_RE = re.compile(r"\b[A-Za-z][A-Za-z\-]{3,}\b")


def _fallback_noun_extract(text: str, *, limit: int = 5) -> list[str]:
    """Return up to ``limit`` candidate nouns from ``text``.

    Strategy:
      * Tokenize, drop stopwords + generic terms.
      * Prefer multi-word title-case spans first (``"Cash Flow Statement"``).
      * Fall back to the most frequent content-word in the text.
    """
    candidates: list[str] = []

    # Multi-word title-case phrases ("Net Income", "Cash Flow").
    for match in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})\b", text):
        phrase = match.group(1)
        if not _phrase_is_noisy(phrase):
            candidates.append(phrase)

    # Most-frequent content word as last resort.
    if not candidates:
        counter: Counter[str] = Counter()
        for tok in _NOUN_CANDIDATE_RE.findall(text):
            lowered = tok.lower()
            if lowered in _STOPWORDS or lowered in _GENERIC_TERMS:
                continue
            if len(lowered) < 4:
                continue
            counter[lowered] += 1
        for word, count in counter.most_common(limit):
            if count >= 2:
                candidates.append(word)

    # Cap and dedupe (preserve order).
    seen: set[str] = set()
    deduped: list[str] = []
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
        if len(deduped) >= limit:
            break
    return deduped


def _phrase_is_noisy(phrase: str) -> bool:
    """Reject fallback phrases that are obviously stopword stacks or too long."""
    if len(phrase) > 50:
        return True
    tokens = phrase.lower().split()
    if not tokens:
        return True
    return all(tok in _STOPWORDS or tok in _GENERIC_TERMS for tok in tokens)


def _frequency_filter(entities: list[Entity]) -> list[Entity]:
    """Drop low-signal entities that appeared only once and aren't strong.

    Keep an entity if any of:
      * its canonical text appears in ≥ ``_MIN_GLOBAL_FREQUENCY`` chunks, OR
      * its confidence is ≥ ``_HIGH_CONFIDENCE_FLOOR``, OR
      * it is a domain-whitelisted concept.
    """
    global_count: Counter[str] = Counter(e.text.lower() for e in entities)
    survivors: list[Entity] = []
    for entity in entities:
        key = entity.text.lower()
        if global_count[key] >= _MIN_GLOBAL_FREQUENCY:
            survivors.append(entity)
            continue
        if entity.confidence >= _HIGH_CONFIDENCE_FLOOR:
            survivors.append(entity)
            continue
        if key in _DOMAIN_KEYWORDS or any(tok in _DOMAIN_KEYWORDS for tok in key.split()):
            survivors.append(entity)
            continue
        # else: drop one-off noise
    return survivors


def should_trigger_fallback(confidence: float, threshold: float) -> bool:
    """Adaptive fallback gate for potential LLM enrichment."""
    return confidence < threshold
