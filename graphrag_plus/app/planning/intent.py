"""Query intent detection.

Lightweight rule-based classifier — no ML dependency. Routes the query
into one of five intents that downstream modules (retrieval, generator)
adapt their behaviour for:

* ``DEFINITION`` — "what is X", "define X"
* ``LIST``       — "types of X", "examples of X", "kinds / methods / forms / categories"
* ``COMPARISON`` — "X vs Y", "difference between X and Y", "compare X and Y"
* ``EXPLANATION``— "how / why / explain"
* ``FACTUAL``    — fallback: specific fact lookup

The classifier is deliberately ordered most-specific → least-specific so
"difference between" beats the standalone "between", and a question
phrased as "how do types of X differ" still routes to COMPARISON.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class QueryIntent(StrEnum):
    """Five-way intent taxonomy used to adapt retrieval + generation."""

    DEFINITION = "definition"
    LIST = "list"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    FACTUAL = "factual"


# Patterns are evaluated in declaration order. Comparison must precede LIST
# (else "differences between types of X" would route to LIST), and DEFINITION
# must precede EXPLANATION (else "what is X and how does it work" routes to
# explanation when definition is more useful).

_COMPARISON_PATTERNS = (
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bdifference[s]?\s+between\b", re.IGNORECASE),
    re.compile(r"\bdiffer(?:s|ence)?\s+(?:from|between)\b", re.IGNORECASE),
    re.compile(r"\bcompare[d]?\s+(?:to|with|and)\b", re.IGNORECASE),
    re.compile(r"\bcomparison\s+(?:of|between)\b", re.IGNORECASE),
    re.compile(r"\bsimilar\s+(?:to|and)\b", re.IGNORECASE),
)

_LIST_PATTERNS = (
    re.compile(
        r"\b(?:types?|kinds?|examples?|categories|forms?|methods?|approaches|varieties|classes)\s+of\b",
        re.IGNORECASE,
    ),
    re.compile(r"\blist\s+(?:of|all|some|the)\b", re.IGNORECASE),
    re.compile(
        r"\bwhat\s+are\s+(?:the\s+)?(?:types?|kinds?|examples?|methods?|categories|forms?|varieties|classes)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bname\s+(?:some|all|a few)\b", re.IGNORECASE),
    re.compile(r"\benumerate\b", re.IGNORECASE),
    re.compile(r"^(?:types?|kinds?|methods?|examples?|categories|forms?|varieties)\s+", re.IGNORECASE),
)

_DEFINITION_PATTERNS = (
    re.compile(r"\bwhat\s+is\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+(?!the\s+(?:types?|kinds?|methods?|examples?|categories))", re.IGNORECASE),
    re.compile(r"\bdefine\b", re.IGNORECASE),
    re.compile(r"\bdefinition\s+of\b", re.IGNORECASE),
    re.compile(r"\bmeaning\s+of\b", re.IGNORECASE),
    re.compile(r"\bdefined\s+as\b", re.IGNORECASE),
    re.compile(r"^(?:what\s+is|whats?)\b", re.IGNORECASE),
)

_EXPLANATION_PATTERNS = (
    re.compile(r"\bhow\s+(?:does|do|is|are|can|can\s+i|to|the)\b", re.IGNORECASE),
    re.compile(r"\bwhy\s+(?:does|do|is|are|did|would|should)\b", re.IGNORECASE),
    re.compile(r"\bexplain\b", re.IGNORECASE),
    re.compile(r"\bdescribe\s+how\b", re.IGNORECASE),
    re.compile(r"\bhow\s+does\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class IntentSignal:
    """Result of intent detection — exposed so callers can log the rule that fired."""

    intent: QueryIntent
    matched_pattern: str | None  # Pattern source string that matched.


def detect_query_intent(query: str) -> QueryIntent:
    """Classify ``query`` into one of the five :class:`QueryIntent` values.

    Returns ``QueryIntent.FACTUAL`` when nothing else matches — that's the
    safe default because it preserves baseline retrieval behaviour.
    """
    return detect_intent_signal(query).intent


def detect_intent_signal(query: str) -> IntentSignal:
    """Same as :func:`detect_query_intent` but also returns the matched rule."""
    if not query or not query.strip():
        return IntentSignal(QueryIntent.FACTUAL, None)

    q = query.strip()
    # Comparison first — most specific multi-word patterns.
    for pattern in _COMPARISON_PATTERNS:
        if pattern.search(q):
            return IntentSignal(QueryIntent.COMPARISON, pattern.pattern)
    # List second — "types of X" etc.
    for pattern in _LIST_PATTERNS:
        if pattern.search(q):
            return IntentSignal(QueryIntent.LIST, pattern.pattern)
    # Definition before explanation so "what is X" wins over an embedded "how".
    for pattern in _DEFINITION_PATTERNS:
        if pattern.search(q):
            return IntentSignal(QueryIntent.DEFINITION, pattern.pattern)
    for pattern in _EXPLANATION_PATTERNS:
        if pattern.search(q):
            return IntentSignal(QueryIntent.EXPLANATION, pattern.pattern)
    return IntentSignal(QueryIntent.FACTUAL, None)


def adaptive_top_k(intent: QueryIntent, requested_top_k: int) -> int:
    """Adapt the retrieval ``top_k`` based on intent.

    DEFINITION wants 1-2 chunks (concise authoritative). LIST wants 5-8
    chunks because enumerations are often spread across paragraphs.
    EXPLANATION wants broader context (5). COMPARISON keeps the requested k
    but the retrieval layer additionally enforces both-term presence.
    """
    if intent == QueryIntent.DEFINITION:
        return max(2, min(requested_top_k, 2))
    if intent == QueryIntent.LIST:
        return max(5, min(8, requested_top_k * 2))
    if intent == QueryIntent.EXPLANATION:
        return max(5, requested_top_k)
    return requested_top_k


def comparison_terms(query: str) -> tuple[str, str] | None:
    """Pull the two compared terms out of a comparison-intent query.

    Handles ``"X vs Y"``, ``"X versus Y"``, ``"difference between X and Y"``,
    and ``"compare X to/with/and Y"``. Returns ``None`` if both terms can't
    be extracted, in which case the caller falls back to the original query.
    """
    q = query.strip().rstrip("?.")
    # "difference between X and Y"
    m = re.search(
        r"difference[s]?\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+)$",
        q,
        re.IGNORECASE,
    )
    if m:
        return _clean_term(m.group("a")), _clean_term(m.group("b"))
    # "compare X to|with|and Y"
    m = re.search(
        r"compare[d]?\s+(?P<a>.+?)\s+(?:to|with|and)\s+(?P<b>.+)$",
        q,
        re.IGNORECASE,
    )
    if m:
        return _clean_term(m.group("a")), _clean_term(m.group("b"))
    # "X vs|versus Y"
    m = re.search(r"(?P<a>.+?)\s+(?:vs\.?|versus)\s+(?P<b>.+)$", q, re.IGNORECASE)
    if m:
        return _clean_term(m.group("a")), _clean_term(m.group("b"))
    return None


def _clean_term(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,;:?!\"'")
