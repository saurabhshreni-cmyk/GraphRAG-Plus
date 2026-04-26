"""Query decomposition and routing."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryPlan:
    """Simple query plan."""

    subqueries: list[str]
    requires_temporal_reasoning: bool
    graph_dominant: bool


TEMPORAL_TERMS = {"before", "after", "during", "latest", "earliest", "between", "when"}


def plan_query(question: str) -> QueryPlan:
    """Create query plan from question text."""
    cleaned = question.strip()
    subqueries = [part.strip() for part in re.split(r"\band\b|\?", cleaned) if part.strip()]
    lower = cleaned.lower()
    requires_temporal = any(term in lower for term in TEMPORAL_TERMS) or bool(re.search(r"\b\d{4}\b", lower))
    graph_dominant = "relationship" in lower or "connected" in lower or "path" in lower
    return QueryPlan(
        subqueries=subqueries or [cleaned],
        requires_temporal_reasoning=requires_temporal,
        graph_dominant=graph_dominant,
    )
