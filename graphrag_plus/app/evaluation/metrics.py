"""Evaluation metrics."""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Compute precision@k."""
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = len(set(top).intersection(relevant))
    return hits / len(top)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Compute recall@k."""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]).intersection(relevant))
    return hits / len(relevant)


def hallucination_rate(answered: int, unsupported: int) -> float:
    """Unsupported answer rate."""
    if answered == 0:
        return 0.0
    return unsupported / answered


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average metric rows."""
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}
