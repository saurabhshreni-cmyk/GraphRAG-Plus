"""Math helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable


def min_max_normalize(values: Iterable[float]) -> list[float]:
    """Normalize values to [0, 1]."""
    values_list = list(values)
    if not values_list:
        return []
    low = min(values_list)
    high = max(values_list)
    if math.isclose(low, high):
        return [0.5 for _ in values_list]
    return [(value - low) / (high - low) for value in values_list]


def safe_entropy(probability: float) -> float:
    """Binary entropy for uncertainty penalty."""
    p = min(max(probability, 1e-8), 1 - 1e-8)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
