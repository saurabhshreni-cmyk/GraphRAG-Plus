"""Math helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable


def min_max_normalize(values: Iterable[float]) -> list[float]:
    """Normalize values to [0, 1].

    When all input values are (near-)equal — including the single-element
    case — the normal min-max formula divides by zero. The previous
    implementation returned ``0.5`` for every value, which collapsed the
    confidence of a strong single-candidate retrieval to a misleading
    ``LOW_CONFIDENCE`` reading.

    Instead, when the inputs are degenerate we clamp each value to ``[0, 1]``
    and return it as-is. That preserves absolute scale: a single candidate
    with ``cosine=0.7`` keeps its high score, and a corpus where every
    chunk legitimately has the same low relevance keeps its low score
    rather than getting promoted to 0.5.
    """
    values_list = [float(v) for v in values]
    if not values_list:
        return []
    low = min(values_list)
    high = max(values_list)
    if math.isclose(low, high):
        return [max(0.0, min(1.0, value)) for value in values_list]
    return [(value - low) / (high - low) for value in values_list]


def safe_entropy(probability: float) -> float:
    """Binary entropy for uncertainty penalty."""
    p = min(max(probability, 1e-8), 1 - 1e-8)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
