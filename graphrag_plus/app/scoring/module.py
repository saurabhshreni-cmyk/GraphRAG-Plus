"""Uncertainty-aware scoring module."""

from __future__ import annotations

from typing import Any

from graphrag_plus.app.utils.logging_utils import get_logger, log_event
from graphrag_plus.app.utils.math_utils import min_max_normalize


class ScoringModule:
    """Combine retrieval signals into one score."""

    def __init__(self, weights: dict[str, float]):
        self.logger = get_logger(self.__class__.__name__)
        self.weights = weights

    def score_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize and score candidate rows."""
        if not candidates:
            return []
        semantic = min_max_normalize(c["semantic_score"] for c in candidates)
        graph = min_max_normalize(c["graph_score"] for c in candidates)
        confidence = min_max_normalize(c["confidence_score"] for c in candidates)
        trust = min_max_normalize(c["trust_score"] for c in candidates)
        uncertainty = min_max_normalize(c["uncertainty_penalty"] for c in candidates)

        scored: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            final_score = (
                self.weights["w1"] * semantic[idx]
                + self.weights["w2"] * graph[idx]
                + self.weights["w3"] * confidence[idx]
                + self.weights["w4"] * trust[idx]
                - self.weights["w5"] * uncertainty[idx]
            )
            enriched = dict(candidate)
            # Preserve the pre-normalization confidence so downstream callers
            # (calibration, failure classifier) don't see every single-candidate
            # query collapsed to 0.5 by min-max normalization.
            raw_confidence = float(candidate.get("confidence_score", confidence[idx]))
            enriched.update(
                {
                    "semantic_score": semantic[idx],
                    "graph_score": graph[idx],
                    "confidence_score": confidence[idx],
                    "raw_confidence_score": raw_confidence,
                    "trust_score": trust[idx],
                    "uncertainty_penalty": uncertainty[idx],
                    "final_score": final_score,
                }
            )
            scored.append(enriched)
            log_event(
                self.logger,
                "scoring_breakdown",
                {
                    "id": enriched.get("id"),
                    "semantic": semantic[idx],
                    "graph": graph[idx],
                    "confidence": confidence[idx],
                    "trust": trust[idx],
                    "uncertainty_penalty": uncertainty[idx],
                    "final_score": final_score,
                },
            )
        scored.sort(key=lambda item: item["final_score"], reverse=True)
        return scored
