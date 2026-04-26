"""Scoring module tests."""

from graphrag_plus.app.scoring.module import ScoringModule


def test_scoring_module_normalizes_and_sorts() -> None:
    scoring = ScoringModule({"w1": 0.2, "w2": 0.2, "w3": 0.2, "w4": 0.2, "w5": 0.2})
    candidates = [
        {
            "id": "a",
            "semantic_score": 0.9,
            "graph_score": 0.8,
            "confidence_score": 0.7,
            "trust_score": 0.6,
            "uncertainty_penalty": 0.1,
        },
        {
            "id": "b",
            "semantic_score": 0.3,
            "graph_score": 0.2,
            "confidence_score": 0.2,
            "trust_score": 0.4,
            "uncertainty_penalty": 0.8,
        },
    ]
    scored = scoring.score_candidates(candidates)
    assert len(scored) == 2
    assert scored[0]["id"] == "a"
    assert scored[0]["final_score"] > scored[1]["final_score"]
