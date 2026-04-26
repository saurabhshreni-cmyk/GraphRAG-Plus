"""Failure mode tests."""

from graphrag_plus.app.failure.handler import FailureModeHandler


def test_failure_mode_no_evidence() -> None:
    handler = FailureModeHandler()
    payload = handler.classify(
        has_evidence=False,
        confidence=0.0,
        uncertainty=1.0,
        has_conflict=False,
        llm_failed=False,
        confidence_threshold=0.7,
        high_uncertainty_threshold=0.55,
    )
    assert payload["failure_type"] == "NO_EVIDENCE"
