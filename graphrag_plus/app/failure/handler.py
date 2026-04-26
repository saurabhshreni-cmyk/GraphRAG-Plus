"""Failure mode handler."""

from __future__ import annotations

from graphrag_plus.app.schemas.models import FailureType


class FailureModeHandler:
    """Evaluate and annotate failure modes."""

    def classify(
        self,
        *,
        has_evidence: bool,
        confidence: float,
        uncertainty: float,
        has_conflict: bool,
        llm_failed: bool,
        confidence_threshold: float,
        high_uncertainty_threshold: float,
    ) -> dict[str, str | None]:
        """Return failure classification and mitigation strategy."""
        if llm_failed:
            return {
                "failure_type": FailureType.LLM_FAILURE.value,
                "explanation": "LLM call failed; fallback deterministic generation used.",
                "mitigation": "deterministic_fallback",
            }
        if not has_evidence:
            return {
                "failure_type": FailureType.NO_EVIDENCE.value,
                "explanation": "No relevant evidence was retrieved.",
                "mitigation": "abstain_and_request_more_context",
            }
        if has_conflict:
            return {
                "failure_type": FailureType.CONFLICTING_EVIDENCE.value,
                "explanation": "Evidence sources conflict; ranked by trust and consistency.",
                "mitigation": "show_both_sides_with_resolution",
            }
        if uncertainty > high_uncertainty_threshold:
            return {
                "failure_type": FailureType.HIGH_UNCERTAINTY.value,
                "explanation": "The prediction uncertainty is high.",
                "mitigation": "flag_as_unreliable",
            }
        if confidence < confidence_threshold:
            return {
                "failure_type": FailureType.LOW_CONFIDENCE.value,
                "explanation": "Evidence was found but confidence is low.",
                "mitigation": "partial_answer_with_warning",
            }
        return {"failure_type": None, "explanation": None, "mitigation": None}
