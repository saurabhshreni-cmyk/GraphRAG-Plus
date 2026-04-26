"""Agentic analyst response augmentation."""

from __future__ import annotations


class AnalystEngine:
    """Build analyst mode explanation artifacts."""

    def build(
        self,
        question: str,
        evidence_paths: list[list[str]],
        contradictions: list[dict[str, str]],
        confidence: float,
    ) -> dict[str, list[str]]:
        """Generate analyst output fields."""
        reasoning_steps = [
            "Parsed question and decomposed into subqueries.",
            "Retrieved hybrid evidence from vector, keyword, and graph signals.",
            "Applied uncertainty-aware reranking with trust and confidence.",
        ]
        if contradictions:
            reasoning_steps.append("Detected contradictory claims and ranked source reliability.")
        if confidence < 0.7:
            reasoning_steps.append("Marked uncertainty as elevated and suggested follow-up queries.")

        follow_ups = [
            f"What additional sources validate: {question[:80]}?",
            "Can you provide a time range to narrow temporal ambiguity?",
            "Would you like source-by-source contradiction adjudication?",
        ]
        return {
            "reasoning_steps": reasoning_steps,
            "follow_up_questions": follow_ups,
            "evidence_paths": [" -> ".join(path) for path in evidence_paths],
            "contradictions": [item.get("explanation", "") for item in contradictions],
        }
