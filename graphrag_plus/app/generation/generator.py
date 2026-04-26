"""Grounded answer generation."""

from __future__ import annotations


class AnswerGenerator:
    """Deterministic generator with optional LLM fallback hook."""

    def __init__(self, llm_enabled: bool):
        self.llm_enabled = llm_enabled

    def generate(
        self,
        question: str,
        evidence: list[dict[str, object]],
        confidence: float,
        answer_threshold: float,
    ) -> tuple[str, bool, bool]:
        """Generate answer text. Returns answer, used_llm, llm_failed."""
        if not evidence:
            return "I cannot answer reliably because no evidence was found.", False, False
        top_snippets = [str(item.get("snippet", "")) for item in evidence[:3]]

        if confidence >= answer_threshold or not self.llm_enabled:
            answer = " ".join(snippet[:180] for snippet in top_snippets if snippet).strip()
            return answer or "Evidence was found but insufficient for a complete answer.", False, False

        try:
            # Placeholder LLM adapter path: intentionally deterministic until provider is configured.
            llm_answer = f"LLM-style synthesis: {top_snippets[0][:220]}"
            return llm_answer, True, False
        except Exception:
            fallback = " ".join(snippet[:160] for snippet in top_snippets).strip()
            return fallback or "LLM failed and fallback evidence was weak.", False, True
