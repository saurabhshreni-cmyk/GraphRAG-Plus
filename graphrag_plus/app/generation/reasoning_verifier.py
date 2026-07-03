"""DeepSeek R1 reasoning verifier — final answer quality gate.

Runs ONLY at the final answer stage: after retrieval and after the draft
answer is generated, a reasoning model (``deepseek-r1:8b``) re-reads the
evidence and either confirms the draft, corrects it, or abstains.

Flow::

    qwen3.5:4b            → extraction (never this module)
    hybrid retrieval      → top-K evidence chunks
    qwen3.5:4b            → draft answer
    deepseek-r1:8b        → verification (this module)
    final verified answer → user

Failure policy: verification is strictly additive. On timeout, missing
model, or any error, the original draft is returned with
``verified=False`` — the verifier can never make the system worse.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from graphrag_plus.app.utils.logging_utils import get_logger

load_dotenv()

logger = get_logger(__name__)

_DEFAULT_MODEL = os.environ.get("REASONING_MODEL", "deepseek-r1:8b")
_DEFAULT_TIMEOUT_S = float(os.environ.get("GRAPHRAG_VERIFIER_TIMEOUT_S", "60"))
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

_ANSWER_MARKER = "FINAL VERIFIED ANSWER:"
_ABSTAIN_TEXT = "I cannot answer based on the provided evidence."

_VERIFY_PROMPT = """You are a rigorous fact-checker and reasoning verifier for a GraphRAG system.
You will be given a question, a draft answer generated from retrieved evidence, and the source evidence chunks.

Your job:
1. Read the evidence carefully
2. Check if the draft answer is actually supported by the evidence
3. Check if the answer is complete, accurate, and not hallucinating
4. If the draft answer is correct and complete: return it exactly as is
5. If the draft answer has errors or missing information: correct it using ONLY the evidence
6. If the evidence cannot support any answer: say exactly "I cannot answer based on the provided evidence."

Think step by step before giving your final answer.

QUESTION:
{question}

DRAFT ANSWER:
{draft_answer}

EVIDENCE:
{evidence_text}

After your reasoning, on a new line write exactly:
FINAL VERIFIED ANSWER: [your answer here]"""

# R1-style models wrap hidden reasoning in <think> tags when the API doesn't
# separate it; strip before parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class VerificationResult(BaseModel):
    """Outcome of one reasoning-verification pass."""

    final_answer: str
    verified: bool = Field(description="True if the reasoning model ran successfully")
    changed: bool = Field(default=False, description="True if the draft answer was modified")
    reasoning_summary: str = Field(default="", description="Brief summary of what was checked")
    time_taken_s: float = 0.0


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".")


class ReasoningVerifier:
    """Verify a draft answer against evidence with a reasoning LLM."""

    def __init__(self, model: str = _DEFAULT_MODEL, timeout_s: float = _DEFAULT_TIMEOUT_S):
        self.model = model
        self.timeout_s = timeout_s
        self._client: Any = None
        self._available: bool | None = None

    # ------------------------------------------------------------- availability
    def _get_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=_OLLAMA_BASE_URL, timeout=self.timeout_s)
        return self._client

    def available(self) -> bool:
        """True iff the reasoning model is present in the local Ollama. Cached."""
        if self._available is not None:
            return self._available
        try:
            listed = self._get_client().list()
            names = {m.get("model", "") for m in listed.get("models", [])} if isinstance(listed, dict) else {
                getattr(m, "model", "") for m in getattr(listed, "models", [])
            }
            self._available = any(name.startswith(self.model) for name in names)
        except Exception as exc:
            logger.warning("verifier.availability_check_failed error=%s", str(exc)[:150])
            self._available = False
        if not self._available:
            logger.warning("verifier.model_unavailable model=%s — verification disabled", self.model)
        return self._available

    # ------------------------------------------------------------------- verify
    def verify(
        self, question: str, draft_answer: str, evidence_chunks: list[dict]
    ) -> VerificationResult:
        """Check ``draft_answer`` against evidence; never raises.

        Returns the draft unchanged (``verified=False``) on any failure,
        timeout, or when the model is not installed.
        """
        start = time.perf_counter()

        def _fallback(reason: str) -> VerificationResult:
            return VerificationResult(
                final_answer=draft_answer,
                verified=False,
                changed=False,
                reasoning_summary=reason,
                time_taken_s=round(time.perf_counter() - start, 2),
            )

        if not draft_answer.strip() or not evidence_chunks:
            return _fallback("skipped: empty draft or no evidence")
        if not self.available():
            return _fallback(f"skipped: {self.model} not available in Ollama")

        evidence_text = "\n\n".join(
            f"[{idx + 1}] {str(chunk.get('full_text') or chunk.get('snippet') or '')[:1200]}"
            for idx, chunk in enumerate(evidence_chunks[:5])
        )
        prompt = _VERIFY_PROMPT.format(
            question=question.strip(),
            draft_answer=draft_answer.strip(),
            evidence_text=evidence_text,
        )

        try:
            response = self._get_client().generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.0, "num_predict": 2048},
            )
            if isinstance(response, dict):
                raw = str(response.get("response", ""))
                thinking = str(response.get("thinking", "") or "")
            else:
                raw = str(getattr(response, "response", ""))
                thinking = str(getattr(response, "thinking", "") or "")
        except Exception as exc:
            logger.warning("verifier.call_failed model=%s error=%s", self.model, str(exc)[:200])
            return _fallback(f"error: {type(exc).__name__} (timeout={self.timeout_s}s)")

        final_answer, summary = self._parse(raw)
        if not final_answer:
            return _fallback("parse_failed: no FINAL VERIFIED ANSWER marker in response")
        if not summary and thinking:
            # Newer Ollama separates R1's reasoning into a ``thinking`` field;
            # summarize it: first two + last thought lines.
            lines = [ln.strip() for ln in thinking.splitlines() if ln.strip()]
            summary = (" ".join(lines[:2]) + (" … " + lines[-1] if len(lines) > 2 else ""))[:600]

        changed = _normalize_for_compare(final_answer) != _normalize_for_compare(draft_answer)
        elapsed = round(time.perf_counter() - start, 2)
        logger.info(
            "verifier.done model=%s changed=%s elapsed=%.1fs", self.model, changed, elapsed
        )
        return VerificationResult(
            final_answer=final_answer,
            verified=True,
            changed=changed,
            reasoning_summary=summary,
            time_taken_s=elapsed,
        )

    # -------------------------------------------------------------------- parse
    @staticmethod
    def _parse(raw: str) -> tuple[str, str]:
        """Extract (final_answer, reasoning_summary) from the model output."""
        visible = _THINK_RE.sub("", raw).strip()
        marker_at = visible.rfind(_ANSWER_MARKER)
        if marker_at == -1:
            return "", ""
        final = visible[marker_at + len(_ANSWER_MARKER) :].strip()
        # Model sometimes wraps the answer in brackets per the prompt example.
        final = re.sub(r"^\[|\]$", "", final).strip()
        reasoning = visible[:marker_at].strip()
        # Compact multi-line reasoning into a short summary (first + last point).
        lines = [ln.strip() for ln in reasoning.splitlines() if ln.strip()]
        if len(lines) > 4:
            summary = " ".join(lines[:2]) + " … " + lines[-1]
        else:
            summary = " ".join(lines)
        return final, summary[:600]


__all__ = ["ReasoningVerifier", "VerificationResult", "_ABSTAIN_TEXT"]
