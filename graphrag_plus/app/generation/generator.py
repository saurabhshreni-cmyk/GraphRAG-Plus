"""Grounded answer generation.

Two strategies, both grounded in retrieved evidence:

* **Extractive (default)** — score each sentence in the top-ranked chunk(s)
  by how many query terms it covers, and return the best 1-2 sentences.
  Deterministic, fast, never hallucinates.

* **LLM (opt-in via ``llm_enabled``)** — pass the same evidence into a
  pluggable :class:`LLMClient` to produce a concise summary. The interface
  is intentionally minimal (one ``complete`` method) so any
  Anthropic / OpenAI / local backend can be wired in without changing
  pipeline code. If the client errors, we fall back to extractive.

Trust-aware behaviour:

* When ``evidence`` is empty we never call the LLM; the failure handler
  upstream classifies the query as ``NO_EVIDENCE``.
* The LLM only sees the retrieved snippets — there is no out-of-band
  context that could leak ungrounded facts.
"""

from __future__ import annotations

import re
from typing import Protocol

# Sentinel string the LLM is instructed to emit when evidence doesn't cover
# the question. Detecting it lets us fall back to extractive output instead
# of returning a verbatim refusal that the user can't act on.
LLM_ABSTAIN_TOKEN = "I cannot answer based on the provided context."

# Stopwords used for scoring sentences against the query. Kept in sync with
# the retrieval service philosophy: we want content overlap, not boilerplate
# overlap.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "if",
        "then",
        "else",
        "than",
        "what",
        "which",
        "who",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "tell",
        "about",
        "explain",
        "show",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class LLMClient(Protocol):
    """Minimal completion interface for any backend.

    A concrete implementation receives the user question plus retrieval
    context and returns a single answer string. Implementations are
    responsible for their own timeouts / retries.
    """

    def complete(self, question: str, context: str) -> str: ...


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text or "") if tok.lower() not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    # Normalize whitespace and split on .?! followed by whitespace + capital.
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    # Keep the original sentence-ending punctuation by re-joining single
    # trailing characters that the regex may have stripped.
    return [p.strip() for p in parts if p.strip()]


def _rank_sentences(question: str, sentences: list[str]) -> list[tuple[float, str]]:
    """Score each sentence by query-term recall (Jaccard-ish)."""
    q_tokens = _tokens(question)
    if not q_tokens:
        return [(0.0, s) for s in sentences]
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        s_tokens = _tokens(sentence)
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens)
        if overlap == 0:
            continue
        # Recall (overlap / |q|) weighted by precision (overlap / |s|),
        # which prefers tight, on-topic sentences over rambling ones.
        recall = overlap / len(q_tokens)
        precision = overlap / len(s_tokens)
        scored.append((recall + 0.3 * precision, sentence))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


class AnswerGenerator:
    """Deterministic generator with optional LLM-backed synthesis."""

    def __init__(self, llm_enabled: bool, llm_client: LLMClient | None = None):
        self.llm_enabled = llm_enabled
        self.llm_client = llm_client

    # --------------------------------------------------------- extractive path
    @staticmethod
    def _extractive_answer(question: str, evidence: list[dict[str, object]]) -> str:
        # Use the single best chunk first; fall back to the next chunk if it
        # has no question-relevant sentences. Caps at 2 sentences to keep the
        # answer focused.
        for item in evidence[:3]:
            snippet = str(item.get("snippet", "") or "")
            sentences = _split_sentences(snippet)
            ranked = _rank_sentences(question, sentences)
            if ranked:
                top_two = [sentence for _, sentence in ranked[:2]]
                return " ".join(top_two)
        # Nothing matched the query — return the highest-final-score snippet
        # truncated, so the user at least sees the most relevant retrieved
        # text.
        if evidence:
            return str(evidence[0].get("snippet", ""))[:300].strip()
        return "Evidence was found but insufficient for a complete answer."

    # ---------------------------------------------------------------- LLM path
    @staticmethod
    def _build_context(evidence: list[dict[str, object]]) -> str:
        """Top-3 strongest chunks, deduped sentence-by-sentence.

        Sorted-by-final-score happens upstream (ScoringModule), so we just
        take the prefix here. We split into sentences so duplicate sentences
        across overlapping chunks (common when ingestion produced
        near-identical paragraphs) don't show up twice in the prompt.
        """
        seen: set[str] = set()
        rendered: list[str] = []
        for idx, item in enumerate(evidence[:3]):
            source = str(item.get("source_id", "?"))
            snippet = str(item.get("snippet", "") or "").strip()
            if not snippet:
                continue
            kept_sentences: list[str] = []
            for sentence in _split_sentences(snippet):
                key = sentence.lower().rstrip(".!?").strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                kept_sentences.append(sentence)
            if kept_sentences:
                rendered.append(f"[{idx + 1}] ({source}) {' '.join(kept_sentences)}")
        return "\n".join(rendered)

    def _llm_answer(self, question: str, evidence: list[dict[str, object]]) -> str:
        client = self.llm_client
        if client is None:
            # No client wired — fall through to deterministic placeholder so
            # behaviour stays predictable in test environments.
            best = str(evidence[0].get("snippet", "") or "")[:240]
            return f"LLM-style synthesis: {best}".strip()
        context = self._build_context(evidence)
        return client.complete(question, context).strip()

    # --------------------------------------------------------- quality filter
    @staticmethod
    def _llm_answer_passes_quality(question: str, answer: str) -> bool:
        """Reject LLM output that doesn't share key terms with the question.

        Without this gate a model that drifts off-topic ("Here is a poem
        about graphs...") would be returned to the user verbatim. We require
        either at least one shared content token, OR — for very short
        questions — that the answer be substantive (>= 6 content tokens).
        """
        q_tokens = _tokens(question)
        a_tokens = _tokens(answer)
        if not q_tokens:
            return bool(a_tokens)
        if q_tokens & a_tokens:
            return True
        # If the question has only one content token (e.g. "NetworkX?") and
        # the answer is substantive, allow it through. The retrieval gate
        # already enforced that the chunks are on-topic, so the answer is
        # almost certainly grounded even if it paraphrases the term.
        return len(q_tokens) <= 1 and len(a_tokens) >= 6

    # ----------------------------------------------------------------- public
    def generate(  # noqa: PLR0911 -- early returns mirror the documented gate flow
        self,
        question: str,
        evidence: list[dict[str, object]],
        confidence: float,
        answer_threshold: float,
    ) -> tuple[str, bool, bool]:
        """Generate answer text. Returns (answer, used_llm, llm_failed).

        * ``used_llm`` — True iff the LLM client produced the final answer.
        * ``llm_failed`` — True iff the LLM was attempted and raised; in
          that case we transparently fall back to extractive output.
        """
        if not evidence:
            # NO_EVIDENCE handling stays in the failure classifier; we just
            # emit a stable abstain string and never call the LLM.
            return (
                "I cannot answer reliably because no evidence was found.",
                False,
                False,
            )

        extractive = self._extractive_answer(question, evidence)

        # Flow (matches the architecture diagram):
        #     evidence empty             -> abstain, never call LLM (handled above)
        #     llm_enabled=False          -> extractive
        #     llm_enabled=True
        #         confidence >= threshold -> extractive (already strong, save tokens)
        #         confidence <  threshold -> LLM, with extractive fallback on error
        if not self.llm_enabled:
            return extractive, False, False
        if confidence >= answer_threshold:
            return extractive, False, False
        try:
            llm_answer = self._llm_answer(question, evidence)
        except Exception:
            # LLM raised (timeout / connection / decode) -> extractive fallback,
            # surface llm_failed=True in the flags for observability.
            return extractive, False, True

        if not llm_answer:
            # Empty response -> soft failure, fall back to extractive.
            return extractive, False, True

        # The LLM was instructed to emit a fixed abstain string when the
        # context doesn't cover the question. We have evidence (this branch
        # only runs when len(evidence) > 0), so an abstain means the model
        # disagreed with retrieval. Trust retrieval and fall back to the
        # extractive sentence-rank answer.
        if LLM_ABSTAIN_TOKEN.lower().rstrip(".") in llm_answer.lower():
            return extractive, False, True

        # Quality filter: reject completions that drifted off-topic. The
        # retrieval gates already ensure the chunks share query terms with
        # the question, so the answer should too.
        if not self._llm_answer_passes_quality(question, llm_answer):
            return extractive, False, True

        return llm_answer, True, False
