"""Concrete :class:`LLMClient` implementations.

Currently ships:

* :class:`AnthropicClient` — calls the Anthropic Messages API. Activated when
  ``ANTHROPIC_API_KEY`` is set in the environment.
* :class:`EchoClient`      — deterministic, no-network fallback used when no
  real provider is configured. Returns a slightly cleaned-up version of the
  best snippet so the answer pipeline still benefits from "LLM enabled".

The pipeline picks a client at startup via :func:`build_default_llm_client`.
That function returns ``None`` when ``llm_enabled`` is ``False`` — which keeps
the cold-path identical to today's tests.

Design rules (preserved from the parent module's docstring):

* The LLM only sees retrieved snippets — no external context.
* ``NO_EVIDENCE`` queries never reach the client.
* On any client error, callers fall back to extractive answers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from graphrag_plus.app.generation.generator import LLMClient
from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL_DEFAULT = "claude-3-5-haiku-latest"
ANTHROPIC_VERSION = "2023-06-01"

OLLAMA_URL_DEFAULT = "http://localhost:11434/api/generate"
# Default to qwen3.5:4b — fits in less RAM than gemma4:e2b and reloads
# reliably on memory-constrained hosts. Override with OLLAMA_MODEL.
OLLAMA_MODEL_DEFAULT = "qwen3.5:4b"

_SYSTEM_PROMPT = (
    "You are GraphRAG++'s answer composer. Answer the user's question using "
    "ONLY the numbered context snippets provided. If the context does not "
    "contain enough information, say so plainly. Do not add facts that are "
    "not in the snippets. Keep the answer to 1-3 sentences."
)

_OLLAMA_PROMPT_TEMPLATE = (
    "You are answering strictly based on the following evidence:\n\n"
    "{context}\n\n"
    "Question:\n{question}\n\n"
    "Instructions:\n"
    "- Answer clearly and concisely\n"
    "- Use ONLY the provided evidence\n"
    "- Do NOT add external knowledge\n"
    "- If the answer is not in the evidence, say you cannot answer\n\n"
    "Answer:"
)


class EchoClient:
    """Deterministic fallback used when no real LLM provider is configured.

    Trims the highest-relevance snippet to a single tight statement. Marked
    "(local synth)" so callers / logs can tell it apart from a real LLM.
    """

    def complete(self, question: str, context: str) -> str:
        # Take the first context line — it's already the highest-ranked
        # snippet from the retrieval stack.
        first_line = next((ln for ln in context.splitlines() if ln.strip()), "")
        cleaned = first_line.split(") ", 1)[-1] if ") " in first_line else first_line
        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = "Evidence was found but the synthesizer received no usable context."
        return f"{cleaned} (local synth)"


class AnthropicClient:
    """Anthropic Messages API client.

    Uses ``urllib`` so we don't add a runtime dependency on the official SDK.
    Reads the API key from the constructor or the ``ANTHROPIC_API_KEY``
    environment variable. ``timeout_s`` caps the request so a slow API never
    holds up the FastAPI event loop indefinitely.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = ANTHROPIC_MODEL_DEFAULT,
        max_tokens: int = 400,
        timeout_s: float = 20.0,
    ):
        resolved = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved:
            raise ValueError("AnthropicClient requires an api_key argument or " "ANTHROPIC_API_KEY env var.")
        self.api_key: str = resolved
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def complete(self, question: str, context: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Context (numbered snippets):\n{context}\n\n"
                        "Answer using only the snippets above."
                    ),
                }
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=data,
            method="POST",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        try:
            # request URL is hard-coded to api.anthropic.com (constant above).
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning("anthropic.request_failed error=%s", exc)
            raise
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("anthropic.decode_failed error=%s body=%r", exc, body[:200])
            raise
        # Messages API returns content as a list of {type, text} blocks.
        blocks = parsed.get("content", [])
        text_parts = [str(blk.get("text", "")) for blk in blocks if blk.get("type") == "text"]
        return " ".join(part.strip() for part in text_parts if part.strip())


class LocalLLMClient:
    """Ollama-backed local LLM client.

    Talks to a locally running Ollama daemon (default
    ``http://localhost:11434``) using the ``/api/generate`` endpoint with
    streaming disabled. Designed for the gemma4:e2b model but works with
    any model the daemon has loaded.

    Trust-aware design rules from the parent module apply: the prompt only
    embeds retrieved snippets, the caller never invokes us when there is no
    evidence, and any HTTP/timeout error raises so the pipeline can fall
    back to extractive output.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        model: str = OLLAMA_MODEL_DEFAULT,
        timeout_s: float = 60.0,
        max_context_chars: int = 4000,
    ):
        self.url = url or os.environ.get("OLLAMA_URL", OLLAMA_URL_DEFAULT)
        self.model = os.environ.get("OLLAMA_MODEL", model)
        self.timeout_s = timeout_s
        # Cap the context block we hand the model so the prompt stays inside
        # gemma4:e2b's working window and the request returns within timeout.
        self.max_context_chars = max_context_chars

    def _build_prompt(self, question: str, context: str) -> str:
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars].rstrip() + "..."
        return _OLLAMA_PROMPT_TEMPLATE.format(question=question.strip(), context=context.strip())

    def complete(self, question: str, context: str) -> str:
        prompt = self._build_prompt(question, context)
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Disable Ollama's thinking mode (Qwen3, DeepSeek-R1, etc. emit
            # long internal reasoning by default which can take 100+s for a
            # one-line answer). We want the visible response only. Models that
            # don't support thinking mode ignore this flag.
            "think": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            # URL is operator-controlled (env var) and defaults to localhost.
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # Surface Ollama's own error body (e.g. OOM, model not pulled) so
            # operators can debug without re-running with a sniffer attached.
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # pragma: no cover - defensive
                detail = ""
            logger.warning(
                "ollama.request_failed url=%s status=%s detail=%s",
                self.url,
                exc.code,
                detail,
            )
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning("ollama.request_failed url=%s error=%s", self.url, exc)
            raise
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("ollama.decode_failed error=%s body=%r", exc, body[:200])
            raise
        # Non-streaming /api/generate returns {"response": "...", "done": true, ...}
        return str(parsed.get("response", "")).strip()


def _ollama_available(url: str = OLLAMA_URL_DEFAULT, timeout_s: float = 1.0) -> bool:
    """Quick liveness probe so we don't pick LocalLLMClient when Ollama is
    not actually running on the host."""
    tags_url = url.rsplit("/api/", 1)[0] + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def build_default_llm_client(*, llm_enabled: bool) -> LLMClient | None:
    """Pick a sensible default client.

    Order of preference:

    1. ``llm_enabled=False`` -> ``None`` (pure-extractive cold path).
    2. ``OLLAMA_URL`` set or a daemon running on the default URL -> :class:`LocalLLMClient`.
    3. ``ANTHROPIC_API_KEY`` set -> :class:`AnthropicClient`.
    4. :class:`EchoClient` so the LLM-enabled path still runs in dev / CI
       without external dependencies.
    """
    if not llm_enabled:
        return None
    explicit_ollama = os.environ.get("OLLAMA_URL")
    if explicit_ollama or _ollama_available():
        try:
            return LocalLLMClient()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("llm.local_init_failed error=%s", exc)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicClient()
        except ValueError as exc:
            logger.warning("llm.anthropic_init_failed error=%s", exc)
    return EchoClient()


__all__ = [
    "AnthropicClient",
    "EchoClient",
    "LLMClient",
    "LocalLLMClient",
    "build_default_llm_client",
]
