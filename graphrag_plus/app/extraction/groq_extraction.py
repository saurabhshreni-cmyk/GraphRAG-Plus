"""Groq backend for the LLM extraction stage.

The extractor was written against the Ollama SDK (``client.chat(...)``). On
the cloud deployment there is no Ollama, so this adapter exposes the same
``chat`` call over Groq's OpenAI-compatible API. Without it the cloud graph
only ever gets spaCy entities and generic ``co_occurs`` edges — never typed
relations like "acquired" or "founded".
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from graphrag_plus.app.utils.http_utils import json_headers

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_EXTRACTION_MODEL = "openai/gpt-oss-20b"
_TIMEOUT_S = 30.0
# gpt-oss counts its hidden reasoning against max_tokens; "low" effort keeps
# that small so the JSON is not truncated.
_MAX_TOKENS = 2048


class GroqExtractionClient:
    """Duck-types the Ollama client's ``chat`` method for the extractor."""

    def __init__(self, api_key: str, model: str | None = None, timeout_s: float = _TIMEOUT_S):
        if not api_key:
            raise ValueError("GroqExtractionClient requires an API key")
        self._api_key = api_key
        self.model = (
            model
            or os.environ.get("GROQ_EXTRACTION_MODEL")
            or os.environ.get("GROQ_MODEL")
            or DEFAULT_EXTRACTION_MODEL
        )
        self.timeout_s = timeout_s

    def chat(self, *, messages: list[dict[str, str]], **_ignored: Any) -> dict[str, Any]:
        """Return ``{"message": {"content": <json text>}}`` like Ollama does.

        Ollama-only arguments (``model``, ``format``, ``think``, ``options``)
        are accepted and ignored; JSON output is requested via Groq's
        ``response_format`` instead. Raises on any HTTP or decode error so the
        extractor's retry loop and spaCy fallback handle it.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": _MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
            "include_reasoning": False,
        }
        request = urllib.request.Request(
            GROQ_CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=json_headers(authorization=f"Bearer {self._api_key}"),
        )
        # URL is the hard-coded Groq endpoint above.
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("Groq returned no choices")
        content = str(choices[0].get("message", {}).get("content") or "")
        return {"message": {"content": content}}


def build_groq_extraction_client() -> GroqExtractionClient | None:
    """A client when ``GROQ_API_KEY`` is set, else None."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    return GroqExtractionClient(api_key) if api_key else None
