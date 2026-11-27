"""Shared HTTP settings for the hand-rolled ``urllib`` API clients.

Groq (and other APIs behind Cloudflare) reject Python's default
``Python-urllib/3.x`` User-Agent with HTTP 403 "error code: 1010" before
the request reaches the API. Every outbound API call sends this instead.
"""

from __future__ import annotations

import os

USER_AGENT = "graphrag-plus/1.0 (+https://github.com/saurabhshreni-cmyk/GraphRAG-Plus)"


def json_headers(**extra: str) -> dict[str, str]:
    """Headers for a JSON POST: content type, our User-Agent, plus ``extra``."""
    return {"content-type": "application/json", "user-agent": USER_AGENT, **extra}


def env_secret(*names: str) -> str:
    """First non-empty env var among ``names``, with whitespace trimmed.

    Values pasted into a hosting dashboard often carry a trailing newline;
    sent raw in an HTTP header that makes urllib raise before the request.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""
