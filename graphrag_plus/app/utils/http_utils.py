"""Shared HTTP settings for the hand-rolled ``urllib`` API clients.

Groq (and other APIs behind Cloudflare) reject Python's default
``Python-urllib/3.x`` User-Agent with HTTP 403 "error code: 1010" before
the request reaches the API. Every outbound API call sends this instead.
"""

from __future__ import annotations

USER_AGENT = "graphrag-plus/1.0 (+https://github.com/saurabhshreni-cmyk/GraphRAG-Plus)"


def json_headers(**extra: str) -> dict[str, str]:
    """Headers for a JSON POST: content type, our User-Agent, plus ``extra``."""
    return {"content-type": "application/json", "user-agent": USER_AGENT, **extra}
