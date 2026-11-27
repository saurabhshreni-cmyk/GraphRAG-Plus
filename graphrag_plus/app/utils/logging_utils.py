"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

# API-key shapes that must never reach a log line: an exception message can
# embed a whole request header (it did: a key with a trailing newline made
# urllib raise "Invalid header value b'Bearer <key>\n'").
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"\bgsk_[A-Za-z0-9]{8,}"),  # Groq
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),  # Google / Gemini
    re.compile(r"\bjina_[A-Za-z0-9]{8,}"),  # Jina
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}"),  # Anthropic
    # Database URLs carry a password: postgresql://user:PASSWORD@host/db
    re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s'\"]+"),
)
REDACTED = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Replace anything shaped like an API key or bearer token."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


class RedactSecretsFilter(logging.Filter):
    """Scrub secrets from a record before any handler (or the root logger
    it propagates to) formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # malformed args: leave the record to logging's own handling
            return True
        cleaned = redact_secrets(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = None
        return True


_REDACT_FILTER = RedactSecretsFilter()


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for console output, with secrets redacted."""
    logger = logging.getLogger(name)
    if _REDACT_FILTER not in logger.filters:
        logger.addFilter(_REDACT_FILTER)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, payload: dict[str, Any]) -> None:
    """Log a structured event safely."""
    logger.info("%s %s", event, json.dumps(payload, default=str))
