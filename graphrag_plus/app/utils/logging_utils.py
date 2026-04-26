"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for console output."""
    logger = logging.getLogger(name)
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
