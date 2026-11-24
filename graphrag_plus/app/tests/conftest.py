"""Shared test configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_rate_limit_by_default(monkeypatch):
    """Keep the public-demo rate limiter out of unrelated tests.

    Every TestClient request comes from the same client key, so an API test
    that issues many queries would otherwise trip the limiter. Tests that
    exercise the limiter re-enable it explicitly.
    """
    monkeypatch.setenv("GRAPHRAG_RATE_LIMIT", "0")
