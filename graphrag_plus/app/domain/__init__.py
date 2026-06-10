"""Lightweight keyword-based domain classifier.

Tags each corpus with its dominant subject area so retrieval and the UI
can prioritize / colour-code accordingly. No ML dependency — runs in
microseconds against the ingested text.
"""

from graphrag_plus.app.domain.detector import detect_domain

__all__ = ["detect_domain"]
