"""Per-client rate limiting for the quota-spending endpoints.

A public demo link spends the owner's free LLM and embedding quota on every
query and ingest. Without a brake, one script can exhaust a day's Groq or
Gemini allowance in minutes and leave the demo answering extractively for
everyone else. This module caps how often a single client can hit those
routes.

Scope, honestly stated: state is in-process. On a serverless host each warm
instance keeps its own counters, so the effective limit is "per client per
instance" - it stops bursts and casual scripting, not a distributed attacker.
That matches the threat (protecting a free quota on a portfolio demo) without
adding a datastore dependency. Read-only routes (graph, corpora, health) are
not limited: they cost nothing.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Limit:
    """At most ``max_requests`` per ``window_s`` seconds per client."""

    max_requests: int
    window_s: float


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        return default
    return value if value > 0 else default


DELETE_GROUP = "/corpora:delete"


# Route prefix -> limit. Ingest is far more expensive than a query (it runs LLM
# extraction over many chunks and embeds every one), so it gets a tighter cap.
def default_limits() -> dict[str, Limit]:
    return {
        "/query": Limit(_env_int("GRAPHRAG_RATE_QUERY_PER_MIN", 20), 60.0),
        "/ingest": Limit(_env_int("GRAPHRAG_RATE_INGEST_PER_10MIN", 6), 600.0),
        # Not a URL prefix: DELETE /corpora/{id} is checked against this group
        # explicitly, since GET /corpora must stay unlimited.
        DELETE_GROUP: Limit(_env_int("GRAPHRAG_RATE_DELETE_PER_10MIN", 10), 600.0),
    }


# Bound the number of tracked clients so a flood of distinct IPs can't grow
# memory without limit. Oldest-seen clients are evicted first.
_MAX_TRACKED_CLIENTS = 5000


class RateLimiter:
    """Sliding-window limiter keyed by (client, route group)."""

    def __init__(self, limits: dict[str, Limit] | None = None, clock=time.monotonic) -> None:
        self._limits = limits if limits is not None else default_limits()
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def group_for(self, path: str) -> str | None:
        """The limited route group ``path`` belongs to, or None if unlimited."""
        for prefix in self._limits:
            if path == prefix or path.startswith(prefix + "/"):
                return prefix
        return None

    def check(self, client: str, path: str) -> float | None:
        """Record a request; return seconds to wait if it is over the limit.

        Returns ``None`` when the request is allowed (or the route is not
        limited). A rejected request is not counted, so a client that backs
        off for the advertised time is served.
        """
        group = self.group_for(path)
        if group is None:
            return None
        limit = self._limits[group]
        now = self._clock()
        key = (client, group)
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= _MAX_TRACKED_CLIENTS:
                    self._hits.pop(next(iter(self._hits)))
                hits = deque()
                self._hits[key] = hits
            while hits and now - hits[0] >= limit.window_s:
                hits.popleft()
            if len(hits) >= limit.max_requests:
                return max(0.0, limit.window_s - (now - hits[0]))
            hits.append(now)
            return None


def client_key(headers: dict[str, str], fallback: str) -> str:
    """Best identifier for the caller behind a proxy.

    Vercel (like most PaaS edges) puts the real client address first in
    ``x-forwarded-for``; the socket peer is the proxy itself. Fall back to
    ``x-real-ip`` and then the peer address.
    """
    forwarded = headers.get("x-forwarded-for", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return headers.get("x-real-ip", "").strip() or fallback


__all__ = ["Limit", "RateLimiter", "client_key", "default_limits"]
