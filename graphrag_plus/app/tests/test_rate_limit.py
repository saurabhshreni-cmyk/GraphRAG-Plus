"""Tests for the public-demo rate limiter."""

from __future__ import annotations

import pytest

from graphrag_plus.app.api.rate_limit import Limit, RateLimiter, client_key


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def limiter(clock) -> RateLimiter:
    return RateLimiter({"/query": Limit(3, 60.0), "/ingest": Limit(1, 600.0)}, clock=clock)


class TestRateLimiter:
    def test_allows_up_to_the_limit(self, limiter):
        assert all(limiter.check("a", "/query") is None for _ in range(3))

    def test_rejects_past_the_limit_with_a_wait_time(self, limiter, clock):
        for _ in range(3):
            limiter.check("a", "/query")
        clock.now += 10
        wait = limiter.check("a", "/query")
        assert wait == pytest.approx(50.0)

    def test_window_slides(self, limiter, clock):
        for _ in range(3):
            limiter.check("a", "/query")
        clock.now += 60
        assert limiter.check("a", "/query") is None

    def test_rejected_requests_are_not_counted(self, limiter, clock):
        for _ in range(3):
            limiter.check("a", "/query")
        for _ in range(10):
            limiter.check("a", "/query")
        # A client that waits out the advertised time is served, rather than
        # being locked out by the rejections it accumulated while waiting.
        clock.now += 60
        assert limiter.check("a", "/query") is None

    def test_clients_are_independent(self, limiter):
        for _ in range(3):
            limiter.check("a", "/query")
        assert limiter.check("b", "/query") is None

    def test_route_groups_are_independent(self, limiter):
        for _ in range(3):
            limiter.check("a", "/query")
        assert limiter.check("a", "/ingest") is None

    def test_sub_paths_share_their_group(self, limiter):
        assert limiter.check("a", "/ingest") is None
        assert limiter.check("a", "/ingest/file") is not None

    def test_unlimited_routes_pass_through(self, limiter):
        assert all(limiter.check("a", "/graph/c/full") is None for _ in range(100))

    def test_prefix_match_does_not_overreach(self, limiter):
        # "/queryx" is not "/query" or a sub-path of it.
        assert all(limiter.check("a", "/queryx") is None for _ in range(10))


class TestClientKey:
    def test_uses_the_first_forwarded_address(self):
        assert client_key({"x-forwarded-for": "203.0.113.7, 10.0.0.1"}, "peer") == "203.0.113.7"

    def test_falls_back_to_real_ip_then_peer(self):
        assert client_key({"x-real-ip": "198.51.100.2"}, "peer") == "198.51.100.2"
        assert client_key({}, "peer") == "peer"


class TestMiddleware:
    def test_returns_429_with_retry_after_and_cors(self, monkeypatch):
        from fastapi.testclient import TestClient

        from graphrag_plus.app.api import main

        monkeypatch.setenv("GRAPHRAG_RATE_LIMIT", "1")
        monkeypatch.setattr(main, "_rate_limiter", RateLimiter({"/query": Limit(1, 60.0)}))
        client = TestClient(main.app)
        headers = {"Origin": "http://localhost:5173"}

        client.post("/query", json={"question": "q"}, headers=headers)
        response = client.post("/query", json={"question": "q"}, headers=headers)

        assert response.status_code == 429
        assert int(response.headers["retry-after"]) >= 1
        assert "rate-limited" in response.json()["detail"]
        # Without CORS headers the browser hides this message entirely.
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_read_only_routes_are_never_limited(self, monkeypatch):
        from fastapi.testclient import TestClient

        from graphrag_plus.app.api import main

        monkeypatch.setenv("GRAPHRAG_RATE_LIMIT", "1")
        monkeypatch.setattr(main, "_rate_limiter", RateLimiter({"/query": Limit(1, 60.0)}))
        client = TestClient(main.app)
        assert all(client.get("/health").status_code == 200 for _ in range(5))

    def test_oversized_upload_gets_a_clear_413(self, monkeypatch):
        from fastapi.testclient import TestClient

        from graphrag_plus.app.api import main

        monkeypatch.setattr(main, "_MAX_UPLOAD_BYTES", 10)
        client = TestClient(main.app)
        response = client.post("/ingest/file", files={"file": ("big.txt", b"x" * 11, "text/plain")})
        assert response.status_code == 413
        assert "larger than" in response.json()["detail"]
