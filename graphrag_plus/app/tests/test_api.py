"""API tests."""

from fastapi.testclient import TestClient

from graphrag_plus.app.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_query_no_evidence_failure_mode() -> None:
    response = client.post(
        "/query",
        json={"question": "unknown query", "top_k": 3, "analyst_mode": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert "failure_type" in body


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    # Should be plain text whether prometheus-client is installed or not.
    assert "text/plain" in response.headers.get("content-type", "")


def test_demo_corpus_is_listed_as_not_deletable_and_refused() -> None:
    from graphrag_plus.app.corpus.seed import DEMO_CORPUS_ID

    listing = {c["corpus_id"]: c for c in client.get("/corpora").json()}
    if DEMO_CORPUS_ID in listing:
        assert listing[DEMO_CORPUS_ID]["deletable"] is False
    response = client.delete(f"/corpora/{DEMO_CORPUS_ID}")
    assert response.status_code == 403


def test_an_uploaded_corpus_can_be_deleted() -> None:
    from graphrag_plus.app.api.main import pipeline

    corpus_id = pipeline.corpus_manager.create(name="Delete me", domain="general").meta.corpus_id
    listing = {c["corpus_id"]: c for c in client.get("/corpora").json()}
    assert listing[corpus_id]["deletable"] is True

    assert client.delete(f"/corpora/{corpus_id}").status_code == 200
    assert corpus_id not in {c["corpus_id"] for c in client.get("/corpora").json()}
    assert client.delete(f"/corpora/{corpus_id}").status_code == 404


def test_deletes_are_rate_limited(monkeypatch) -> None:
    from graphrag_plus.app.api import main
    from graphrag_plus.app.api.rate_limit import DELETE_GROUP, Limit, RateLimiter

    monkeypatch.setenv("GRAPHRAG_RATE_LIMIT", "1")
    monkeypatch.setattr(main, "_rate_limiter", RateLimiter({DELETE_GROUP: Limit(1, 600.0)}))
    first = main.pipeline.corpus_manager.create(name="One", domain="general").meta.corpus_id
    second = main.pipeline.corpus_manager.create(name="Two", domain="general").meta.corpus_id
    try:
        assert client.delete(f"/corpora/{first}").status_code == 200
        blocked = client.delete(f"/corpora/{second}")
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers
    finally:
        monkeypatch.setenv("GRAPHRAG_RATE_LIMIT", "0")
        client.delete(f"/corpora/{second}")
