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
