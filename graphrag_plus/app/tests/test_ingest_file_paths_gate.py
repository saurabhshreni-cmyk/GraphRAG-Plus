"""``POST /ingest`` must not read server-side paths on a hosted deployment.

``file_paths`` is a self-hosting convenience. On the public API it was
arbitrary file read: a caller could ingest ``/etc/passwd`` into a corpus and
read it back through ``/corpora/{id}/chunks``. Hosted entrypoints turn it off.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from graphrag_plus.app.api import main
from graphrag_plus.app.schemas.models import IngestResponse

client = TestClient(main.app)


@pytest.fixture()
def ingest_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the real pipeline ingest with a recorder."""
    calls: list[dict[str, Any]] = []

    def fake_ingest(file_paths: list[str], urls: list[str], **kwargs: Any) -> IngestResponse:
        calls.append({"file_paths": file_paths, "urls": urls, **kwargs})
        return IngestResponse(documents=0, chunks=0, entities=0, relations=0, graph_version_id="test")

    monkeypatch.setattr(main.pipeline, "ingest", fake_ingest)
    return calls


def test_file_paths_rejected_when_disabled(
    monkeypatch: pytest.MonkeyPatch, ingest_calls: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("GRAPHRAG_ALLOW_FILE_PATHS", "0")

    response = client.post("/ingest", json={"file_paths": ["/etc/passwd"]})

    assert response.status_code == 403
    assert "file paths are disabled" in response.json()["detail"]
    assert ingest_calls == []  # nothing was read


def test_urls_still_accepted_when_file_paths_disabled(
    monkeypatch: pytest.MonkeyPatch, ingest_calls: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("GRAPHRAG_ALLOW_FILE_PATHS", "0")

    response = client.post("/ingest", json={"urls": ["https://example.com/page"]})

    assert response.status_code == 200
    assert ingest_calls[0]["urls"] == ["https://example.com/page"]


@pytest.mark.parametrize("value", [None, "1", "true"])
def test_file_paths_allowed_for_local_use(
    monkeypatch: pytest.MonkeyPatch, ingest_calls: list[dict[str, Any]], value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("GRAPHRAG_ALLOW_FILE_PATHS", raising=False)
    else:
        monkeypatch.setenv("GRAPHRAG_ALLOW_FILE_PATHS", value)

    response = client.post("/ingest", json={"file_paths": ["notes.txt"]})

    assert response.status_code == 200
    assert ingest_calls[0]["file_paths"] == ["notes.txt"]


def test_hosted_entrypoint_disables_file_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # Work on a copy so the module's /tmp path defaults don't leak into the
    # environment other tests run in.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ.pop("GRAPHRAG_ALLOW_FILE_PATHS", None)

    from graphrag_plus.app.api import serverless

    importlib.reload(serverless)

    assert os.environ["GRAPHRAG_ALLOW_FILE_PATHS"] == "0"


def test_hosted_entrypoint_respects_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ["GRAPHRAG_ALLOW_FILE_PATHS"] = "1"

    from graphrag_plus.app.api import serverless

    importlib.reload(serverless)

    assert os.environ["GRAPHRAG_ALLOW_FILE_PATHS"] == "1"
