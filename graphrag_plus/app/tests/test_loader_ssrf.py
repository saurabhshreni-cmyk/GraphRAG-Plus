"""SSRF guard tests for the URL loader."""

from __future__ import annotations

import pytest

from graphrag_plus.app.ingestion.loader import (
    UnsafeURLError,
    load_documents,
    validate_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file.txt",
        "http://127.0.0.1/admin",
        "http://localhost:8080/",
        "http://10.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata
        "http://[::1]/",
    ],
)
def test_validate_url_blocks_unsafe(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        validate_url(url)


def test_load_documents_silently_skips_bad_url() -> None:
    docs = load_documents(file_paths=[], urls=["http://127.0.0.1/", "ftp://x"])
    assert docs == []
