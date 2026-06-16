"""Smoke-test a real Postgres/Supabase connection for corpus storage.

Usage::

    # PowerShell
    $env:GRAPHRAG_DATABASE_URL="postgresql://...:6543/postgres"
    py -3.13 graphrag_plus/scripts/check_postgres.py

Exercises the full PostgresBlobStore contract (schema creation, save,
load, list, exists, delete) against the live database, then cleans up.
Exits non-zero on any failure so it can gate a deploy.
"""

from __future__ import annotations

import contextlib
import os
import sys

from graphrag_plus.app.corpus.blob_store import CorpusBlobs, make_blob_store


def main() -> int:
    url = os.environ.get("GRAPHRAG_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
    if not url:
        print("FAIL: set GRAPHRAG_DATABASE_URL (or DATABASE_URL) first.")
        return 2

    store = make_blob_store(url)
    if store is None:
        print("FAIL: could not connect / driver missing (see logs above).")
        return 1

    cid = "corpus_pgcheck_tmp"
    blobs = CorpusBlobs(
        meta={"corpus_id": cid, "name": "pg-check", "domain": "general", "created_at": "2026-01-01"},
        graph={"nodes": [{"id": "n1", "label": "hello"}], "edges": []},
        chunks=[{"chunk_id": "c1", "text": "hello world"}],
    )
    try:
        store.save(blobs)
        assert store.exists(cid), "exists() should be True after save"
        loaded = store.load(cid)
        assert loaded is not None, "load() returned None"
        assert loaded.graph["nodes"][0]["label"] == "hello", "graph blob mismatch"
        assert any(m["corpus_id"] == cid for m in store.list_meta()), "not in list_meta()"
        store.delete(cid)
        assert not store.exists(cid), "exists() should be False after delete"
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        # Best-effort cleanup.
        with contextlib.suppress(Exception):
            store.delete(cid)
        return 1

    print("OK: Postgres blob store round-trip passed (schema, save, load, list, delete).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
