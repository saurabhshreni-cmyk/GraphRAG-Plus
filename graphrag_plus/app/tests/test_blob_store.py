"""BlobStore contract + CorpusManager integration in blob-store mode.

These cover the durable-storage logic without a real database by using
``InMemoryBlobStore``; the Postgres SQL is validated separately against a
live connection (scripts/check_postgres.py).
"""

from pathlib import Path

import pytest

from graphrag_plus.app.corpus.blob_store import (
    BlobStore,
    CorpusBlobs,
    InMemoryBlobStore,
    make_blob_store,
)
from graphrag_plus.app.corpus.manager import CorpusManager


def test_make_blob_store_none_when_no_url() -> None:
    assert make_blob_store("") is None


def test_inmemory_store_is_a_blobstore() -> None:
    assert isinstance(InMemoryBlobStore(), BlobStore)


def test_inmemory_crud_roundtrip() -> None:
    store = InMemoryBlobStore()
    blobs = CorpusBlobs(
        meta={"corpus_id": "corpus_abc", "name": "A"},
        graph={"nodes": [{"id": "n1"}], "edges": []},
        chunks=[{"chunk_id": "c1"}],
    )
    assert store.exists("corpus_abc") is False
    store.save(blobs)
    assert store.exists("corpus_abc") is True
    loaded = store.load("corpus_abc")
    assert loaded is not None and loaded.graph["nodes"][0]["id"] == "n1"
    assert [m["corpus_id"] for m in store.list_meta()] == ["corpus_abc"]
    store.delete("corpus_abc")
    assert store.load("corpus_abc") is None


def test_create_registers_corpus_in_store(tmp_path: Path) -> None:
    store = InMemoryBlobStore()
    mgr = CorpusManager(tmp_path / "scratch", blob_store=store)
    bundle = mgr.create(name="Demo", domain="general")
    assert store.exists(bundle.meta.corpus_id)
    assert any(m["corpus_id"] == bundle.meta.corpus_id for m in store.list_meta())


def test_list_reads_from_store_not_local_dir(tmp_path: Path) -> None:
    store = InMemoryBlobStore()
    mgr = CorpusManager(tmp_path / "scratch", blob_store=store)
    mgr.create(name="One", domain="general")
    mgr.create(name="Two", domain="general")
    names = {m.name for m in mgr.list()}
    assert names == {"One", "Two"}


def test_delete_removes_from_store(tmp_path: Path) -> None:
    store = InMemoryBlobStore()
    mgr = CorpusManager(tmp_path / "scratch", blob_store=store)
    bundle = mgr.create(name="Temp", domain="general")
    cid = bundle.meta.corpus_id
    mgr.delete(cid)
    assert store.exists(cid) is False
    with pytest.raises(KeyError):
        mgr.get(cid)


def test_unknown_corpus_id_still_validated_in_blob_mode(tmp_path: Path) -> None:
    # Path-traversal ids must be rejected even when the store, not the FS, is
    # the source of truth (delete still touches the local scratch dir).
    store = InMemoryBlobStore()
    mgr = CorpusManager(tmp_path / "scratch", blob_store=store)
    with pytest.raises(KeyError):
        mgr.get("../escape")
    with pytest.raises(KeyError):
        mgr.delete("../escape")
