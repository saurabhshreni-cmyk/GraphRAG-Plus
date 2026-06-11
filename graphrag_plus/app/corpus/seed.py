"""Seed the committed demo corpus into the active storage backend.

Both stateless targets start empty — Vercel's per-instance ``/tmp`` and a
fresh database — so the dashboard would render an empty graph until someone
ingests. This module makes sure ``corpus_demo_nova`` is always present.

In blob-store (Postgres) mode the demo is upserted into the durable store
so every instance sees it; in file mode it's copied into the corpora dir.
Both are idempotent and locate the source relative to the installed
package, so they work regardless of the process working directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import graphrag_plus
from graphrag_plus.app.corpus.blob_store import BlobStore, CorpusBlobs
from graphrag_plus.app.utils.io_utils import load_json
from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEMO_CORPUS_ID = "corpus_demo_nova"

# The committed demo lives alongside the package: <pkg>/data/corpora/<id>.
_PACKAGE_ROOT = Path(graphrag_plus.__file__).resolve().parent
_DEMO_SOURCE = _PACKAGE_ROOT / "data" / "corpora" / DEMO_CORPUS_ID


def seed_demo_corpus(corpora_dir: Path | str, blob_store: BlobStore | None = None) -> bool:
    """Ensure the demo corpus exists in the active storage backend.

    Returns ``True`` when a seed happened, ``False`` otherwise (already
    present, or the source is missing). Never raises on a missing source —
    a deployment without the demo files should still boot.
    """
    if not _DEMO_SOURCE.is_dir():
        logger.warning("demo_seed.source_missing path=%s", _DEMO_SOURCE)
        return False

    if blob_store is not None:
        if blob_store.exists(DEMO_CORPUS_ID):
            return False
        blobs = _read_demo_blobs()
        if blobs is None:
            return False
        blob_store.save(blobs)
        logger.info("demo_seed.stored id=%s", DEMO_CORPUS_ID)
        return True

    target = Path(corpora_dir) / DEMO_CORPUS_ID
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_DEMO_SOURCE, target)
    logger.info("demo_seed.copied into=%s", target.parent)
    return True


def _read_demo_blobs() -> CorpusBlobs | None:
    """Read the committed demo's three JSON files into a ``CorpusBlobs``."""
    meta = load_json(_DEMO_SOURCE / "meta.json", default=None)
    if not isinstance(meta, dict):
        logger.warning("demo_seed.meta_unreadable")
        return None
    graph = load_json(_DEMO_SOURCE / "graph.json", default=None) or {"nodes": [], "edges": []}
    chunks = load_json(_DEMO_SOURCE / "chunks.json", default=None) or []
    return CorpusBlobs(meta=meta, graph=graph, chunks=chunks)
