"""Multi-corpus manager — keeps each ingestion fully isolated.

Disk layout::

    data/corpora/
      corpus_<id>/
        meta.json       — CorpusMeta as JSON
        graph.json      — GraphStore persistence
        chunks.json     — RetrievalService persistence
      corpus_<id2>/
        ...

The manager is constructed once per pipeline. Bundles are loaded lazily on
first access and cached. Creating a new corpus always allocates a fresh
``GraphStore`` + ``RetrievalService`` pair so the new corpus starts clean.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from graphrag_plus.app.corpus.models import CorpusBundle, CorpusMeta
from graphrag_plus.app.graph.store import GraphStore
from graphrag_plus.app.retrieval.service import RetrievalService
from graphrag_plus.app.utils.io_utils import dump_json, load_json
from graphrag_plus.app.utils.logging_utils import get_logger
from graphrag_plus.app.utils.run_logger import utc_now_iso

logger = get_logger(__name__)

# corpus_id arrives from API path params and flows into filesystem paths
# (including ``shutil.rmtree``), so it must match this strict allowlist:
# the literal ``corpus_`` prefix plus 1-64 word characters. Anything else
# (path separators, ``..``, drive letters, empty) is rejected outright.
_CORPUS_ID_RE = re.compile(r"^corpus_[A-Za-z0-9_-]{1,64}$")


class CorpusManager:
    """Owns the directory of corpora and their loaded bundles."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._bundles: dict[str, CorpusBundle] = {}
        # The "active" corpus is the one used when callers don't specify a
        # corpus_id. Defaults to the most recently created or queried.
        self.active_corpus_id: str | None = None
        self._load_existing_meta()

    # ------------------------------------------------------------------ paths
    def _corpus_dir(self, corpus_id: str) -> Path:
        if not _CORPUS_ID_RE.match(corpus_id):
            raise KeyError(f"Unknown corpus_id: {corpus_id!r}")
        return self.base_dir / corpus_id

    def _meta_path(self, corpus_id: str) -> Path:
        return self._corpus_dir(corpus_id) / "meta.json"

    def _graph_path(self, corpus_id: str) -> Path:
        return self._corpus_dir(corpus_id) / "graph.json"

    def _chunks_path(self, corpus_id: str) -> Path:
        return self._corpus_dir(corpus_id) / "chunks.json"

    # ----------------------------------------------------------------- create
    def create(
        self,
        *,
        name: str,
        domain: str,
        source_urls: list[str] | None = None,
        source_files: list[str] | None = None,
    ) -> CorpusBundle:
        """Allocate a brand-new isolated corpus on disk + in memory."""
        corpus_id = f"corpus_{uuid.uuid4().hex[:8]}"
        corpus_dir = self._corpus_dir(corpus_id)
        corpus_dir.mkdir(parents=True, exist_ok=True)

        meta = CorpusMeta(
            corpus_id=corpus_id,
            name=name or corpus_id,
            domain=domain,
            source_urls=list(source_urls or []),
            source_files=list(source_files or []),
            created_at=utc_now_iso(),
        )
        self._persist_meta(meta)

        graph_store = GraphStore(self._graph_path(corpus_id))
        retrieval = RetrievalService(graph_store, self._chunks_path(corpus_id))
        bundle = CorpusBundle(meta=meta, graph_store=graph_store, retrieval=retrieval)
        self._bundles[corpus_id] = bundle
        self.active_corpus_id = corpus_id
        logger.info(
            "corpus.created id=%s name=%r domain=%s",
            corpus_id,
            meta.name,
            domain,
        )
        return bundle

    # -------------------------------------------------------------------- get
    def get(self, corpus_id: str) -> CorpusBundle:
        """Return the bundle for ``corpus_id``. Loads from disk if needed."""
        if corpus_id in self._bundles:
            return self._bundles[corpus_id]
        meta = self._read_meta(corpus_id)
        if meta is None:
            raise KeyError(f"Unknown corpus_id: {corpus_id}")
        graph_store = GraphStore(self._graph_path(corpus_id))
        retrieval = RetrievalService(graph_store, self._chunks_path(corpus_id))
        bundle = CorpusBundle(meta=meta, graph_store=graph_store, retrieval=retrieval)
        self._bundles[corpus_id] = bundle
        return bundle

    def list(self) -> list[CorpusMeta]:
        """Return all known corpora's metadata, newest first."""
        metas: list[CorpusMeta] = []
        for entry in self.base_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("corpus_"):
                continue
            meta = self._read_meta(entry.name)
            if meta is not None:
                metas.append(meta)
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas

    def delete(self, corpus_id: str) -> None:
        """Permanently remove a corpus directory + cached bundle."""
        bundle = self._bundles.pop(corpus_id, None)
        _ = bundle  # released; GraphStore has no explicit close
        path = self._corpus_dir(corpus_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        if self.active_corpus_id == corpus_id:
            remaining = self.list()
            self.active_corpus_id = remaining[0].corpus_id if remaining else None
        logger.info("corpus.deleted id=%s", corpus_id)

    # ----------------------------------------------------------------- active
    def get_active(self) -> CorpusBundle | None:
        """Return the active bundle, or ``None`` if no corpora exist."""
        if self.active_corpus_id is None:
            metas = self.list()
            if not metas:
                return None
            self.active_corpus_id = metas[0].corpus_id
        return self.get(self.active_corpus_id)

    def set_active(self, corpus_id: str) -> CorpusBundle:
        """Switch the active corpus. Raises ``KeyError`` if unknown."""
        bundle = self.get(corpus_id)  # validates existence
        self.active_corpus_id = corpus_id
        return bundle

    def update_meta(self, corpus_id: str, **fields: object) -> CorpusMeta:
        """Update select metadata fields and persist."""
        bundle = self.get(corpus_id)
        old = asdict(bundle.meta)
        old.update(fields)
        new_meta = CorpusMeta(**old)  # type: ignore[arg-type]
        bundle.meta = new_meta
        self._persist_meta(new_meta)
        return new_meta

    # --------------------------------------------------------------- internals
    def _load_existing_meta(self) -> None:
        for meta in self.list():
            # Don't eagerly load GraphStore/Retrieval — bundles are created
            # on first access via ``get(corpus_id)``.
            _ = meta
        if self.active_corpus_id is None:
            metas = self.list()
            if metas:
                self.active_corpus_id = metas[0].corpus_id

    def _read_meta(self, corpus_id: str) -> CorpusMeta | None:
        path = self._meta_path(corpus_id)
        if not path.exists():
            return None
        raw = load_json(path, default=None)
        if not isinstance(raw, dict):
            return None
        try:
            return CorpusMeta(**raw)
        except TypeError as exc:
            logger.warning("corpus.meta_invalid id=%s error=%s", corpus_id, exc)
            return None

    def _persist_meta(self, meta: CorpusMeta) -> None:
        dump_json(self._meta_path(meta.corpus_id), asdict(meta))


# --- helpers ---------------------------------------------------------------


def derive_corpus_name(file_paths: list[str], urls: list[str]) -> str:
    """Pick a human-readable name from the first source.

    URLs win over files (more descriptive). Wikipedia URLs get their last
    path segment used as the name (``"Long_short-term_memory"`` →
    ``"Long short-term memory"``). Otherwise we fall back to the file stem.
    """
    if urls:
        url = urls[0]
        match = re.search(r"/wiki/([^/?#]+)", url)
        if match:
            slug = match.group(1).replace("_", " ")
            return _truncate(slug, 80)
        # Generic URL → use last meaningful path segment.
        tail = url.rstrip("/").rsplit("/", 1)[-1] or url
        return _truncate(tail.replace("_", " ").replace("-", " "), 80)
    if file_paths:
        return _truncate(Path(file_paths[0]).stem, 80)
    return "untitled-corpus"


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def export_meta(meta: CorpusMeta) -> dict[str, object]:
    """JSON-safe serialization for API responses."""
    return json.loads(json.dumps(asdict(meta)))
