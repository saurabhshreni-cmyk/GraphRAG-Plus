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

import base64
import binascii
import json
import re
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from graphrag_plus.app.corpus.blob_store import (
    EMPTY_CHUNKS,
    EMPTY_GRAPH,
    BlobStore,
    CorpusBlobs,
)
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

    def __init__(self, base_dir: Path, blob_store: BlobStore | None = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # When set, the blob store is the durable source of truth and
        # ``base_dir`` is per-instance scratch. When None, ``base_dir`` is
        # the source of truth (single-host / local file mode).
        self.blob_store = blob_store
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
        # Register the (empty) corpus in the durable store so other instances
        # can see it immediately; ingest will flush its populated blobs later.
        if self.blob_store is not None:
            self.blob_store.save(
                CorpusBlobs(meta=asdict(meta), graph=dict(EMPTY_GRAPH), chunks=list(EMPTY_CHUNKS))
            )
        logger.info(
            "corpus.created id=%s name=%r domain=%s",
            corpus_id,
            meta.name,
            domain,
        )
        return bundle

    # -------------------------------------------------------------------- get
    def get(self, corpus_id: str) -> CorpusBundle:
        """Return the bundle for ``corpus_id``.

        In blob-store mode the corpus is hydrated from the durable store
        into the local scratch dir on a cache miss, so an ingest performed
        on another instance is visible here. In file mode it loads from the
        local directory.
        """
        if corpus_id in self._bundles:
            # In blob mode another instance may have deleted it since we
            # cached it; drop the stale copy rather than keep serving it.
            if self.blob_store is not None and not self.blob_store.exists(corpus_id):
                self._bundles.pop(corpus_id, None)
                raise KeyError(f"Unknown corpus_id: {corpus_id}")
            return self._bundles[corpus_id]
        if self.blob_store is not None:
            return self._hydrate_from_store(corpus_id)
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
        if self.blob_store is not None:
            for raw in self.blob_store.list_meta():
                meta = self._meta_from_dict(raw)
                if meta is not None:
                    metas.append(meta)
        else:
            for entry in self.base_dir.iterdir():
                if not entry.is_dir() or not entry.name.startswith("corpus_"):
                    continue
                meta = self._read_meta(entry.name)
                if meta is not None:
                    metas.append(meta)
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas

    def delete(self, corpus_id: str) -> None:
        """Permanently remove a corpus from the store + local scratch."""
        # Validate id shape even in blob mode (also guards the rmtree below).
        path = self._corpus_dir(corpus_id)
        known = self.blob_store.exists(corpus_id) if self.blob_store is not None else path.exists()
        if not known and corpus_id not in self._bundles:
            raise KeyError(f"Unknown corpus_id: {corpus_id}")
        self._bundles.pop(corpus_id, None)
        if self.blob_store is not None:
            self.blob_store.delete(corpus_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        if self.active_corpus_id == corpus_id:
            remaining = self.list()
            self.active_corpus_id = remaining[0].corpus_id if remaining else None
        logger.info("corpus.deleted id=%s", corpus_id)

    def flush(self, corpus_id: str) -> None:
        """Persist a corpus's current local files back to the durable store.

        No-op in file mode (the files *are* the source of truth). Called
        after a mutating ingest so the populated graph + chunks become
        visible to every other instance.
        """
        if self.blob_store is None:
            return
        meta = self._read_meta(corpus_id)
        if meta is None:
            logger.warning("corpus.flush_no_meta id=%s", corpus_id)
            return
        graph = load_json(self._graph_path(corpus_id), default=None) or dict(EMPTY_GRAPH)
        chunks = load_json(self._chunks_path(corpus_id), default=None) or list(EMPTY_CHUNKS)
        self.blob_store.save(
            CorpusBlobs(
                meta=asdict(meta),
                graph=graph,
                chunks=chunks,
                vectors=self._read_vectors(corpus_id),
            )
        )
        logger.info("corpus.flushed id=%s", corpus_id)

    # ----------------------------------------------------------------- active
    def get_active(self) -> CorpusBundle | None:
        """Return the active bundle, or ``None`` if no corpora exist."""
        if self.active_corpus_id is not None:
            try:
                return self.get(self.active_corpus_id)
            except KeyError:
                # Deleted (possibly by another instance): fall back below.
                self.active_corpus_id = None
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
        """Update select metadata fields and persist (local + store)."""
        bundle = self.get(corpus_id)
        old = asdict(bundle.meta)
        old.update(fields)
        new_meta = CorpusMeta(**old)  # type: ignore[arg-type]
        bundle.meta = new_meta
        self._persist_meta(new_meta)
        if self.blob_store is not None:
            # Persist the meta change immediately, preserving existing blobs.
            existing = self.blob_store.load(corpus_id)
            graph = existing.graph if existing else dict(EMPTY_GRAPH)
            chunks = existing.chunks if existing else list(EMPTY_CHUNKS)
            self.blob_store.save(CorpusBlobs(meta=asdict(new_meta), graph=graph, chunks=chunks))
        return new_meta

    # --------------------------------------------------------------- internals
    def _hydrate_from_store(self, corpus_id: str) -> CorpusBundle:
        """Pull a corpus's blobs from the store into local scratch files."""
        assert self.blob_store is not None
        blobs = self.blob_store.load(corpus_id)
        if blobs is None:
            raise KeyError(f"Unknown corpus_id: {corpus_id}")
        meta = self._meta_from_dict(blobs.meta)
        if meta is None:
            raise KeyError(f"Corrupt corpus meta: {corpus_id}")
        # Materialize the three files so GraphStore/RetrievalService load them.
        self._corpus_dir(corpus_id).mkdir(parents=True, exist_ok=True)
        self._persist_meta(meta)
        dump_json(self._graph_path(corpus_id), blobs.graph or dict(EMPTY_GRAPH))
        dump_json(self._chunks_path(corpus_id), blobs.chunks or list(EMPTY_CHUNKS))
        if blobs.vectors:
            self._write_vectors(corpus_id, blobs.vectors)
        graph_store = GraphStore(self._graph_path(corpus_id))
        # Reuses the restored FAISS index when its chunk ids and embedding
        # model still match; otherwise embeds the chunks afresh.
        retrieval = RetrievalService(graph_store, self._chunks_path(corpus_id))
        bundle = CorpusBundle(meta=meta, graph_store=graph_store, retrieval=retrieval)
        self._bundles[corpus_id] = bundle
        # If that meant a fresh embedding pass, store the result so the next
        # instance doesn't pay for it again.
        current = self._read_vectors(corpus_id)
        if current is not None and current != blobs.vectors:
            try:
                self.blob_store.save_vectors(corpus_id, current)
                logger.info("corpus.vectors_saved id=%s", corpus_id)
            except Exception as exc:  # non-fatal: this instance still has them
                logger.warning("corpus.vectors_save_failed id=%s error=%s", corpus_id, str(exc)[:200])
        return bundle

    # --------------------------------------------------------- vector files
    def _faiss_paths(self, corpus_id: str) -> tuple[Path, Path]:
        index_path = self._corpus_dir(corpus_id) / "faiss.index"
        return index_path, index_path.with_suffix(index_path.suffix + ".meta.json")

    def _read_vectors(self, corpus_id: str) -> dict[str, object] | None:
        """The corpus's saved FAISS index as a JSON-safe dict, or None."""
        index_path, meta_path = self._faiss_paths(corpus_id)
        if not index_path.exists() or not meta_path.exists():
            return None
        try:
            return {
                "meta": json.loads(meta_path.read_text(encoding="utf-8")),
                "index_b64": base64.b64encode(index_path.read_bytes()).decode("ascii"),
            }
        except (OSError, ValueError) as exc:
            logger.warning("corpus.vectors_read_failed id=%s error=%s", corpus_id, str(exc)[:200])
            return None

    def _write_vectors(self, corpus_id: str, vectors: dict[str, object]) -> None:
        """Materialize a stored FAISS index so RetrievalService can reuse it."""
        index_path, meta_path = self._faiss_paths(corpus_id)
        try:
            raw = base64.b64decode(str(vectors["index_b64"]), validate=True)
            meta_path.write_text(json.dumps(vectors["meta"]), encoding="utf-8")
            index_path.write_bytes(raw)
        except (KeyError, OSError, binascii.Error, TypeError) as exc:
            logger.warning("corpus.vectors_restore_failed id=%s error=%s", corpus_id, str(exc)[:200])

    def _load_existing_meta(self) -> None:
        if self.active_corpus_id is None:
            metas = self.list()
            if metas:
                self.active_corpus_id = metas[0].corpus_id

    def _meta_from_dict(self, raw: object) -> CorpusMeta | None:
        if not isinstance(raw, dict):
            return None
        try:
            return CorpusMeta(**raw)
        except TypeError as exc:
            logger.warning("corpus.meta_invalid error=%s", exc)
            return None

    def _read_meta(self, corpus_id: str) -> CorpusMeta | None:
        path = self._meta_path(corpus_id)
        if not path.exists():
            return None
        return self._meta_from_dict(load_json(path, default=None))

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
