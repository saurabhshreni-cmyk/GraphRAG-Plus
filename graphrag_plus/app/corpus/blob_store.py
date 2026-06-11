"""Durable blob storage for corpora — file-free, shared across instances.

Each corpus is three JSON documents: its ``meta``, its serialized graph
(``{"nodes": [...], "edges": [...]}``), and its retrieval ``chunks``
(``[chunk, ...]``). On a single host these live on disk; on stateless
serverless (Vercel) they must live somewhere every instance can reach.

A :class:`BlobStore` is that somewhere. :class:`PostgresBlobStore` keeps
the blobs in one Postgres table (works with Supabase/Neon free tiers);
:class:`InMemoryBlobStore` is a process-local stand-in used by tests and
as the fallback when no database is configured.

The :class:`CorpusManager` treats the blob store as the source of truth
and the local directory as scratch: it hydrates a corpus's files from the
store on access and flushes them back after a mutating ingest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Empty-but-valid blob shapes, so a brand-new corpus round-trips cleanly.
EMPTY_GRAPH: dict[str, list[Any]] = {"nodes": [], "edges": []}
EMPTY_CHUNKS: list[Any] = []


@dataclass(frozen=True)
class CorpusBlobs:
    """The three JSON documents that fully describe a persisted corpus."""

    meta: dict[str, Any]
    graph: dict[str, Any]
    chunks: list[Any]


@runtime_checkable
class BlobStore(Protocol):
    """Durable key/value store of corpora keyed by ``corpus_id``."""

    def list_meta(self) -> list[dict[str, Any]]:
        """Return every corpus's ``meta`` dict (order unspecified)."""
        ...

    def load(self, corpus_id: str) -> CorpusBlobs | None:
        """Return the corpus's blobs, or ``None`` if it doesn't exist."""
        ...

    def save(self, blobs: CorpusBlobs) -> None:
        """Upsert a corpus (keyed by ``blobs.meta['corpus_id']``)."""
        ...

    def delete(self, corpus_id: str) -> None:
        """Remove a corpus. A no-op if it doesn't exist."""
        ...

    def exists(self, corpus_id: str) -> bool:
        """True if the corpus is present."""
        ...


class InMemoryBlobStore:
    """Process-local :class:`BlobStore`. Used in tests and as a safe default.

    Not shared across processes — it exists so the manager logic can be
    exercised without a database, and so a misconfigured deployment still
    boots (degrading to per-instance state rather than crashing).
    """

    def __init__(self) -> None:
        self._data: dict[str, CorpusBlobs] = {}

    def list_meta(self) -> list[dict[str, Any]]:
        return [dict(blobs.meta) for blobs in self._data.values()]

    def load(self, corpus_id: str) -> CorpusBlobs | None:
        return self._data.get(corpus_id)

    def save(self, blobs: CorpusBlobs) -> None:
        corpus_id = str(blobs.meta["corpus_id"])
        self._data[corpus_id] = blobs

    def delete(self, corpus_id: str) -> None:
        self._data.pop(corpus_id, None)

    def exists(self, corpus_id: str) -> bool:
        return corpus_id in self._data


class PostgresBlobStore:
    """:class:`BlobStore` backed by a single Postgres table (jsonb columns).

    Opens a short-lived connection per operation — friendly to serverless
    invocations and pgbouncer transaction pooling (use Supabase's pooled
    connection string, port 6543). The table is created on first use.
    """

    _TABLE = "graphrag_corpora"

    def __init__(self, dsn: str) -> None:
        # Import lazily so the package works without psycopg installed when
        # no database is configured.
        import psycopg  # noqa: F401, PLC0415  (import-time availability check)

        self._dsn = dsn
        self._ensure_schema()

    # --------------------------------------------------------------- internals
    def _connect(self) -> Any:
        import psycopg  # noqa: PLC0415

        return psycopg.connect(self._dsn)

    def _ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    corpus_id  TEXT PRIMARY KEY,
                    meta       JSONB NOT NULL,
                    graph      JSONB NOT NULL,
                    chunks     JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)
            conn.commit()

    # ------------------------------------------------------------------- ops
    def list_meta(self) -> list[dict[str, Any]]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT meta FROM {self._TABLE}")
            return [row[0] for row in cur.fetchall()]

    def load(self, corpus_id: str) -> CorpusBlobs | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT meta, graph, chunks FROM {self._TABLE} WHERE corpus_id = %s",
                (corpus_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return CorpusBlobs(meta=row[0], graph=row[1], chunks=row[2])

    def save(self, blobs: CorpusBlobs) -> None:
        from psycopg.types.json import Jsonb  # noqa: PLC0415

        corpus_id = str(blobs.meta["corpus_id"])
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._TABLE} (corpus_id, meta, graph, chunks, updated_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (corpus_id) DO UPDATE SET
                    meta = EXCLUDED.meta,
                    graph = EXCLUDED.graph,
                    chunks = EXCLUDED.chunks,
                    updated_at = now()
                """,
                (corpus_id, Jsonb(blobs.meta), Jsonb(blobs.graph), Jsonb(blobs.chunks)),
            )
            conn.commit()

    def delete(self, corpus_id: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._TABLE} WHERE corpus_id = %s", (corpus_id,))
            conn.commit()

    def exists(self, corpus_id: str) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {self._TABLE} WHERE corpus_id = %s", (corpus_id,))
            return cur.fetchone() is not None


def make_blob_store(database_url: str) -> BlobStore | None:
    """Build the configured blob store, or ``None`` for file-based storage.

    Never raises: if a URL is set but the driver is missing or the database
    is unreachable, logs and returns ``None`` so the app still boots in
    file mode rather than crashing on startup.
    """
    if not database_url:
        return None
    try:
        store = PostgresBlobStore(database_url)
        logger.info("blob_store.postgres_ready")
        return store
    except Exception as exc:  # pragma: no cover - exercised only on misconfig
        logger.error(
            "blob_store.postgres_unavailable error=%s — falling back to file storage",
            exc,
        )
        return None
