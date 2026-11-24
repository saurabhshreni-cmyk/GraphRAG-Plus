"""ASGI entrypoint for read-only-filesystem hosts (Vercel, containers).

Serverless and most PaaS filesystems are read-only except ``/tmp``, so every
writable path is redirected there via the ``GRAPHRAG_*`` environment
overrides BEFORE the app (and its module-level pipeline) is imported - the
settings object reads them at import time. The app then seeds the demo corpus
itself (see ``app.api.main``), so a fresh ``/tmp`` always has a graph to show.

``/tmp`` is per-instance and ephemeral. For durable, shared-across-instances
corpora set ``GRAPHRAG_DATABASE_URL`` to a Postgres connection string (Neon or
Supabase); the app then keeps corpora in the database and uses ``/tmp`` only
as scratch.

Imported by the repo-root ``index.py`` (Vercel's FastAPI entrypoint) and by
``api/index.py`` (the container entrypoint), so both hosts boot identically.
"""

from __future__ import annotations

import os
from pathlib import Path

_DATA = Path("/tmp/graphrag/data")  # noqa: S108 - the only writable dir on these hosts
_CACHE = Path("/tmp/graphrag/.cache")  # noqa: S108

_PATH_DEFAULTS: dict[str, Path] = {
    "GRAPHRAG_DATA_DIR": _DATA,
    "GRAPHRAG_REPORTS_DIR": _DATA / "reports",
    "GRAPHRAG_GRAPH_PATH": _DATA / "graph.json",
    "GRAPHRAG_CHUNKS_PATH": _DATA / "chunks.json",
    "GRAPHRAG_CORPORA_DIR": _DATA / "corpora",
    "GRAPHRAG_GRAPH_VERSIONS_DIR": _DATA / "graph_versions",
    "GRAPHRAG_TRUST_STATE_PATH": _DATA / "trust_state.json",
    "GRAPHRAG_CALIBRATION_STATE_PATH": _DATA / "calibration_state.json",
    "GRAPHRAG_REVIEW_QUEUE_PATH": _DATA / "review_queue.jsonl",
    "GRAPHRAG_ANSWERS_LOG_PATH": _DATA / "answers_log.jsonl",
    "GRAPHRAG_RUN_LOGS_PATH": _DATA / "run_logs.jsonl",
    "GRAPHRAG_OUTPUTS_DIR": _DATA / "outputs",
    "GRAPHRAG_CACHE_DIR": _CACHE,
    "GRAPHRAG_TEMP_DIR": _CACHE / "tmp",
}

for _name, _path in _PATH_DEFAULTS.items():
    os.environ.setdefault(_name, str(_path))

from graphrag_plus.app.api.main import app  # noqa: E402 - must follow the env setup above

__all__ = ["app"]
