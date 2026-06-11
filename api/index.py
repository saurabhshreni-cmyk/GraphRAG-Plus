"""Vercel serverless entrypoint for the GraphRAG++ FastAPI backend.

Vercel's filesystem is read-only except ``/tmp``, so every writable path
is redirected there via the ``GRAPHRAG_*`` environment overrides BEFORE
the app (and its module-level pipeline) is imported. The committed demo
corpus is then seeded by the app itself (see ``app.api.main``), so a fresh
``/tmp`` always has a graph to show.

Note: ``/tmp`` is per-instance and ephemeral — ingested corpora survive
only for the lifetime of the serverless instance. For durable multi-user
state, deploy the backend on Render with a persistent disk (see README +
``render.yaml``), which uses this same env-redirect mechanism pointed at a
mounted volume instead of ``/tmp``.
"""

from __future__ import annotations

import os
from pathlib import Path

_DATA = Path("/tmp/graphrag/data")
_CACHE = Path("/tmp/graphrag/.cache")

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

from graphrag_plus.app.api.main import app  # noqa: E402,F401
