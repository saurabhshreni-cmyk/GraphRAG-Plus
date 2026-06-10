"""Vercel serverless entrypoint for the GraphRAG++ FastAPI backend.

Vercel's filesystem is read-only except ``/tmp``, so every writable path
is redirected there via the ``GRAPHRAG_*`` environment overrides BEFORE
the app (and its module-level pipeline) is imported. The committed demo
corpus is seeded into the writable corpora dir on cold start so the
dashboard has a graph immediately.

Note: ``/tmp`` is per-instance and ephemeral — ingested corpora survive
only for the lifetime of the serverless instance. For durable multi-user
state, run the backend on a host with a persistent disk (see README).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
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


def _seed_demo_corpus() -> None:
    """Copy the committed demo corpus into the writable corpora dir."""
    source = _REPO_ROOT / "graphrag_plus" / "data" / "corpora" / "corpus_demo_nova"
    target = _DATA / "corpora" / "corpus_demo_nova"
    if source.is_dir() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)


_seed_demo_corpus()

from graphrag_plus.app.api.main import app  # noqa: E402,F401
