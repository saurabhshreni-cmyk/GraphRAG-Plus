"""Structured run logging and output artifact helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from graphrag_plus.app.utils.io_utils import append_jsonl, dump_json


def utc_now_iso() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_run_log(path: Path, payload: Dict[str, Any]) -> None:
    """Append one run log row."""
    append_jsonl(path, [payload])


def write_query_output(outputs_dir: Path, query_id: str, payload: Dict[str, Any]) -> Path:
    """Persist query output artifact to JSON."""
    output_path = outputs_dir / f"{query_id}.json"
    dump_json(output_path, payload)
    return output_path

