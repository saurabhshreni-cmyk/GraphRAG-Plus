"""Graph version and drift manager."""

from __future__ import annotations

from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
from pathlib import Path
from typing import Dict, List

from graphrag_plus.app.utils.io_utils import append_jsonl, dump_json, load_json
from graphrag_plus.app.utils.logging_utils import get_logger, log_event


class GraphVersionManager:
    """Maintain version snapshots and stale answer detection."""

    def __init__(self, versions_dir: Path, answers_log_path: Path):
        self.logger = get_logger(self.__class__.__name__)
        self.versions_dir = versions_dir
        self.answers_log_path = answers_log_path
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.versions_dir / "index.json"
        self.index = load_json(self.index_path, default={"versions": []})

    def create_version(self, snapshot: Dict[str, List[Dict[str, str]]], changed_nodes: List[str], changed_edges: List[str]) -> Dict[str, object]:
        """Persist snapshot and delta."""
        version_id = _utcnow().strftime("v%Y%m%d%H%M%S%f")
        version_path = self.versions_dir / f"{version_id}.json"
        dump_json(
            version_path,
            {
                "graph_version_id": version_id,
                "changed_nodes": changed_nodes,
                "changed_edges": changed_edges,
                "affected_subgraphs": changed_nodes[:50],
                "snapshot": snapshot,
            },
        )
        self.index["versions"].append({"graph_version_id": version_id, "path": str(version_path)})
        dump_json(self.index_path, self.index)
        log_event(
            self.logger,
            "graph_version_created",
            {"graph_version_id": version_id, "changed_nodes": len(changed_nodes), "changed_edges": len(changed_edges)},
        )
        return {"graph_version_id": version_id, "changed_nodes": changed_nodes, "changed_edges": changed_edges}

    def record_answer(self, answer_id: str, graph_version_id: str, supporting_nodes: List[str]) -> None:
        """Track answer lineage for drift checks."""
        append_jsonl(
            self.answers_log_path,
            [
                {
                    "answer_id": answer_id,
                    "graph_version_id": graph_version_id,
                    "supporting_nodes": supporting_nodes,
                    "timestamp": _utcnow().isoformat(),
                }
            ],
        )

    def detect_answer_state(self, supporting_nodes: List[str], changed_nodes: List[str]) -> str:
        """Mark answer stale or updated."""
        overlap = set(supporting_nodes).intersection(changed_nodes)
        if overlap:
            return "stale"
        return "updated"
