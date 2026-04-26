"""Graph versioning tests."""

from pathlib import Path

from graphrag_plus.app.graph.versioning.manager import GraphVersionManager


def test_version_creation_and_stale_detection(tmp_path: Path) -> None:
    manager = GraphVersionManager(tmp_path / "versions", tmp_path / "answers.jsonl")
    version = manager.create_version(
        snapshot={"nodes": [], "edges": []},
        changed_nodes=["n1"],
        changed_edges=["n1->n2:edge"],
    )
    assert version["graph_version_id"]
    assert manager.detect_answer_state(["n1"], ["n1"]) == "stale"
    assert manager.detect_answer_state(["n2"], ["n1"]) == "updated"
