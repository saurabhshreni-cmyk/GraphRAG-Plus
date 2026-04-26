"""Ablation tests."""

from pathlib import Path

from graphrag_plus.app.evaluation.runner import run_ablation_matrix


def test_ablation_output_schema(tmp_path: Path) -> None:
    result = run_ablation_matrix(
        reports_dir=tmp_path,
        base_metrics={"precision_at_5": 0.8, "recall_at_5": 0.7},
    )
    assert "rows" in result
    assert "report_path" in result
    assert len(result["rows"]) >= 1

