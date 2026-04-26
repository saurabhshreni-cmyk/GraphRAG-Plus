"""Evaluation and ablation runners."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from graphrag_plus.app.evaluation.benchmark import ensure_benchmark
from graphrag_plus.app.evaluation.metrics import aggregate_metrics
from graphrag_plus.app.utils.io_utils import dump_json


class EvaluationResult(TypedDict):
    """Output payload for ``evaluate_stub``."""

    metrics: dict[str, float]
    report_path: str


class AblationResult(TypedDict):
    """Output payload for ``run_ablation_matrix``."""

    rows: list[dict[str, object]]
    report_path: str


def evaluate_stub(reports_dir: Path, benchmark_path: Path) -> EvaluationResult:
    """Run lightweight synthetic evaluation."""
    benchmark = ensure_benchmark(benchmark_path)
    rows: list[dict[str, float]] = []
    for item in benchmark:
        difficulty = str(item.get("difficulty_level", ""))
        rows.append(
            {
                "precision_at_5": 0.8 if difficulty != "adversarial" else 0.6,
                "recall_at_5": 0.75 if difficulty != "adversarial" else 0.55,
                "multi_hop_accuracy": 0.8 if difficulty == "multi-hop" else 0.7,
                "temporal_correctness": 0.82 if difficulty == "temporal" else 0.7,
                "contradiction_resolution_accuracy": 0.85 if difficulty == "contradiction" else 0.7,
                "calibration_quality_ece": 0.08,
                "hallucination_rate": 0.1,
                "latency_ms": 45.0,
            }
        )
    metrics = aggregate_metrics(rows)
    report_path = reports_dir / f"research_eval_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.json"
    dump_json(report_path, {"metrics": metrics, "items": benchmark})
    return {"metrics": metrics, "report_path": str(report_path)}


def run_ablation_matrix(reports_dir: Path, base_metrics: dict[str, float]) -> AblationResult:
    """Run ablation combinations on synthetic deltas."""
    configs = [
        {"use_gnn": True, "use_calibration": True, "use_graph": True, "use_vector": True, "use_trust": True},
        {"use_gnn": False, "use_calibration": True, "use_graph": True, "use_vector": True, "use_trust": True},
        {"use_gnn": True, "use_calibration": False, "use_graph": True, "use_vector": True, "use_trust": True},
        {"use_gnn": True, "use_calibration": True, "use_graph": False, "use_vector": True, "use_trust": True},
        {"use_gnn": True, "use_calibration": True, "use_graph": True, "use_vector": False, "use_trust": True},
        {"use_gnn": True, "use_calibration": True, "use_graph": True, "use_vector": True, "use_trust": False},
    ]
    rows = []
    for config in configs:
        penalty = 0.0
        for _key, value in config.items():
            if not value:
                penalty += 0.03
        metrics = {key: max(0.0, float(value) - penalty) for key, value in base_metrics.items()}
        rows.append({"config": config, "metrics": metrics})
    report_path = reports_dir / f"ablation_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.json"
    dump_json(report_path, rows)
    return {"rows": rows, "report_path": str(report_path)}
