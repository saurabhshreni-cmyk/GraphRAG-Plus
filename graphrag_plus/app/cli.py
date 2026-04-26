"""Command line interface."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from graphrag_plus.app.config.settings import get_settings
from graphrag_plus.app.utils.runtime import backend_status, enabled_modules


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="GraphRAG++ CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest documents")
    ingest.add_argument("--files", nargs="*", default=[])
    ingest.add_argument("--urls", nargs="*", default=[])

    build_graph = sub.add_parser("build-graph", help="Alias for ingest")
    build_graph.add_argument("--files", nargs="*", default=[])
    build_graph.add_argument("--urls", nargs="*", default=[])

    query = sub.add_parser("query", help="Query system")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--analyst-mode", action="store_true")

    sub.add_parser("evaluate", help="Run evaluation")
    export = sub.add_parser("export-graph", help="Export graph as GraphML")
    export.add_argument("--path", default="graphrag_plus/data/reports/graph.graphml")

    sub.add_parser("run_ablation", help="Run ablation matrix")
    sub.add_parser("health_check", help="Self-check dependencies and backends")
    return parser


def _warn_if_not_in_venv() -> None:
    in_venv = bool(os.environ.get("VIRTUAL_ENV")) or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    if not in_venv:
        print(
            "WARNING: You are not using a virtual environment. This may cause permission issues.",
            file=sys.stderr,
        )


def _package_version() -> str:
    try:
        return importlib.metadata.version("graphrag-plus")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0-dev"


def _print_system_status() -> None:
    try:
        settings = get_settings()
    except Exception as exc:
        print(f"GraphRAG++ | version={_package_version()} | settings_error={exc}", file=sys.stderr)
        return
    print(f"GraphRAG++ | version={_package_version()}", file=sys.stderr)
    print(f"Enabled modules: {', '.join(enabled_modules(settings)) or 'none'}", file=sys.stderr)
    print(
        "Backends: " + ", ".join(f"{name}={value}" for name, value in backend_status(settings).items()),
        file=sys.stderr,
    )


def _health_check() -> dict:
    issues: list[str] = []
    status = "healthy"

    # Import checks (report missing deps instead of crashing).
    try:
        from graphrag_plus.app.config.settings import get_settings  # noqa: PLC0415
    except Exception as exc:
        issues.append(f"settings_import_error: {exc}")
        status = "degraded"
        return {"status": status, "issues": issues}

    settings = None
    try:
        settings = get_settings()
    except Exception as exc:
        issues.append(f"settings_validation_error: {exc}")

    try:
        from graphrag_plus.app.pipeline import GraphRAGPipeline  # noqa: PLC0415
    except Exception as exc:
        issues.append(f"pipeline_import_error: {exc}")
        status = "degraded"
        return {"status": status, "issues": issues}

    try:
        pipeline = GraphRAGPipeline(settings)  # type: ignore[arg-type]
    except Exception as exc:
        issues.append(f"pipeline_init_error: {exc}")
        status = "degraded"
        return {"status": status, "issues": issues}

    # Graph backend
    try:
        _ = pipeline.graph_store.graph.number_of_nodes()
    except Exception as exc:
        issues.append(f"graph_backend_error: {exc}")

    # Vector index readiness (degraded if no ingest yet)
    try:
        if pipeline.retrieval.chunk_matrix is None:
            issues.append("vector_index_not_built: run ingest first")
    except Exception as exc:
        issues.append(f"vector_index_error: {exc}")

    # Scoring module
    try:
        _ = pipeline.scoring.weights
    except Exception as exc:
        issues.append(f"scoring_module_error: {exc}")

    # Calibration module
    try:
        _ = pipeline.calibration.temperature
    except Exception as exc:
        issues.append(f"calibration_module_error: {exc}")

    # Trust module
    try:
        _ = pipeline.trust_manager.get_trust_score("health_probe")
    except Exception as exc:
        issues.append(f"trust_module_error: {exc}")

    if issues:
        status = "degraded"

    return {"status": status, "issues": issues}


def main() -> None:
    """CLI entry point."""
    _warn_if_not_in_venv()
    _print_system_status()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "health_check":
        print(json.dumps(_health_check(), indent=2))
        return

    # Lazy imports so `health_check` can run even when dependencies are missing.
    from graphrag_plus.app.evaluation.runner import (  # noqa: PLC0415
        evaluate_stub,
        run_ablation_matrix,
    )
    from graphrag_plus.app.pipeline import GraphRAGPipeline  # noqa: PLC0415
    from graphrag_plus.app.schemas.models import QueryRequest  # noqa: PLC0415

    settings = get_settings()
    pipeline = GraphRAGPipeline(settings)

    if args.command in {"ingest", "build-graph"}:
        ingest_result = pipeline.ingest(args.files, args.urls)
        print(ingest_result.model_dump_json(indent=2))
        return
    if args.command == "query":
        response = pipeline.query(
            QueryRequest(question=args.question, top_k=args.top_k, analyst_mode=args.analyst_mode)
        )
        print(response.model_dump_json(indent=2))
        return
    if args.command == "evaluate":
        eval_result = evaluate_stub(settings.reports_dir, settings.data_dir / "benchmark.json")
        print(json.dumps(eval_result, indent=2))
        return
    if args.command == "export-graph":
        target_path = Path(args.path)
        pipeline.graph_store.export_graphml(target_path)
        print(json.dumps({"exported_to": str(target_path)}, indent=2))
        return
    if args.command == "run_ablation":
        evaluation = evaluate_stub(settings.reports_dir, settings.data_dir / "benchmark.json")
        ablation_result = run_ablation_matrix(settings.reports_dir, evaluation["metrics"])
        print(json.dumps(ablation_result, indent=2))
        return


if __name__ == "__main__":
    main()
