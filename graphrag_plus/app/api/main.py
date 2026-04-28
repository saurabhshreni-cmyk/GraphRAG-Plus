"""FastAPI application."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from graphrag_plus.app.config.settings import get_settings
from graphrag_plus.app.evaluation.runner import evaluate_stub
from graphrag_plus.app.pipeline import GraphRAGPipeline
from graphrag_plus.app.schemas.models import (
    EvalResult,
    GraphResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from graphrag_plus.app.utils.metrics import METRICS

settings = get_settings()
pipeline = GraphRAGPipeline(settings)
app = FastAPI(title="GraphRAG++")

# CORS — origins controlled by env (comma-separated). Defaults are local Vite dev servers.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_origins = [
    origin.strip()
    for origin in os.environ.get("GRAPHRAG_CORS_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check."""
    return HealthResponse(
        status="ok",
        llm_enabled=settings.llm_enabled,
        graph_exists=settings.graph_path.exists(),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Ingest files and URLs."""
    return pipeline.ingest(request.file_paths, request.urls)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run question answering."""
    return pipeline.query(request)


@app.get("/graph")
def graph_snapshot(limit: int = 500) -> dict[str, list[dict[str, Any]]]:
    """Return the full current graph (nodes + edges) for visualization.

    The ``limit`` caps each list defensively so the frontend never has to render
    a runaway graph.
    """
    snapshot = pipeline.graph_store.current_snapshot()
    return {
        "nodes": snapshot.get("nodes", [])[:limit],
        "edges": snapshot.get("edges", [])[:limit],
    }


@app.get("/graph/{node_id}", response_model=GraphResponse)
def graph(node_id: str) -> GraphResponse:
    """Return neighborhood for node."""
    neighbors = pipeline.graph_store.neighbors(node_id)
    if not neighbors:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found or has no neighbors.")
    return GraphResponse(node_id=node_id, neighbors=neighbors)


@app.get("/evaluate", response_model=EvalResult)
def evaluate() -> EvalResult:
    """Run benchmark evaluation."""
    result = evaluate_stub(settings.reports_dir, settings.data_dir / "benchmark.json")
    return EvalResult(metrics=result["metrics"], report_path=result["report_path"])


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus-compatible metrics."""
    body, content_type = METRICS.render()
    return Response(content=body, media_type=content_type)
