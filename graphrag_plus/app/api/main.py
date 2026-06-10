"""FastAPI application."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from graphrag_plus.app.config.settings import get_settings
from graphrag_plus.app.evaluation.runner import evaluate_stub
from graphrag_plus.app.pipeline import GraphRAGPipeline
from graphrag_plus.app.schemas.models import (
    CorpusInfo,
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

# Edge filtering for the "important" graph view.
#   ``mentions`` is kept — it's the chunk→entity backbone, removing it leaves
#   entity nodes floating with no connections in the visualization.
#   Only drop edges that have an explicit confidence below the floor.
_LOW_VALUE_EDGES: set[str] = set()
_MIN_EDGE_CONFIDENCE = 0.6

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


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a CORS-friendly JSON 500 for any unhandled error.

    Starlette's default ServerErrorMiddleware sits *outside* CORSMiddleware,
    so a bare unhandled exception yields a 500 with no ``Access-Control-
    Allow-Origin`` header — browsers then surface an opaque "Failed to fetch"
    instead of the real error. Registering this handler keeps the response
    inside the CORS layer so the frontend can read and display the message.
    """
    pipeline.logger.exception("api.unhandled_error path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {type(exc).__name__}"},
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
    """Ingest files and URLs into a (new or existing) corpus."""
    return pipeline.ingest(
        request.file_paths,
        request.urls,
        corpus_id=request.corpus_id,
        new_corpus=request.new_corpus,
        corpus_name=request.corpus_name,
    )


@app.get("/corpora", response_model=list[CorpusInfo])
def list_corpora() -> list[CorpusInfo]:
    """List all known corpora (newest first)."""
    return [CorpusInfo(**asdict(meta)) for meta in pipeline.corpus_manager.list()]


@app.get("/corpora/active", response_model=CorpusInfo | None)
def active_corpus() -> CorpusInfo | None:
    """Return the currently active corpus, if any."""
    bundle = pipeline.corpus_manager.get_active()
    return CorpusInfo(**asdict(bundle.meta)) if bundle else None


@app.post("/corpora/{corpus_id}/select", response_model=CorpusInfo)
def select_corpus(corpus_id: str) -> CorpusInfo:
    """Switch the pipeline's active corpus to ``corpus_id``."""
    try:
        bundle = pipeline.corpus_manager.set_active(corpus_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CorpusInfo(**asdict(bundle.meta))


@app.delete("/corpora/{corpus_id}")
def delete_corpus(corpus_id: str) -> dict[str, str]:
    """Permanently delete a corpus (its directory + cached state)."""
    try:
        pipeline.corpus_manager.delete(corpus_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "deleted", "corpus_id": corpus_id}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run question answering."""
    return pipeline.query(request)


@app.get("/graph")
def graph_snapshot(
    mode: str = "important",
    limit: int = 60,
    full_limit: int = 500,
    corpus_id: str | None = None,
    max_edges_per_node: int = 8,
    min_co_occurs: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Return a corpus's graph for visualization.

    * ``corpus_id`` — which isolated corpus to read; defaults to the active
      corpus. Each corpus's graph is fully isolated from the others.
    * ``mode=important`` (default) — concept-centric view: top ``limit``
      entity nodes ranked by degree, with per-node edge cap and a
      co-occurrence frequency floor to keep dense graphs readable.
    * ``mode=full`` — raw graph snapshot up to ``full_limit``.
    * ``max_edges_per_node`` — only the strongest N edges per node survive
      in IMPORTANT view.
    * ``min_co_occurs`` — co-occurrence edges with weight below this are
      dropped (still meaningful relations like is_a / supports stay).
    """
    if corpus_id:
        try:
            bundle = pipeline.corpus_manager.get(corpus_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        bundle = pipeline.corpus_manager.get_active() or pipeline._resolve_corpus(None)
    snapshot = bundle.graph_store.current_snapshot()
    nodes: list[dict[str, Any]] = snapshot.get("nodes", [])
    edges: list[dict[str, Any]] = snapshot.get("edges", [])

    if mode == "full":
        return {"nodes": nodes[:full_limit], "edges": edges[:full_limit]}

    # ---- "important" mode: filters for a readable concept graph ------------
    # Filter out: chunk-mention noise, low-confidence edges, weak co-occurrence
    # pairs (weight < min_co_occurs), and generic doc→chunk structural edges.

    def _edge_weight(edge: dict[str, Any]) -> float:
        """Strength used to rank edges. co_occurs uses ``weight``;
        meaningful relations use ``confidence``; everything else gets 0.5."""
        if edge.get("edge_type") == "co_occurs":
            return float(edge.get("weight", 1))
        conf = edge.get("confidence")
        if isinstance(conf, (int | float)):
            return float(conf)
        return 0.5

    def _keep_edge(edge: dict[str, Any]) -> bool:
        et = edge.get("edge_type")
        # Drop the chunk → entity boilerplate and doc → chunk backbone — this
        # view is concept-centric.
        if et in {"mentions", "contains"}:
            return False
        # Drop weak co-occurrence pairs that are below the frequency floor.
        if et == "co_occurs" and float(edge.get("weight", 0)) < min_co_occurs:
            return False
        conf = edge.get("confidence")
        return not (isinstance(conf, (int | float)) and conf < _MIN_EDGE_CONFIDENCE)

    filtered_edges = [e for e in edges if _keep_edge(e)]

    degree: dict[str, int] = {}
    for edge in filtered_edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    # Concept-centric view: drop doc + chunk nodes entirely. The graph
    # answers "what concepts in this corpus relate to each other?".
    entity_nodes = [n for n in nodes if n.get("node_type") not in {"Document", "Chunk"}]
    entity_nodes.sort(key=lambda n: degree.get(n["id"], 0), reverse=True)

    keep_entities = entity_nodes[: max(1, limit)]
    keep_ids = {n["id"] for n in keep_entities}

    pruned_nodes: list[dict[str, Any]] = [dict(n) for n in nodes if n["id"] in keep_ids]
    candidate_edges = [e for e in filtered_edges if e["source"] in keep_ids and e["target"] in keep_ids]

    # Per-node edge cap: keep at most ``max_edges_per_node`` edges
    # incident to each node, picking the strongest ones first. Two-sided
    # cap — both endpoints must still have budget.
    candidate_edges.sort(key=_edge_weight, reverse=True)
    edge_count: dict[str, int] = {}
    capped_edges: list[dict[str, Any]] = []
    for edge in candidate_edges:
        s, t = edge["source"], edge["target"]
        if edge_count.get(s, 0) >= max_edges_per_node or edge_count.get(t, 0) >= max_edges_per_node:
            continue
        capped_edges.append(edge)
        edge_count[s] = edge_count.get(s, 0) + 1
        edge_count[t] = edge_count.get(t, 0) + 1

    # Annotate each node with its degree (after capping) so the frontend
    # can size by importance.
    final_degree: dict[str, int] = {}
    for edge in capped_edges:
        final_degree[edge["source"]] = final_degree.get(edge["source"], 0) + 1
        final_degree[edge["target"]] = final_degree.get(edge["target"], 0) + 1
    for n in pruned_nodes:
        n["degree"] = final_degree.get(n["id"], 0)

    return {"nodes": pruned_nodes, "edges": capped_edges}


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
