"""Pydantic schemas for GraphRAG++."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FailureType(StrEnum):
    """Typed failure modes."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_EVIDENCE = "NO_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    LLM_FAILURE = "LLM_FAILURE"


class IngestRequest(BaseModel):
    """Ingestion request."""

    file_paths: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    """Ingestion response."""

    documents: int
    chunks: int
    entities: int
    relations: int
    graph_version_id: str


class QueryRequest(BaseModel):
    """Query request."""

    question: str
    top_k: int = 5
    analyst_mode: bool = False
    start_time: datetime | None = None
    end_time: datetime | None = None
    # Per-request override for the LLM path. ``None`` means "use the
    # server-wide ``settings.llm_enabled``"; an explicit ``True`` / ``False``
    # forces the strategy for this query only. Keeps trust-aware gating
    # intact: ``NO_EVIDENCE`` still skips the LLM regardless of this flag.
    llm_enabled: bool | None = None


class EvidenceItem(BaseModel):
    """Evidence result."""

    id: str
    source_id: str
    snippet: str
    semantic_score: float
    graph_score: float
    confidence_score: float
    trust_score: float
    uncertainty_penalty: float
    final_score: float


class ContradictionItem(BaseModel):
    """Contradiction payload."""

    source_a: str
    source_b: str
    claim: str
    explanation: str


class QueryResponse(BaseModel):
    """Query response."""

    query_id: str | None = None
    answer: str
    confidence: float
    raw_confidence: float
    calibrated_confidence: float
    calibration_error: float
    used_llm: bool
    generated_by: str = "extractive"
    evidence: list[EvidenceItem] = Field(default_factory=list)
    evidence_paths: list[list[str]] = Field(default_factory=list)
    explanation: str
    conflicting_evidence: list[ContradictionItem] = Field(default_factory=list)
    resolution_explanation: str | None = None
    failure_type: FailureType | None = None
    mitigation_strategy_used: str | None = None
    reasoning_steps: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    graph_version_id: str | None = None
    answer_state: str = "updated"
    output_path: str | None = None


class GraphResponse(BaseModel):
    """Graph API response."""

    node_id: str
    neighbors: list[dict[str, Any]]


class EvalResult(BaseModel):
    """Evaluation report payload."""

    metrics: dict[str, float]
    report_path: str


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    llm_enabled: bool
    graph_exists: bool
