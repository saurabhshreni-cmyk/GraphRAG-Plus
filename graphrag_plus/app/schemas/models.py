"""Pydantic schemas for GraphRAG++."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """Typed failure modes."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_EVIDENCE = "NO_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    LLM_FAILURE = "LLM_FAILURE"


class IngestRequest(BaseModel):
    """Ingestion request."""

    file_paths: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


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
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


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

    query_id: Optional[str] = None
    answer: str
    confidence: float
    raw_confidence: float
    calibrated_confidence: float
    calibration_error: float
    used_llm: bool
    evidence: List[EvidenceItem] = Field(default_factory=list)
    evidence_paths: List[List[str]] = Field(default_factory=list)
    explanation: str
    conflicting_evidence: List[ContradictionItem] = Field(default_factory=list)
    resolution_explanation: Optional[str] = None
    failure_type: Optional[FailureType] = None
    mitigation_strategy_used: Optional[str] = None
    reasoning_steps: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    graph_version_id: Optional[str] = None
    answer_state: str = "updated"
    output_path: Optional[str] = None


class GraphResponse(BaseModel):
    """Graph API response."""

    node_id: str
    neighbors: List[Dict[str, Any]]


class EvalResult(BaseModel):
    """Evaluation report payload."""

    metrics: Dict[str, float]
    report_path: str


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    llm_enabled: bool
    graph_exists: bool
