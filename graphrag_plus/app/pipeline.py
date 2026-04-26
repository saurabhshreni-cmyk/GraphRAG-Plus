"""Main GraphRAG++ orchestration."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Set, Tuple, TypeVar

from graphrag_plus.app.active_learning.manager import ActiveLearningManager
from graphrag_plus.app.analyst.engine import AnalystEngine
from graphrag_plus.app.calibration.module import CalibrationModule
from graphrag_plus.app.config.settings import Settings
from graphrag_plus.app.contradiction.reasoner import ContradictionReasoner
from graphrag_plus.app.extraction.extractor import extract_from_chunks
from graphrag_plus.app.failure.handler import FailureModeHandler
from graphrag_plus.app.generation.generator import AnswerGenerator
from graphrag_plus.app.gnn.scorer import GNNScorer
from graphrag_plus.app.graph.store import GraphStore
from graphrag_plus.app.graph.versioning.manager import GraphVersionManager
from graphrag_plus.app.ingestion.chunker import chunk_documents
from graphrag_plus.app.ingestion.loader import load_documents
from graphrag_plus.app.planning.query_planner import plan_query
from graphrag_plus.app.retrieval.service import RetrievalService
from graphrag_plus.app.schemas.models import (
    ContradictionItem,
    EvidenceItem,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from graphrag_plus.app.scoring.module import ScoringModule
from graphrag_plus.app.trust.manager import SourceTrustManager
from graphrag_plus.app.utils.logging_utils import get_logger, log_event
from graphrag_plus.app.utils.metrics import METRICS
from graphrag_plus.app.utils.run_logger import utc_now_iso, write_query_output, write_run_log
from graphrag_plus.app.utils.runtime import apply_global_seed

T = TypeVar("T")


class GraphRAGPipeline:
    """End-to-end pipeline service."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        apply_global_seed(settings.random_seed)
        self.graph_store = GraphStore(settings.graph_path)
        self.version_manager = GraphVersionManager(
            settings.graph_versions_dir, settings.answers_log_path
        )
        self.retrieval = RetrievalService(self.graph_store)
        self.trust_manager = SourceTrustManager(
            settings.trust_state_path,
            settings.default_trust_prior,
            settings.source_trust_priors,
        )
        self.scoring = ScoringModule(
            {
                "w1": settings.scoring_w1_semantic,
                "w2": settings.scoring_w2_graph,
                "w3": settings.scoring_w3_confidence,
                "w4": settings.scoring_w4_trust,
                "w5": settings.scoring_w5_uncertainty_penalty,
            }
        )
        self.calibration = CalibrationModule(settings.calibration_state_path)
        self.reasoner = ContradictionReasoner()
        self.failure_handler = FailureModeHandler()
        self.active_learning = ActiveLearningManager(settings.review_queue_path)
        self.analyst = AnalystEngine()
        self.generator = AnswerGenerator(settings.llm_enabled)
        self.gnn = GNNScorer()
        self.latest_changed_nodes: List[str] = []
        self.latest_graph_version: str | None = None
        # Persisted contradiction signal: chunk_ids that participated in a
        # contradiction during ingestion. Used by the query path to surface
        # actually-conflicting evidence rather than substring guesses.
        self._chunk_contradictions: Dict[str, List[ContradictionItem]] = {}

    # ------------------------------------------------------------------ utils
    def _safe(self, stage: str, fn: Callable[[], T], fallback: T) -> T:
        """Run ``fn`` and return ``fallback`` if it raises, with structured logging."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we deliberately catch all
            self.logger.exception("stage_failed=%s", stage)
            log_event(self.logger, "stage_failed", {"stage": stage, "error": str(exc)})
            METRICS.errors_total.labels(stage=stage).inc()
            return fallback

    @staticmethod
    def _ms_since(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)

    # --------------------------------------------------------------- ingestion
    def ingest(self, file_paths: List[str], urls: List[str]) -> IngestResponse:
        """Ingest and index documents."""
        ingestion_start = time.perf_counter()
        timings: Dict[str, float] = {}

        load_start = time.perf_counter()
        documents = self._safe(
            "ingestion.load_documents",
            lambda: load_documents(file_paths=file_paths, urls=urls),
            [],
        )
        timings["load_ms"] = self._ms_since(load_start)

        chunk_start = time.perf_counter()
        chunks = self._safe(
            "ingestion.chunk_documents",
            lambda: chunk_documents(
                documents, self.settings.chunk_size, self.settings.chunk_overlap
            ),
            [],
        )
        timings["chunk_ms"] = self._ms_since(chunk_start)

        extract_start = time.perf_counter()
        entities, relations = self._safe(
            "extraction.extract_from_chunks",
            lambda: extract_from_chunks(chunks),
            ([], []),
        )
        timings["extract_ms"] = self._ms_since(extract_start)

        contradictions: List[ContradictionItem] = []
        if self.settings.enable_contradiction:
            contradiction_start = time.perf_counter()
            relations, contradictions = self._safe(
                "contradiction.detect",
                lambda: self.reasoner.detect(relations),
                (relations, []),
            )
            timings["contradiction_ms"] = self._ms_since(contradiction_start)

        graph_start = time.perf_counter()
        changed_nodes, changed_edges = self._safe(
            "graph.upsert",
            lambda: self.graph_store.upsert_from_extractions(
                documents, chunks, entities, relations
            ),
            ([], []),
        )
        timings["graph_upsert_ms"] = self._ms_since(graph_start)

        version_info = self._safe(
            "graph.versioning.create",
            lambda: self.version_manager.create_version(
                snapshot=self.graph_store.current_snapshot(),
                changed_nodes=changed_nodes,
                changed_edges=changed_edges,
            ),
            {"graph_version_id": "error", "changed_nodes": [], "changed_edges": []},
        )
        self.latest_changed_nodes = changed_nodes
        self.latest_graph_version = str(version_info.get("graph_version_id"))

        index_start = time.perf_counter()
        self._safe("retrieval.build_indexes", lambda: self.retrieval.build_indexes(chunks), None)
        timings["index_ms"] = self._ms_since(index_start)

        # Fan contradiction signal into trust + per-chunk lookup.
        self._record_contradictions(contradictions)

        timings["total_ms"] = self._ms_since(ingestion_start)
        METRICS.ingest_total.inc()
        METRICS.ingest_documents.inc(len(documents))
        METRICS.observe_modules(
            (name, ms) for name, ms in timings.items() if name != "total_ms"
        )
        log_event(
            self.logger,
            "ingest_complete",
            {
                "documents": len(documents),
                "chunks": len(chunks),
                "entities": len(entities),
                "relations": len(relations),
                "contradictions": len(contradictions),
                "graph_version_id": self.latest_graph_version,
                "module_timings": timings,
            },
        )

        return IngestResponse(
            documents=len(documents),
            chunks=len(chunks),
            entities=len(entities),
            relations=len(relations),
            graph_version_id=str(version_info.get("graph_version_id", "error")),
        )

    def _record_contradictions(self, contradictions: List[ContradictionItem]) -> None:
        """Persist contradictions for query-time consumption + update trust."""
        for item in contradictions:
            for chunk_id in (item.source_a, item.source_b):
                self._chunk_contradictions.setdefault(chunk_id, []).append(item)

            # Bind locals explicitly so each safe call captures the right value.
            source_a = item.source_a
            source_b = item.source_b
            self._safe(
                "trust.update_a",
                lambda sa=source_a: self.trust_manager.update(
                    sa, agrees=False, is_correct=False, low_confidence=True
                ),
                None,
            )
            self._safe(
                "trust.update_b",
                lambda sb=source_b: self.trust_manager.update(
                    sb, agrees=False, is_correct=False, low_confidence=True
                ),
                None,
            )

    # ------------------------------------------------------------------ query
    def query(self, request: QueryRequest) -> QueryResponse:
        """Run planned retrieval and generation."""
        query_id = f"qry_{uuid.uuid4().hex[:12]}"
        started_at = utc_now_iso()
        query_start = time.perf_counter()
        module_timings: Dict[str, float] = {
            "planning_ms": 0.0,
            "retrieval_ms": 0.0,
            "scoring_ms": 0.0,
            "generation_ms": 0.0,
        }

        plan_start = time.perf_counter()
        plan = self._safe("planning.plan_query", lambda: plan_query(request.question), None)
        module_timings["planning_ms"] = self._ms_since(plan_start)
        _ = plan  # reserved for future routing policy

        trust_lookup = self._build_trust_lookup()

        retrieval_start = time.perf_counter()
        candidates = self._safe(
            "retrieval.query",
            lambda: self.retrieval.query(request.question, request.top_k, trust_lookup),
            [],
        )
        module_timings["retrieval_ms"] = self._ms_since(retrieval_start)

        if self.settings.use_gnn and candidates:
            gnn_scores = self._safe("gnn.score", lambda: self.gnn.score(candidates), [])
            if gnn_scores and len(gnn_scores) == len(candidates):
                for idx, gnn_score in enumerate(gnn_scores):
                    candidates[idx]["graph_score"] = (
                        0.5 * candidates[idx]["graph_score"] + 0.5 * gnn_score
                    )

        scoring_start = time.perf_counter()
        scored = self._safe(
            "scoring.score_candidates", lambda: self.scoring.score_candidates(candidates), []
        )
        module_timings["scoring_ms"] = self._ms_since(scoring_start)

        top = scored[: request.top_k] if scored else []
        raw_confidence = (
            float(sum(item["confidence_score"] for item in top) / len(top)) if top else 0.0
        )

        calibrated_confidence, calibration_error = self._calibrate(raw_confidence)

        evidence_items = [self._evidence_from(item) for item in top]
        evidence_paths = [[e.source_id, "supports", e.id] for e in evidence_items]

        # Real contradiction detection: an evidence chunk is conflicting iff the
        # ingestion-time reasoner flagged it.
        conflicting, resolution_explanation, has_conflict = self._collect_conflicts(
            evidence_items, request.question
        )

        generation_start = time.perf_counter()
        answer_text, used_llm, llm_failed = self._safe(
            "generation.generate",
            lambda: self.generator.generate(
                request.question,
                [item.model_dump() for item in evidence_items],
                calibrated_confidence,
                self.settings.answer_threshold,
            ),
            ("I cannot answer reliably due to an internal error.", False, True),
        )
        module_timings["generation_ms"] = self._ms_since(generation_start)

        failure = self.failure_handler.classify(
            has_evidence=bool(evidence_items),
            confidence=calibrated_confidence,
            uncertainty=(1.0 - calibrated_confidence),
            has_conflict=has_conflict,
            llm_failed=llm_failed,
            confidence_threshold=self.settings.answer_threshold,
            high_uncertainty_threshold=self.settings.high_uncertainty_threshold,
        )
        if failure["failure_type"] == "LOW_CONFIDENCE":
            answer_text = f"Partial answer: {answer_text}"
        if failure["failure_type"] == "NO_EVIDENCE":
            answer_text = "I must abstain because retrieval produced no evidence."

        if self.settings.enable_active_learning and (
            calibrated_confidence < self.settings.answer_threshold or has_conflict
        ):
            self._safe(
                "active_learning.process",
                lambda: self.active_learning.process_cases(
                    [
                        {
                            "question": request.question,
                            "confidence": calibrated_confidence,
                            "has_conflict": has_conflict,
                        }
                    ]
                ),
                [],
            )

        answer_id = f"ans_{uuid.uuid4().hex[:12]}"
        supporting_nodes = [item.id for item in evidence_items]
        graph_version_id = self.latest_graph_version or "unknown"
        self._safe(
            "versioning.record_answer",
            lambda: self.version_manager.record_answer(
                answer_id, graph_version_id, supporting_nodes
            ),
            None,
        )
        answer_state = self._safe(
            "versioning.detect_answer_state",
            lambda: self.version_manager.detect_answer_state(
                supporting_nodes, self.latest_changed_nodes
            ),
            "updated",
        )

        reasoning_steps: List[str] = []
        follow_ups: List[str] = []
        if request.analyst_mode or self.settings.analyst_mode_default:
            analyst_conflicts = [item.model_dump() for item in conflicting]
            analyst = self._safe(
                "analyst.build",
                lambda: self.analyst.build(
                    request.question, evidence_paths, analyst_conflicts, calibrated_confidence
                ),
                {"reasoning_steps": [], "follow_up_questions": []},
            )
            reasoning_steps = list(analyst.get("reasoning_steps", []))
            follow_ups = list(analyst.get("follow_up_questions", []))

        log_event(
            self.logger,
            "query_complete",
            {
                "query_id": query_id,
                "question": request.question,
                "raw_confidence": raw_confidence,
                "calibrated_confidence": calibrated_confidence,
                "failure_type": failure["failure_type"],
                "graph_version_id": graph_version_id,
                "module_timings": module_timings,
            },
        )

        latency_ms = self._ms_since(query_start)
        failure_label = str(failure["failure_type"]) if failure["failure_type"] else "ok"
        METRICS.queries_total.labels(failure_type=failure_label).inc()
        METRICS.query_latency.observe(latency_ms / 1000.0)
        METRICS.observe_modules(module_timings.items())
        response = QueryResponse(
            query_id=query_id,
            answer=answer_text,
            confidence=calibrated_confidence,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            calibration_error=calibration_error,
            used_llm=used_llm,
            evidence=evidence_items,
            evidence_paths=evidence_paths,
            explanation=failure["explanation"] or "Answer generated from top ranked evidence.",
            conflicting_evidence=conflicting,
            resolution_explanation=resolution_explanation,
            failure_type=failure["failure_type"],
            mitigation_strategy_used=failure["mitigation"],
            reasoning_steps=reasoning_steps,
            follow_up_questions=follow_ups,
            graph_version_id=graph_version_id,
            answer_state=answer_state,
        )

        output_payload = {
            "query_id": query_id,
            "question": request.question,
            "answer": response.answer,
            "confidence": response.confidence,
            "evidence": [item.model_dump() for item in response.evidence],
            "explanation": response.explanation,
            "failure_type": response.failure_type,
            "graph_version_id": response.graph_version_id,
        }
        output_path = write_query_output(self.settings.outputs_dir, query_id, output_payload)
        response.output_path = str(output_path)
        write_run_log(
            self.settings.run_logs_path,
            {
                "query_id": query_id,
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "latency_ms": latency_ms,
                "module_timings": module_timings,
                "question": request.question,
                "graph_version_id": graph_version_id,
                "used_llm": used_llm,
                "failure_type": str(response.failure_type) if response.failure_type else None,
            },
        )
        return response

    # ----------------------------------------------------------- query helpers
    def _build_trust_lookup(self) -> Dict[str, float]:
        lookup: Dict[str, float] = {}
        try:
            for node_id, attrs in self.graph_store.graph.nodes(data=True):
                if attrs.get("node_type") == "Document":
                    lookup[node_id] = self.trust_manager.get_trust_score(node_id)
        except Exception as exc:  # noqa: BLE001
            log_event(self.logger, "trust_lookup_failed", {"error": str(exc)})
        return lookup

    def _calibrate(self, raw_confidence: float) -> Tuple[float, float]:
        if not (self.settings.enable_calibration and self.settings.use_calibration):
            return raw_confidence, 0.0
        cal = self._safe(
            "calibration.calibrate",
            lambda: self.calibration.calibrate(raw_confidence),
            None,
        )
        if cal is None:
            return raw_confidence, 0.0
        return cal.calibrated_confidence, cal.calibration_error

    @staticmethod
    def _evidence_from(item: Dict[str, Any]) -> EvidenceItem:
        return EvidenceItem(
            id=item["id"],
            source_id=item["source_id"],
            snippet=item["snippet"],
            semantic_score=float(item["semantic_score"]),
            graph_score=float(item["graph_score"]),
            confidence_score=float(item["confidence_score"]),
            trust_score=float(item["trust_score"]),
            uncertainty_penalty=float(item["uncertainty_penalty"]),
            final_score=float(item["final_score"]),
        )

    def _collect_conflicts(
        self, evidence_items: List[EvidenceItem], question: str
    ) -> Tuple[List[ContradictionItem], str | None, bool]:
        """Return contradictions whose chunk ids are present in current evidence."""
        if not evidence_items or not self._chunk_contradictions:
            return [], None, False

        evidence_ids: Set[str] = {e.id for e in evidence_items}
        seen: Set[Tuple[str, str, str]] = set()
        conflicting: List[ContradictionItem] = []

        for evidence in evidence_items:
            for item in self._chunk_contradictions.get(evidence.id, []):
                if item.source_a not in evidence_ids and item.source_b not in evidence_ids:
                    continue
                key = (item.source_a, item.source_b, item.claim)
                if key in seen:
                    continue
                seen.add(key)
                conflicting.append(item)

        if not conflicting:
            return [], None, False

        resolution = (
            "Source ranking favored higher trust and confidence evidence "
            "during contradiction resolution."
        )
        # Provide question for downstream analyst context but keep ContradictionItem schema stable.
        _ = question
        return conflicting, resolution, True
