"""Main GraphRAG++ orchestration."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from graphrag_plus.app.active_learning.manager import ActiveLearningManager
from graphrag_plus.app.analyst.engine import AnalystEngine
from graphrag_plus.app.calibration.module import CalibrationModule
from graphrag_plus.app.config.settings import Settings
from graphrag_plus.app.contradiction.reasoner import ContradictionReasoner
from graphrag_plus.app.corpus.blob_store import make_blob_store
from graphrag_plus.app.corpus.manager import CorpusManager, derive_corpus_name
from graphrag_plus.app.corpus.models import CorpusBundle
from graphrag_plus.app.domain.detector import detect_domain
from graphrag_plus.app.extraction.extractor import extract_from_chunks
from graphrag_plus.app.failure.handler import FailureModeHandler
from graphrag_plus.app.generation.generator import AnswerGenerator
from graphrag_plus.app.generation.llm_clients import build_default_llm_client
from graphrag_plus.app.gnn.scorer import GNNScorer
from graphrag_plus.app.graph.versioning.manager import GraphVersionManager
from graphrag_plus.app.ingestion.chunker import chunk_documents
from graphrag_plus.app.ingestion.loader import load_documents
from graphrag_plus.app.planning.intent import (
    QueryIntent,
    adaptive_top_k,
    comparison_terms,
    detect_intent_signal,
)
from graphrag_plus.app.planning.query_planner import plan_query
from graphrag_plus.app.schemas.models import (
    ContradictionItem,
    EvidenceItem,
    FailureType,
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


class GraphRAGPipeline:
    """End-to-end pipeline service."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        apply_global_seed(settings.random_seed)

        # Multi-corpus support: each ingestion gets its own isolated bundle of
        # (graph store + retrieval service + metadata). The "default" corpus
        # below is created on demand to preserve backward compatibility for
        # tests / ad-hoc usage that don't pass a corpus_id.
        self.corpus_manager = CorpusManager(
            settings.corpora_dir, blob_store=make_blob_store(settings.database_url)
        )
        # Keep the legacy single-graph attributes for tests that touch
        # ``pipeline.graph_store`` / ``pipeline.retrieval`` directly. They
        # point to a "default" corpus whose data is also under corpora_dir.
        default = self.corpus_manager.get_active()
        if default is None:
            default = self.corpus_manager.create(
                name="default", domain="general", source_urls=[], source_files=[]
            )
        self._default_corpus_id = default.meta.corpus_id
        self.graph_store = default.graph_store
        self.retrieval = default.retrieval

        self.version_manager = GraphVersionManager(settings.graph_versions_dir, settings.answers_log_path)
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
        self.generator = AnswerGenerator(
            settings.llm_enabled,
            build_default_llm_client(llm_enabled=settings.llm_enabled),
        )
        self.gnn = GNNScorer()
        self.latest_changed_nodes: list[str] = []
        self.latest_graph_version: str | None = None
        # Persisted contradiction signal: chunk_ids that participated in a
        # contradiction during ingestion. Used by the query path to surface
        # actually-conflicting evidence rather than substring guesses.
        self._chunk_contradictions: dict[str, list[ContradictionItem]] = {}

    # ------------------------------------------------------------------ utils
    def _safe(self, stage: str, fn: Callable[[], Any], fallback: Any) -> Any:
        """Run ``fn`` and return ``fallback`` if it raises, with structured logging.

        Typed loosely as ``Any`` because callers pass empty containers as fallbacks
        whose types mypy infers as ``list[Never]`` / ``tuple[list[Never], ...]``.
        Concrete types are recovered at the call site.
        """
        try:
            return fn()
        except Exception as exc:
            self.logger.exception("stage_failed=%s", stage)
            log_event(self.logger, "stage_failed", {"stage": stage, "error": str(exc)})
            METRICS.errors_total.labels(stage=stage).inc()
            return fallback

    @staticmethod
    def _ms_since(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)

    # ------------------------------------------------------------ corpus utils
    def _resolve_corpus(self, corpus_id: str | None) -> CorpusBundle:
        """Pick the corpus bundle to operate on for a request.

        Falls back to the active corpus when ``corpus_id`` is None — tests
        and old API clients keep working without code changes.

        If an explicit ``corpus_id`` is given but unknown *on this process*
        (e.g. a serverless instance whose ephemeral ``/tmp`` never saw that
        ingest), we self-heal by falling back to the active/most-recent
        corpus instead of raising — the response echoes the corpus actually
        used so the client can re-sync. This keeps the public demo answering
        rather than returning an uncatchable 500.
        """
        if corpus_id:
            try:
                return self.corpus_manager.get(corpus_id)
            except KeyError:
                self.logger.warning(
                    "corpus.unknown_id id=%s — falling back to active corpus",
                    corpus_id,
                )
        active = self.corpus_manager.get_active()
        if active is None:  # pragma: no cover — bootstrap covers this
            active = self.corpus_manager.create(
                name="default", domain="general", source_urls=[], source_files=[]
            )
        return active

    # --------------------------------------------------------------- ingestion
    def ingest(
        self,
        file_paths: list[str],
        urls: list[str],
        *,
        corpus_id: str | None = None,
        new_corpus: bool = True,
        corpus_name: str | None = None,
    ) -> IngestResponse:
        """Ingest and index documents.

        * ``new_corpus=True`` (default) creates a fresh isolated corpus per
          ingestion call — this is the production path that prevents cross-
          domain contamination.
        * Pass an explicit ``corpus_id`` (and ``new_corpus=False``) to add
          documents to an existing corpus.
        * ``corpus_name`` overrides the auto-derived name for a new corpus.
        """
        ingestion_start = time.perf_counter()
        timings: dict[str, float] = {}
        ingest_warnings: list[str] = []

        load_start = time.perf_counter()
        documents = self._safe(
            "ingestion.load_documents",
            lambda: load_documents(file_paths=file_paths, urls=urls, warnings=ingest_warnings),
            [],
        )
        timings["load_ms"] = self._ms_since(load_start)

        # Decide which corpus this ingestion lands in.
        bundle: CorpusBundle
        if new_corpus and documents:
            resolved_name = (corpus_name or "").strip() or derive_corpus_name(file_paths, urls)
            corpus_text = " ".join(d.text for d in documents)
            corpus_domain = detect_domain(corpus_text)
            bundle = self.corpus_manager.create(
                name=resolved_name,
                domain=corpus_domain,
                source_urls=list(urls),
                source_files=list(file_paths),
            )
            self.logger.info(
                "ingest.new_corpus id=%s name=%r domain=%s",
                bundle.meta.corpus_id,
                bundle.meta.name,
                bundle.meta.domain,
            )
        else:
            bundle = self._resolve_corpus(corpus_id)
        # Refresh legacy attributes so existing code paths see this corpus.
        self.graph_store = bundle.graph_store
        self.retrieval = bundle.retrieval
        # Log loaded documents for ingestion traceability.
        for doc in documents:
            self.logger.info(
                "ingest.doc_loaded id=%s source=%s chars=%d",
                doc.doc_id,
                doc.source[:120],
                len(doc.text),
            )

        chunk_start = time.perf_counter()
        chunks = self._safe(
            "ingestion.chunk_documents",
            lambda: chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap),
            [],
        )
        timings["chunk_ms"] = self._ms_since(chunk_start)
        # Log chunk count and a short sample for ingestion diagnostics.
        self.logger.info("ingest.chunks count=%d", len(chunks))
        if chunks:
            self.logger.info("ingest.chunk_sample id=%s text=%r", chunks[0].chunk_id, chunks[0].text[:120])

        extract_start = time.perf_counter()
        entities, relations = self._safe(
            "extraction.extract_from_chunks",
            lambda: extract_from_chunks(chunks),
            ([], []),
        )
        timings["extract_ms"] = self._ms_since(extract_start)
        # Log entity count and a few samples for extraction diagnostics.
        self.logger.info("ingest.entities count=%d", len(entities))
        for ent in entities[:5]:
            self.logger.info("ingest.entity_sample text=%s type=%s", ent.text, ent.entity_type)

        contradictions: list[ContradictionItem] = []
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
            lambda: self.graph_store.upsert_from_extractions(documents, chunks, entities, relations),
            ([], []),
        )
        timings["graph_upsert_ms"] = self._ms_since(graph_start)

        # Mirror the extraction into Neo4j (production graph backend). Failures
        # never block ingestion — retrieval degrades to BM25 + FAISS.
        neo4j_start = time.perf_counter()
        self._safe(
            "graph.neo4j_sync",
            lambda: self._sync_to_neo4j(bundle.meta.corpus_id, documents, chunks, entities, relations),
            None,
        )
        timings["neo4j_sync_ms"] = self._ms_since(neo4j_start)

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
        METRICS.observe_modules((name, ms) for name, ms in timings.items() if name != "total_ms")
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

        # Update corpus metadata with final counts.
        self.corpus_manager.update_meta(
            bundle.meta.corpus_id,
            document_count=bundle.meta.document_count + len(documents),
            chunk_count=bundle.meta.chunk_count + len(chunks),
            entity_count=bundle.meta.entity_count + len(entities),
        )
        # Persist the populated graph + chunks to the durable store (if any)
        # so this ingest is visible to every other instance. No-op in file
        # mode, where the local files are already the source of truth.
        self.corpus_manager.flush(bundle.meta.corpus_id)

        return IngestResponse(
            documents=len(documents),
            chunks=len(chunks),
            entities=len(entities),
            relations=len(relations),
            graph_version_id=str(version_info.get("graph_version_id", "error")),
            warnings=ingest_warnings,
            corpus_id=bundle.meta.corpus_id,
            corpus_name=bundle.meta.name,
            corpus_domain=bundle.meta.domain,
        )

    def _sync_to_neo4j(
        self,
        corpus_id: str,
        documents: list[Any],
        chunks: list[Any],
        entities: list[Any],
        relations: list[Any],
    ) -> None:
        """Push this ingest's extractions into Neo4j, batched, corpus-scoped."""
        from graphrag_plus.app.graph.neo4j_store import get_neo4j_store
        from graphrag_plus.app.models.schemas import Entity, EntityType, Relationship

        store = get_neo4j_store()
        if not store.health_check():
            self.logger.warning("neo4j.sync_skipped corpus=%s — database unreachable", corpus_id)
            return

        valid_types = {member.value for member in EntityType}
        doc_sources = {doc.doc_id: doc.source for doc in documents}

        schema_entities = [
            Entity(
                name=entity.text,
                type=EntityType(entity.entity_type.upper() if entity.entity_type.upper() in valid_types else "OTHER"),
                confidence=float(entity.confidence),
            )
            for entity in entities
        ]
        schema_rels = [
            Relationship(
                source=rel.subject,
                target=rel.obj,
                relation=rel.predicate,
                confidence=float(rel.confidence),
            )
            for rel in relations
        ]
        chunk_rows = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": doc_sources.get(chunk.doc_id, chunk.doc_id),
            }
            for chunk in chunks
        ]
        link_rows = [
            {"entity_name": entity.text, "chunk_id": entity.source_chunk_id} for entity in entities
        ]

        store.add_chunks_batch(chunk_rows, corpus_id)
        store.add_entities_batch(schema_entities, corpus_id)
        store.add_relationships_batch(schema_rels, corpus_id)
        store.link_entities_batch(link_rows, corpus_id)
        log_event(
            self.logger,
            "neo4j.sync_complete",
            {
                "corpus_id": corpus_id,
                "chunks": len(chunk_rows),
                "entities": len(schema_entities),
                "relations": len(schema_rels),
                "links": len(link_rows),
            },
        )

    def _record_contradictions(self, contradictions: list[ContradictionItem]) -> None:
        """Persist contradictions for query-time consumption + update trust."""
        for item in contradictions:
            for chunk_id in (item.source_a, item.source_b):
                self._chunk_contradictions.setdefault(chunk_id, []).append(item)

            # Bind locals explicitly so each safe call captures the right value.
            source_a = item.source_a
            source_b = item.source_b

            def _update_a(sa: str = source_a) -> None:
                self.trust_manager.update(sa, agrees=False, is_correct=False, low_confidence=True)

            def _update_b(sb: str = source_b) -> None:
                self.trust_manager.update(sb, agrees=False, is_correct=False, low_confidence=True)

            self._safe("trust.update_a", _update_a, None)
            self._safe("trust.update_b", _update_b, None)

    # ------------------------------------------------------------------ query
    def query(self, request: QueryRequest) -> QueryResponse:
        """Run planned retrieval and generation against a specific corpus.

        ``request.corpus_id`` selects which isolated corpus to query. When
        omitted, falls through to the active corpus — preserves backward
        compatibility for existing tests / API clients.
        """
        # Bind the right corpus's stores into ``self`` for the duration of
        # this call so all the helpers below transparently use it.
        bundle = self._resolve_corpus(getattr(request, "corpus_id", None))
        self.graph_store = bundle.graph_store
        self.retrieval = bundle.retrieval
        query_id = f"qry_{uuid.uuid4().hex[:12]}"
        started_at = utc_now_iso()
        query_start = time.perf_counter()
        module_timings: dict[str, float] = {
            "planning_ms": 0.0,
            "retrieval_ms": 0.0,
            "scoring_ms": 0.0,
            "generation_ms": 0.0,
        }

        plan_start = time.perf_counter()
        plan = self._safe("planning.plan_query", lambda: plan_query(request.question), None)
        # Intent detection drives adaptive retrieval + intent-aware generation.
        intent_signal = self._safe(
            "planning.detect_intent",
            lambda: detect_intent_signal(request.question),
            None,
        )
        intent: QueryIntent = intent_signal.intent if intent_signal else QueryIntent.FACTUAL
        cmp_terms: tuple[str, str] | None = None
        if intent == QueryIntent.COMPARISON:
            cmp_terms = self._safe(
                "planning.comparison_terms",
                lambda: comparison_terms(request.question),
                None,
            )
        # Adapt top_k: definition wants 2; list wants 5-8; explanation 5+.
        effective_top_k = adaptive_top_k(intent, request.top_k)
        module_timings["planning_ms"] = self._ms_since(plan_start)
        _ = plan  # reserved for future routing policy

        trust_lookup = self._build_trust_lookup()

        retrieval_start = time.perf_counter()
        candidates = self._safe(
            "retrieval.query",
            lambda: self.retrieval.query(
                request.question,
                effective_top_k,
                trust_lookup,
                intent=intent.value,
                comparison_terms=cmp_terms,
            ),
            [],
        )
        # Fallback: if strict retrieval found nothing, try a loose pass that
        # halves the relevance floor and skips the term-overlap gate. This
        # only fires when the strict pass produced 0 candidates — a strong
        # match still wins. NO_EVIDENCE classification still applies if even
        # the loose pass returns nothing.
        if not candidates:
            self.logger.info("retrieval.fallback_loose question=%r", request.question[:80])
            candidates = self._safe(
                "retrieval.query.loose",
                lambda: self.retrieval.query(
                    request.question,
                    effective_top_k,
                    trust_lookup,
                    intent=intent.value,
                    comparison_terms=cmp_terms,
                    loose=True,
                ),
                [],
            )
        module_timings["retrieval_ms"] = self._ms_since(retrieval_start)

        if self.settings.use_gnn and candidates:
            gnn_scores = self._safe("gnn.score", lambda: self.gnn.score(candidates), [])
            if gnn_scores and len(gnn_scores) == len(candidates):
                for idx, gnn_score in enumerate(gnn_scores):
                    candidates[idx]["graph_score"] = 0.5 * candidates[idx]["graph_score"] + 0.5 * gnn_score

        scoring_start = time.perf_counter()
        scored = self._safe("scoring.score_candidates", lambda: self.scoring.score_candidates(candidates), [])
        module_timings["scoring_ms"] = self._ms_since(scoring_start)

        top = scored[:effective_top_k] if scored else []
        # Use the *pre-normalization* confidence so a strong single-doc match
        # doesn't get crushed to 0.5 by min-max scaling. Falls back to the
        # normalized value for older candidate rows that don't carry the raw
        # field.
        raw_confidence = (
            float(sum(item.get("raw_confidence_score", item["confidence_score"]) for item in top) / len(top))
            if top
            else 0.0
        )

        calibrated_confidence, calibration_error = self._calibrate(raw_confidence)

        evidence_items = [self._evidence_from(item) for item in top]
        evidence_paths = [[e.source_id, "supports", e.id] for e in evidence_items]

        # Real contradiction detection: an evidence chunk is conflicting iff the
        # ingestion-time reasoner flagged it.
        conflicting, resolution_explanation, has_conflict = self._collect_conflicts(
            evidence_items, request.question
        )

        # Per-request LLM override — restored after generation so the global
        # default isn't perturbed across requests.
        prior_llm_enabled = self.generator.llm_enabled
        if request.llm_enabled is not None:
            self.generator.llm_enabled = request.llm_enabled

        generation_start = time.perf_counter()
        answer_text, used_llm, llm_failed, verification = self._safe(
            "generation.generate",
            lambda: self.generator.generate_verified(
                request.question,
                [item.model_dump() for item in evidence_items],
                calibrated_confidence,
                self.settings.answer_threshold,
                intent=intent.value,
                comparison_terms=cmp_terms,
            ),
            ("I cannot answer reliably due to an internal error.", False, True, None),
        )
        module_timings["generation_ms"] = self._ms_since(generation_start)
        # Restore the global default so subsequent requests aren't affected.
        self.generator.llm_enabled = prior_llm_enabled

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
            lambda: self.version_manager.record_answer(answer_id, graph_version_id, supporting_nodes),
            None,
        )
        answer_state = self._safe(
            "versioning.detect_answer_state",
            lambda: self.version_manager.detect_answer_state(supporting_nodes, self.latest_changed_nodes),
            "updated",
        )

        reasoning_steps: list[str] = []
        follow_ups: list[str] = []
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
            generated_by="llm" if used_llm else "extractive",
            evidence=evidence_items,
            evidence_paths=evidence_paths,
            explanation=failure["explanation"] or "Answer generated from top ranked evidence.",
            conflicting_evidence=conflicting,
            resolution_explanation=resolution_explanation,
            failure_type=FailureType(failure["failure_type"]) if failure["failure_type"] else None,
            mitigation_strategy_used=failure["mitigation"],
            reasoning_steps=reasoning_steps,
            follow_up_questions=follow_ups,
            graph_version_id=graph_version_id,
            answer_state=answer_state,
            query_intent=intent.value,
            corpus_id=bundle.meta.corpus_id,
            corpus_name=bundle.meta.name,
            corpus_domain=bundle.meta.domain,
            verified_by_reasoning=bool(verification and getattr(verification, "verified", False)),
            reasoning_summary=str(getattr(verification, "reasoning_summary", "") or ""),
            answer_changed_by_reasoning=bool(verification and getattr(verification, "changed", False)),
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
                "failure_type": (response.failure_type.value if response.failure_type is not None else None),
            },
        )
        return response

    # ----------------------------------------------------------- query helpers
    def _build_trust_lookup(self) -> dict[str, float]:
        lookup: dict[str, float] = {}
        try:
            for node_id, attrs in self.graph_store.graph.nodes(data=True):
                if attrs.get("node_type") == "Document":
                    lookup[node_id] = self.trust_manager.get_trust_score(node_id)
        except Exception as exc:
            log_event(self.logger, "trust_lookup_failed", {"error": str(exc)})
        return lookup

    def _calibrate(self, raw_confidence: float) -> tuple[float, float]:
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
    def _evidence_from(item: dict[str, Any]) -> EvidenceItem:
        return EvidenceItem(
            id=item["id"],
            source_id=item["source_id"],
            snippet=item["snippet"],
            full_text=item.get("full_text"),
            semantic_score=float(item["semantic_score"]),
            graph_score=float(item["graph_score"]),
            confidence_score=float(item["confidence_score"]),
            trust_score=float(item["trust_score"]),
            uncertainty_penalty=float(item["uncertainty_penalty"]),
            final_score=float(item["final_score"]),
        )

    def _collect_conflicts(
        self, evidence_items: list[EvidenceItem], question: str
    ) -> tuple[list[ContradictionItem], str | None, bool]:
        """Return contradictions whose chunk ids are present in current evidence."""
        if not evidence_items or not self._chunk_contradictions:
            return [], None, False

        evidence_ids: set[str] = {e.id for e in evidence_items}
        seen: set[tuple[str, str, str]] = set()
        conflicting: list[ContradictionItem] = []

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
            "Source ranking favored higher trust and confidence evidence " "during contradiction resolution."
        )
        # Provide question for downstream analyst context but keep ContradictionItem schema stable.
        _ = question
        return conflicting, resolution, True
