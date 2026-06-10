"""Domain models for corpus isolation."""

from __future__ import annotations

from dataclasses import dataclass, field

from graphrag_plus.app.graph.store import GraphStore
from graphrag_plus.app.retrieval.service import RetrievalService


@dataclass(frozen=True)
class CorpusMeta:
    """Persisted metadata for a corpus (name / domain / sources)."""

    corpus_id: str
    name: str
    domain: str
    source_urls: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    created_at: str = ""
    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0


@dataclass
class CorpusBundle:
    """In-memory bundle of (graph store + retrieval service + metadata).

    The bundle is the unit of isolation: every operation on a corpus goes
    through its own ``graph_store`` and ``retrieval`` so two corpora can
    never share state. The pipeline's ``CorpusManager`` holds these.
    """

    meta: CorpusMeta
    graph_store: GraphStore
    retrieval: RetrievalService
