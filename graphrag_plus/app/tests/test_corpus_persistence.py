"""Cross-instance persistence: ingest on one instance, query on another.

This is the whole point of the blob store. Two pipelines with *separate*
scratch dirs but a *shared* blob store stand in for two serverless
instances. Data ingested through the first must be queryable through the
second, and survive a fresh manager (a "restart").
"""

from pathlib import Path

from graphrag_plus.app.config.settings import Settings
from graphrag_plus.app.corpus.blob_store import InMemoryBlobStore
from graphrag_plus.app.corpus.manager import CorpusManager
from graphrag_plus.app.corpus.seed import DEMO_CORPUS_ID, seed_demo_corpus
from graphrag_plus.app.pipeline import GraphRAGPipeline
from graphrag_plus.app.schemas.models import QueryRequest


def _settings(scratch: Path) -> Settings:
    return Settings(
        data_dir=scratch,
        corpora_dir=scratch / "corpora",
        graph_versions_dir=scratch / "gv",
        outputs_dir=scratch / "out",
        reports_dir=scratch / "reports",
        cache_dir=scratch / ".cache",
        temp_dir=scratch / ".cache" / "tmp",
    )


def _pipeline_on(store: InMemoryBlobStore, scratch: Path) -> GraphRAGPipeline:
    """A pipeline whose corpus manager is backed by the shared store."""
    pipe = GraphRAGPipeline(_settings(scratch))
    # Re-bind the corpus manager onto the shared store + this instance's
    # scratch dir (the constructor builds one from settings.database_url,
    # which we can't point at an in-memory object).
    pipe.corpus_manager = CorpusManager(scratch / "corpora", blob_store=store)
    return pipe


def test_ingest_on_one_instance_query_on_another(tmp_path: Path) -> None:
    store = InMemoryBlobStore()

    # --- Instance A ingests a document into a new corpus ------------------
    doc = tmp_path / "doc.txt"
    doc.write_text("Acme Corp launched Project Titan in 2024.", encoding="utf-8")
    inst_a = _pipeline_on(store, tmp_path / "a")
    res = inst_a.ingest([str(doc)], [], new_corpus=True, corpus_name="Acme")
    corpus_id = res.corpus_id
    assert res.entities > 0

    # --- Instance B (separate scratch dir) must see + query it -----------
    inst_b = _pipeline_on(store, tmp_path / "b")
    assert corpus_id in {m.corpus_id for m in inst_b.corpus_manager.list()}
    answer = inst_b.query(QueryRequest(question="What did Acme Corp launch?", corpus_id=corpus_id))
    assert answer.corpus_id == corpus_id  # served the right corpus, no self-heal
    assert answer.answer


def test_demo_seed_into_store_visible_to_fresh_instance(tmp_path: Path) -> None:
    store = InMemoryBlobStore()
    assert seed_demo_corpus(tmp_path / "seed_scratch", store) is True
    # Idempotent second call.
    assert seed_demo_corpus(tmp_path / "seed_scratch", store) is False

    mgr = CorpusManager(tmp_path / "fresh", blob_store=store)
    ids = {m.corpus_id for m in mgr.list()}
    assert DEMO_CORPUS_ID in ids
    bundle = mgr.get(DEMO_CORPUS_ID)
    assert bundle.graph_store.current_snapshot()["nodes"]


def test_meta_counts_persist_across_instances(tmp_path: Path) -> None:
    store = InMemoryBlobStore()
    doc = tmp_path / "d.txt"
    doc.write_text("Nova Dynamics acquired Orion Labs on 2024-01-15.", encoding="utf-8")
    inst_a = _pipeline_on(store, tmp_path / "a")
    res = inst_a.ingest([str(doc)], [], new_corpus=True, corpus_name="Nova")

    inst_b = _pipeline_on(store, tmp_path / "b")
    meta = next(m for m in inst_b.corpus_manager.list() if m.corpus_id == res.corpus_id)
    assert meta.document_count == 1
    assert meta.entity_count > 0
