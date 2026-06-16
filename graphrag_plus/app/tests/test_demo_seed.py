"""File-mode demo seeding (no database configured)."""

from pathlib import Path

from graphrag_plus.app.corpus.manager import CorpusManager
from graphrag_plus.app.corpus.seed import DEMO_CORPUS_ID, seed_demo_corpus


def test_seed_copies_into_fresh_dir(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    assert seed_demo_corpus(corpora) is True
    target = corpora / DEMO_CORPUS_ID
    assert (target / "meta.json").exists()
    assert (target / "graph.json").exists()
    assert (target / "chunks.json").exists()


def test_seed_is_idempotent(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    assert seed_demo_corpus(corpora) is True
    assert seed_demo_corpus(corpora) is False


def test_seeded_corpus_is_visible_to_manager(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    seed_demo_corpus(corpora)
    manager = CorpusManager(corpora)
    assert DEMO_CORPUS_ID in {meta.corpus_id for meta in manager.list()}
    bundle = manager.get(DEMO_CORPUS_ID)
    assert bundle.graph_store.current_snapshot()["nodes"]
