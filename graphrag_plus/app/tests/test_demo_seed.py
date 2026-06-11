"""The demo corpus seeding used by Render/Vercel boot must be reliable."""

from pathlib import Path

from graphrag_plus.app.corpus.manager import CorpusManager
from graphrag_plus.app.corpus.seed import DEMO_CORPUS_ID, seed_demo_corpus


def test_seed_copies_into_fresh_dir(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"

    copied = seed_demo_corpus(corpora)

    assert copied is True
    target = corpora / DEMO_CORPUS_ID
    assert (target / "meta.json").exists()
    assert (target / "graph.json").exists()
    assert (target / "chunks.json").exists()


def test_seed_is_idempotent(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    assert seed_demo_corpus(corpora) is True
    # Second call must not re-copy or raise — mirrors every restart.
    assert seed_demo_corpus(corpora) is False


def test_seeded_corpus_is_visible_to_manager(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    seed_demo_corpus(corpora)

    manager = CorpusManager(corpora)
    ids = {meta.corpus_id for meta in manager.list()}

    assert DEMO_CORPUS_ID in ids
    # And it loads as a real bundle with a graph behind it.
    bundle = manager.get(DEMO_CORPUS_ID)
    assert bundle.graph_store.current_snapshot()["nodes"]
