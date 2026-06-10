"""Corpus manager security tests — corpus_id is user input from the API.

``corpus_id`` flows from URL path params straight into filesystem paths
(including ``shutil.rmtree`` on delete), so it MUST be validated against
a strict allowlist pattern before touching disk.
"""

from pathlib import Path

import pytest

from graphrag_plus.app.corpus.manager import CorpusManager

TRAVERSAL_IDS = [
    "../outside",
    "..\\outside",
    "corpus_x/../../outside",
    "corpus_x/..",
    "/absolute/path",
    "C:\\Windows\\Temp",
    "corpus_x/nested",
    "..",
    ".",
    "",
    "corpus_" + "a" * 200,  # absurdly long
    "not_a_corpus",
]


@pytest.fixture()
def manager(tmp_path: Path) -> CorpusManager:
    return CorpusManager(tmp_path / "corpora")


@pytest.mark.parametrize("bad_id", TRAVERSAL_IDS)
def test_get_rejects_malformed_corpus_id(manager: CorpusManager, bad_id: str) -> None:
    with pytest.raises(KeyError):
        manager.get(bad_id)


@pytest.mark.parametrize("bad_id", TRAVERSAL_IDS)
def test_delete_rejects_malformed_corpus_id(manager: CorpusManager, bad_id: str, tmp_path: Path) -> None:
    # Plant a sibling directory that a traversal delete would destroy.
    victim = tmp_path / "outside"
    victim.mkdir(exist_ok=True)
    (victim / "keep.txt").write_text("important", encoding="utf-8")

    with pytest.raises(KeyError):
        manager.delete(bad_id)

    assert (victim / "keep.txt").exists(), "traversal delete escaped the corpora dir"


def test_well_formed_ids_still_work(manager: CorpusManager) -> None:
    bundle = manager.create(name="ok", domain="general")
    assert manager.get(bundle.meta.corpus_id) is bundle
    # The shipped demo corpus naming style must stay valid.
    manager.delete(bundle.meta.corpus_id)


def test_demo_corpus_id_shape_is_valid(manager: CorpusManager) -> None:
    # "corpus_demo_nova" is committed to the repo; the validator must accept it.
    with pytest.raises(KeyError, match="Unknown corpus_id"):
        manager.get("corpus_demo_nova")  # unknown here, but shape-valid
