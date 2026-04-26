"""Trust manager tests."""

from pathlib import Path

from graphrag_plus.app.trust.manager import SourceTrustManager


def test_trust_update_and_persist(tmp_path: Path) -> None:
    path = tmp_path / "trust.json"
    manager = SourceTrustManager(path, default_prior=0.5, priors={})
    initial = manager.get_trust_score("doc_a")
    manager.update("doc_a", agrees=True, is_correct=True, low_confidence=False)
    updated = manager.get_trust_score("doc_a")
    assert updated >= initial
    assert path.exists()

