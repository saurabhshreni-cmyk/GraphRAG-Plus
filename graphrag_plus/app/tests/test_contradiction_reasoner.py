"""Tests for contradiction detection."""

from __future__ import annotations

from graphrag_plus.app.contradiction.reasoner import ContradictionReasoner
from graphrag_plus.app.extraction.models import Relation


def _rel(subject: str, predicate: str, obj: str, source: str) -> Relation:
    return Relation(
        subject=subject,
        predicate=predicate,
        obj=obj,
        stance="supports",
        confidence=0.9,
        method="rule",
        source_chunk_id=source,
    )


def test_detects_disagreement_on_same_subject_predicate() -> None:
    reasoner = ContradictionReasoner()
    relations = [
        _rel("Project Helios", "status", "active", "chunk_a"),
        _rel("Project Helios", "status", "cancelled", "chunk_b"),
    ]
    updated, contradictions = reasoner.detect(relations)
    assert len(contradictions) == 1
    item = contradictions[0]
    assert {item.source_a, item.source_b} == {"chunk_a", "chunk_b"}
    # The second relation should be marked as contradicting after detection.
    assert any(r.stance == "contradicts" for r in updated)


def test_no_false_positive_on_agreeing_relations() -> None:
    reasoner = ContradictionReasoner()
    relations = [
        _rel("Acme", "founded_in", "1999", "chunk_a"),
        _rel("Acme", "founded_in", "1999", "chunk_b"),
    ]
    _, contradictions = reasoner.detect(relations)
    assert contradictions == []
