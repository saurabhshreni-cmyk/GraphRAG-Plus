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


def test_multi_valued_predicates_never_conflict() -> None:
    # Regression, seen live once Groq extracted typed relations: a company
    # "based_in" both a city and its state was reported as a contradiction.
    relations = [
        _rel("Helix Sensors", "based_in", "Boulder", "chunk_a"),
        _rel("Helix Sensors", "based_in", "Colorado", "chunk_b"),
        _rel("Nova Dynamics", "acquired", "Helix Sensors", "chunk_a"),
        _rel("Nova Dynamics", "acquired", "Orion Labs", "chunk_b"),
        _rel("Nova Dynamics", "Partners With", "Orbit Logistics", "chunk_a"),
        _rel("Nova Dynamics", "partners-with", "Acme", "chunk_b"),
    ]
    _, contradictions = ContradictionReasoner().detect(relations)
    assert contradictions == []


def test_single_valued_predicates_still_conflict() -> None:
    relations = [
        _rel("Nova Dynamics", "headquartered_in", "Austin", "chunk_a"),
        _rel("Nova Dynamics", "headquartered_in", "Boston", "chunk_b"),
        _rel("Acme", "founded_in", "1999", "chunk_a"),
        _rel("Acme", "Founded In", "2004", "chunk_b"),
    ]
    _, contradictions = ContradictionReasoner().detect(relations)
    assert {c.claim for c in contradictions} == {"Nova Dynamics headquartered_in", "Acme Founded In"}


def test_a_source_does_not_contradict_itself() -> None:
    relations = [
        _rel("Nova Dynamics", "headquartered_in", "Austin", "chunk_a"),
        _rel("Nova Dynamics", "headquartered_in", "Texas", "chunk_a"),
    ]
    _, contradictions = ContradictionReasoner().detect(relations)
    assert contradictions == []
