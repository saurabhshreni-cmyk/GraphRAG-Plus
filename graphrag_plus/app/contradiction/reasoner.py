"""Contradiction and stance detection."""

from __future__ import annotations

from collections import defaultdict

from graphrag_plus.app.extraction.models import Relation
from graphrag_plus.app.schemas.models import ContradictionItem

# Predicates where one subject legitimately has several objects. Two values
# here are not a contradiction: a company is "based_in" both Boulder and
# Colorado, "acquired" many startups, "partners_with" many firms. Only
# single-valued facts (status, founded_in, headquartered_in, ceo_of, ...)
# can conflict. This list matters once an LLM extracts typed relations: on
# the cloud deploy it flagged "based_in Boulder" vs "based_in Colorado".
MULTI_VALUED_PREDICATES = frozenset(
    {
        "acquired",
        "based_in",
        "located_in",
        "operates_in",
        "has_office_in",
        "partner_with",
        "partners_with",
        "partnered_with",
        "collaborates_with",
        "works_with",
        "competes_with",
        "deploys",
        "uses",
        "used_in",
        "used_by",
        "develops",
        "builds",
        "produces",
        "makes",
        "sells",
        "offers",
        "owns",
        "invests_in",
        "invested_in",
        "employs",
        "works_at",
        "member_of",
        "part_of",
        "has_part",
        "includes",
        "related_to",
        "mentions",
        "co_occurs",
    }
)


def _normalize_predicate(predicate: str) -> str:
    return "_".join(predicate.lower().replace("-", " ").split())


def can_conflict(predicate: str) -> bool:
    """True when two different objects for this predicate are a contradiction."""
    return _normalize_predicate(predicate) not in MULTI_VALUED_PREDICATES


class ContradictionReasoner:
    """Detect contradictory claims across relations."""

    def detect(self, relations: list[Relation]) -> tuple[list[Relation], list[ContradictionItem]]:
        grouped: dict[tuple[str, str], list[Relation]] = defaultdict(list)
        for relation in relations:
            if not can_conflict(relation.predicate):
                continue
            key = (relation.subject.lower(), _normalize_predicate(relation.predicate))
            grouped[key].append(relation)

        contradictions: list[ContradictionItem] = []
        for _key, rel_group in grouped.items():
            seen_objects: dict[str, Relation] = {}
            for relation in rel_group:
                obj_key = relation.obj.lower().strip()
                if obj_key in seen_objects:
                    continue
                # A source does not contradict itself: two objects stated in
                # the same chunk ("Boulder, Colorado") are both part of one claim.
                rivals = [r for r in seen_objects.values() if r.source_chunk_id != relation.source_chunk_id]
                if rivals:
                    first = rivals[0]
                    contradictions.append(
                        ContradictionItem(
                            source_a=first.source_chunk_id,
                            source_b=relation.source_chunk_id,
                            claim=f"{relation.subject} {relation.predicate}",
                            explanation=f"Different objects observed: '{first.obj}' vs '{relation.obj}'.",
                        )
                    )
                    relation.stance = "contradicts"
                seen_objects[obj_key] = relation
        return relations, contradictions
