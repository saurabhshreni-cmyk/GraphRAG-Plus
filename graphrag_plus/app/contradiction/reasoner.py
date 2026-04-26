"""Contradiction and stance detection."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from graphrag_plus.app.extraction.models import Relation
from graphrag_plus.app.schemas.models import ContradictionItem


class ContradictionReasoner:
    """Detect contradictory claims across relations."""

    def detect(self, relations: List[Relation]) -> Tuple[List[Relation], List[ContradictionItem]]:
        grouped: Dict[Tuple[str, str], List[Relation]] = defaultdict(list)
        for relation in relations:
            key = (relation.subject.lower(), relation.predicate.lower())
            grouped[key].append(relation)

        contradictions: List[ContradictionItem] = []
        for key, rel_group in grouped.items():
            seen_objects: Dict[str, Relation] = {}
            for relation in rel_group:
                obj_key = relation.obj.lower().strip()
                if obj_key in seen_objects:
                    continue
                if seen_objects and obj_key not in seen_objects:
                    first = next(iter(seen_objects.values()))
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

