"""Entity resolution — merge duplicate entity nodes in Neo4j.

The #1 production GraphRAG failure mode: the same real-world entity stored
as "Tim Cook", "T. Cook", and "Timothy Cook" fragments the graph into three
weakly-connected nodes. This module collapses them into one canonical node.

Two strategies, run in order per corpus:

1. **String similarity** (fast) — ``difflib.SequenceMatcher`` over pairs of
   same-type entities; pairs above ``_STRING_THRESHOLD`` merge.
2. **Embedding similarity** (semantic) — remaining names are embedded with
   the shared embedder; same-type pairs with cosine above
   ``_EMBEDDING_THRESHOLD`` merge ("IBM" vs "International Business
   Machines" — zero string overlap, near-identical meaning).

Merge policy: the LONGER name is canonical (more complete), all
relationships (RELATES_TO both directions + MENTIONED_IN) transfer to the
canonical node via MERGE (no duplicate edges), then the duplicate node is
deleted. A union-find keeps chains (A→B, B→C) consistent.
"""

from __future__ import annotations

import time
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

_STRING_THRESHOLD = 0.85
_EMBEDDING_THRESHOLD = 0.92
# Names shorter than this are too ambiguous to auto-merge ("AI" vs "Al").
_MIN_NAME_LEN = 3


def _is_token_prefix(a: str, b: str) -> bool:
    """True when one name's tokens are a strict prefix of the other's.

    "Apple" / "Apple Incorporated" → True (1 extra token).
    Capped at 2 extra tokens so "Apple" never absorbs a long unrelated
    phrase that merely starts with the same word.
    """
    ta, tb = a.lower().split(), b.lower().split()
    shorter, longer = (ta, tb) if len(ta) < len(tb) else (tb, ta)
    if not shorter or len(longer) - len(shorter) > 2 or len(shorter) == len(longer):
        return False
    return longer[: len(shorter)] == shorter


class _UnionFind:
    """Canonical-representative tracking for transitive merges."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        while self.parent.get(key, key) != key:
            self.parent[key] = self.parent.get(self.parent[key], self.parent[key])
            key = self.parent[key]
        return key

    def union(self, duplicate: str, canonical: str) -> None:
        self.parent[self.find(duplicate)] = self.find(canonical)


class EntityResolver:
    """Deduplicate Entity nodes for a corpus inside Neo4j."""

    def __init__(self, neo4j_store: Any, embedder: Any):
        self.store = neo4j_store
        self.embedder = embedder

    # ---------------------------------------------------------------- public
    def resolve_corpus(self, corpus_id: str) -> dict[str, Any]:
        """Run both strategies; returns merge stats.

        Never raises — any failure logs and returns partial stats so
        ingestion is never blocked by resolution.
        """
        start = time.perf_counter()
        result: dict[str, Any] = {"merged_count": 0, "canonical_pairs": [], "time_taken_s": 0.0}
        try:
            entities = self._fetch_entities(corpus_id)
            if len(entities) < 2:
                result["time_taken_s"] = round(time.perf_counter() - start, 2)
                return result

            uf = _UnionFind()
            pairs = self._string_pass(entities, uf)
            pairs += self._embedding_pass(entities, uf, already_merged={d for d, _, _ in pairs})
            similarity_of = {frozenset((d, c)): s for d, c, s in pairs}

            # Merge by union-find CLUSTER, not by recorded pair: chains like
            # Apple → Apple Inc → Apple Incorporated must fully collapse even
            # when a pair was linked only transitively. Canonical = the
            # longest (most complete) name in the cluster.
            clusters: dict[str, list[str]] = {}
            for row in entities:
                clusters.setdefault(uf.find(row["name"]), []).append(row["name"])
            for members in clusters.values():
                if len(members) < 2:
                    continue
                canonical = max(members, key=len)
                for duplicate in members:
                    if duplicate == canonical:
                        continue
                    similarity = similarity_of.get(frozenset((duplicate, canonical)), 0.9)
                    self._merge_nodes(duplicate, canonical, corpus_id)
                    result["canonical_pairs"].append(
                        {
                            "duplicate": duplicate,
                            "canonical": canonical,
                            "similarity": round(similarity, 3),
                        }
                    )
                    logger.info(
                        "resolver.merged corpus=%s [%s] -> [%s] (similarity: %.3f)",
                        corpus_id,
                        duplicate,
                        canonical,
                        similarity,
                    )
            result["merged_count"] = len(result["canonical_pairs"])
        except Exception as exc:
            logger.warning("resolver.failed corpus=%s error=%s", corpus_id, str(exc)[:200])
        result["time_taken_s"] = round(time.perf_counter() - start, 2)
        return result

    # ------------------------------------------------------------- strategies
    def _string_pass(
        self, entities: list[dict[str, str]], uf: _UnionFind
    ) -> list[tuple[str, str, float]]:
        """SequenceMatcher over same-type pairs. Returns (dup, canon, sim)."""
        merges: list[tuple[str, str, float]] = []
        by_type: dict[str, list[str]] = {}
        for row in entities:
            by_type.setdefault(row["type"], []).append(row["name"])

        for names in by_type.values():
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    if uf.find(a) == uf.find(b):
                        continue
                    if min(len(a), len(b)) < _MIN_NAME_LEN:
                        continue
                    similarity = SequenceMatcher(None, a.lower(), b.lower()).ratio()
                    if a.lower() == b.lower():
                        similarity = 1.0
                    elif _is_token_prefix(a, b):
                        # "Apple" ⊂ "Apple Incorporated": same-type token-prefix
                        # containment is a corporate/person alias with near
                        # certainty even when the char-level ratio is low.
                        similarity = max(similarity, 0.90)
                    if similarity <= _STRING_THRESHOLD:
                        continue
                    duplicate, canonical = (a, b) if len(a) < len(b) else (b, a)
                    uf.union(duplicate, canonical)
                    merges.append((duplicate, canonical, similarity))
        return merges

    def _embedding_pass(
        self, entities: list[dict[str, str]], uf: _UnionFind, already_merged: set[str]
    ) -> list[tuple[str, str, float]]:
        """Cosine similarity of name embeddings over surviving same-type pairs."""
        survivors = [row for row in entities if row["name"] not in already_merged]
        if len(survivors) < 2:
            return []
        try:
            if not self.embedder.available():
                return []
            vectors = self.embedder.embed_batch([row["name"] for row in survivors])
            if len(vectors) != len(survivors):
                return []
        except Exception as exc:
            logger.warning("resolver.embedding_pass_failed error=%s", str(exc)[:150])
            return []

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
        similarity_matrix = matrix @ matrix.T

        merges: list[tuple[str, str, float]] = []
        for i in range(len(survivors)):
            for j in range(i + 1, len(survivors)):
                if survivors[i]["type"] != survivors[j]["type"]:
                    continue
                a, b = survivors[i]["name"], survivors[j]["name"]
                if uf.find(a) == uf.find(b):
                    continue
                if min(len(a), len(b)) < _MIN_NAME_LEN:
                    continue
                similarity = float(similarity_matrix[i, j])
                if similarity <= _EMBEDDING_THRESHOLD:
                    continue
                duplicate, canonical = (a, b) if len(a) < len(b) else (b, a)
                uf.union(duplicate, canonical)
                merges.append((duplicate, canonical, similarity))
        return merges

    # -------------------------------------------------------------- neo4j ops
    def _fetch_entities(self, corpus_id: str) -> list[dict[str, str]]:
        rows = self.store._run(
            "MATCH (e:Entity {corpus_id: $corpus_id}) RETURN e.name AS name, coalesce(e.type, 'OTHER') AS type",
            corpus_id=corpus_id,
        )
        return [{"name": str(r["name"]), "type": str(r["type"])} for r in rows if r.get("name")]

    def _merge_nodes(self, duplicate: str, canonical: str, corpus_id: str) -> None:
        """Transfer every edge from ``duplicate`` to ``canonical``, delete dup."""
        params = {"dup": duplicate, "canon": canonical, "corpus_id": corpus_id}
        # Outgoing typed relations.
        self.store._run(
            """
            MATCH (d:Entity {name: $dup, corpus_id: $corpus_id})-[r:RELATES_TO]->(t)
            MATCH (c:Entity {name: $canon, corpus_id: $corpus_id})
            WHERE t <> c
            MERGE (c)-[nr:RELATES_TO {relation: r.relation}]->(t)
            SET nr.confidence = coalesce(nr.confidence, r.confidence)
            """,
            **params,
        )
        # Incoming typed relations.
        self.store._run(
            """
            MATCH (s)-[r:RELATES_TO]->(d:Entity {name: $dup, corpus_id: $corpus_id})
            MATCH (c:Entity {name: $canon, corpus_id: $corpus_id})
            WHERE s <> c
            MERGE (s)-[nr:RELATES_TO {relation: r.relation}]->(c)
            SET nr.confidence = coalesce(nr.confidence, r.confidence)
            """,
            **params,
        )
        # Chunk mentions.
        self.store._run(
            """
            MATCH (d:Entity {name: $dup, corpus_id: $corpus_id})-[:MENTIONED_IN]->(ch:Chunk)
            MATCH (c:Entity {name: $canon, corpus_id: $corpus_id})
            MERGE (c)-[:MENTIONED_IN]->(ch)
            """,
            **params,
        )
        # Remove the duplicate (and any leftover edges).
        self.store._run(
            "MATCH (d:Entity {name: $dup, corpus_id: $corpus_id}) DETACH DELETE d",
            **params,
        )
