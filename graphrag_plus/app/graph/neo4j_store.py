"""Neo4j AuraDB graph store.

Production graph backend that complements (and can replace) the NetworkX
JSON store. All nodes carry a ``corpus_id`` property so multiple corpora
stay fully isolated inside one database.

Graph model::

    (:Entity {name, type, description, corpus_id})
    (:Chunk  {chunk_id, text, source, corpus_id})
    (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
    (a:Entity)-[:RELATES_TO {relation, confidence}]->(b:Entity)

Connection settings come from the environment (``NEO4J_URI``,
``NEO4J_USERNAME``, ``NEO4J_PASSWORD``) — loaded via ``.env`` at the project
root. Credentials are never hardcoded.

The store degrades gracefully: if the driver cannot connect, every method
logs a warning and returns an empty/False result instead of raising, so the
pipeline keeps serving from BM25 + FAISS.
"""

from __future__ import annotations

import atexit
import os
import threading
from typing import Any

from dotenv import load_dotenv

from graphrag_plus.app.models.schemas import Entity, Relationship
from graphrag_plus.app.utils.logging_utils import get_logger

load_dotenv()  # no-op when .env is absent; existing env vars win

logger = get_logger(__name__)

_BATCH_SIZE = 500


class Neo4jStore:
    """Thin, batched wrapper over the Neo4j Python driver."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        self.uri = uri or os.environ.get("NEO4J_URI", "")
        self.username = username or os.environ.get("NEO4J_USERNAME", "")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "")
        self._driver: Any = None
        self._connect_failed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------- connection
    def _get_driver(self) -> Any:
        """Lazily build the driver; None when config is missing or broken."""
        if self._driver is not None or self._connect_failed:
            return self._driver
        with self._lock:
            if self._driver is not None or self._connect_failed:
                return self._driver
            if not (self.uri and self.username and self.password):
                self._connect_failed = True
                logger.warning("neo4j.not_configured — set NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD in .env")
                return None
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    connection_timeout=15.0,
                    max_connection_lifetime=300,
                )
                self._driver.verify_connectivity()
                self._create_indexes()
                logger.info("neo4j.connected uri=%s", self.uri.split("@")[-1])
            except Exception as exc:
                self._connect_failed = True
                self._driver = None
                logger.warning("neo4j.connect_failed error=%s — graph signal disabled", str(exc)[:200])
        return self._driver

    def _create_indexes(self) -> None:
        """Idempotent index/constraint creation on first connection."""
        statements = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_corpus IF NOT EXISTS FOR (e:Entity) ON (e.corpus_id)",
            "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)",
            "CREATE INDEX chunk_corpus IF NOT EXISTS FOR (c:Chunk) ON (c.corpus_id)",
        ]
        assert self._driver is not None
        with self._driver.session() as session:
            for statement in statements:
                session.run(statement)

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute one statement; [] on any failure (logged, never raised)."""
        driver = self._get_driver()
        if driver is None:
            return []
        try:
            with driver.session() as session:
                return [dict(record) for record in session.run(cypher, **params)]
        except Exception as exc:
            logger.warning("neo4j.query_failed error=%s cypher=%s", str(exc)[:200], cypher[:120])
            return []

    # ------------------------------------------------------------ single ops
    def add_entity(self, entity: Entity, corpus_id: str) -> None:
        """Create or merge an Entity node (MERGE keyed on name + corpus)."""
        self._run(
            """
            MERGE (e:Entity {name: $name, corpus_id: $corpus_id})
            SET e.type = $type,
                e.description = coalesce($description, e.description),
                e.confidence = $confidence
            """,
            name=entity.name,
            corpus_id=corpus_id,
            type=entity.type.value,
            description=entity.description,
            confidence=entity.confidence,
        )

    def add_relationship(self, rel: Relationship, corpus_id: str) -> None:
        """Create a typed relationship between two entities (both merged)."""
        self._run(
            """
            MERGE (a:Entity {name: $source, corpus_id: $corpus_id})
            MERGE (b:Entity {name: $target, corpus_id: $corpus_id})
            MERGE (a)-[r:RELATES_TO {relation: $relation}]->(b)
            SET r.confidence = $confidence
            """,
            source=rel.source,
            target=rel.target,
            corpus_id=corpus_id,
            relation=rel.relation,
            confidence=rel.confidence,
        )

    def add_chunk(self, chunk_id: str, text: str, source: str, corpus_id: str) -> None:
        """Create or merge a Chunk node."""
        self._run(
            """
            MERGE (c:Chunk {chunk_id: $chunk_id, corpus_id: $corpus_id})
            SET c.text = $text, c.source = $source
            """,
            chunk_id=chunk_id,
            corpus_id=corpus_id,
            text=text[:2000],
            source=source,
        )

    def link_entity_to_chunk(self, entity_name: str, chunk_id: str, corpus_id: str) -> None:
        """MENTIONED_IN edge from an entity to the chunk that mentions it."""
        self._run(
            """
            MATCH (e:Entity {name: $name, corpus_id: $corpus_id})
            MATCH (c:Chunk {chunk_id: $chunk_id, corpus_id: $corpus_id})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """,
            name=entity_name,
            chunk_id=chunk_id,
            corpus_id=corpus_id,
        )

    # ------------------------------------------------------------- batch ops
    # Aura is a remote database: one round-trip per MERGE makes large ingests
    # crawl. These UNWIND variants push rows in batches of _BATCH_SIZE.
    def add_entities_batch(self, entities: list[Entity], corpus_id: str) -> None:
        rows = [
            {
                "name": e.name,
                "type": e.type.value,
                "description": e.description,
                "confidence": e.confidence,
            }
            for e in entities
        ]
        for i in range(0, len(rows), _BATCH_SIZE):
            self._run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {name: row.name, corpus_id: $corpus_id})
                SET e.type = row.type,
                    e.description = coalesce(row.description, e.description),
                    e.confidence = row.confidence
                """,
                rows=rows[i : i + _BATCH_SIZE],
                corpus_id=corpus_id,
            )

    def add_relationships_batch(self, rels: list[Relationship], corpus_id: str) -> None:
        rows = [
            {"source": r.source, "target": r.target, "relation": r.relation, "confidence": r.confidence}
            for r in rels
        ]
        for i in range(0, len(rows), _BATCH_SIZE):
            self._run(
                """
                UNWIND $rows AS row
                MERGE (a:Entity {name: row.source, corpus_id: $corpus_id})
                MERGE (b:Entity {name: row.target, corpus_id: $corpus_id})
                MERGE (a)-[r:RELATES_TO {relation: row.relation}]->(b)
                SET r.confidence = row.confidence
                """,
                rows=rows[i : i + _BATCH_SIZE],
                corpus_id=corpus_id,
            )

    def add_chunks_batch(self, chunks: list[dict[str, str]], corpus_id: str) -> None:
        """``chunks`` rows: {chunk_id, text, source}."""
        rows = [
            {"chunk_id": c["chunk_id"], "text": (c.get("text") or "")[:2000], "source": c.get("source", "")}
            for c in chunks
        ]
        for i in range(0, len(rows), _BATCH_SIZE):
            self._run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id, corpus_id: $corpus_id})
                SET c.text = row.text, c.source = row.source
                """,
                rows=rows[i : i + _BATCH_SIZE],
                corpus_id=corpus_id,
            )

    def link_entities_batch(self, links: list[dict[str, str]], corpus_id: str) -> None:
        """``links`` rows: {entity_name, chunk_id}."""
        for i in range(0, len(links), _BATCH_SIZE):
            self._run(
                """
                UNWIND $rows AS row
                MATCH (e:Entity {name: row.entity_name, corpus_id: $corpus_id})
                MATCH (c:Chunk {chunk_id: row.chunk_id, corpus_id: $corpus_id})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                rows=links[i : i + _BATCH_SIZE],
                corpus_id=corpus_id,
            )

    # -------------------------------------------------------------- traversal
    def get_related_chunks(
        self, entity_names: list[str], corpus_id: str, max_hops: int = 2
    ) -> list[dict[str, Any]]:
        """Chunks reachable from the named entities within ``max_hops``.

        Hop 0: chunks that directly mention a query entity (weight 1.0).
        Hop 1..max_hops: chunks mentioning entities related to a query
        entity, weight decaying with distance.
        """
        if not entity_names:
            return []
        max_hops = max(1, min(int(max_hops), 3))  # bound traversal cost
        rows = self._run(
            f"""
            UNWIND $names AS name
            MATCH (e:Entity {{corpus_id: $corpus_id}})
            WHERE toLower(e.name) = toLower(name)
            CALL (e) {{
                MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
                RETURN c, 0 AS hops
                UNION
                MATCH (e)-[:RELATES_TO*1..{max_hops}]-(nbr:Entity)-[:MENTIONED_IN]->(c:Chunk)
                WHERE nbr.corpus_id = $corpus_id
                RETURN c, 1 AS hops
            }}
            WITH c, min(hops) AS hops
            RETURN c.chunk_id AS chunk_id, c.source AS source, hops,
                   CASE hops WHEN 0 THEN 1.0 ELSE 0.5 END AS weight
            ORDER BY weight DESC
            LIMIT 50
            """,
            names=entity_names,
            corpus_id=corpus_id,
        )
        return rows

    def get_chunks_pagerank(
        self,
        seed_entity_names: list[str],
        corpus_id: str,
        iterations: int = 20,
        damping: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Personalized PageRank seeded on the query's entities.

        Consistently outperforms fixed-depth traversal for GraphRAG: score
        mass flows out from the seed entities across the whole entity graph,
        so strongly-connected multi-hop neighbours outrank weakly-connected
        one-hop ones.

        Adjacency = typed RELATES_TO edges ∪ co-mention edges (two entities
        sharing a chunk), treated as undirected. Chunks are scored by the
        summed PPR mass of the entities that mention them, normalized to
        [0, 1].

        Falls back to :meth:`get_related_chunks` (2-hop) when the corpus has
        fewer than 5 entities, no seed matches, or anything fails.
        """
        try:
            if not seed_entity_names:
                return []
            rows = self._run(
                "MATCH (e:Entity {corpus_id: $corpus_id}) RETURN e.name AS name",
                corpus_id=corpus_id,
            )
            names = [str(r["name"]) for r in rows if r.get("name")]
            if len(names) < 5:
                logger.info("retrieval.graph_signal method=2hop_fallback reason=small_corpus")
                return self.get_related_chunks(seed_entity_names, corpus_id)

            index = {name: i for i, name in enumerate(names)}
            lower_index = {name.lower(): i for i, name in enumerate(names)}

            # Seeds: exact case-insensitive match first, then containment.
            seeds: set[int] = set()
            for seed in seed_entity_names:
                s = seed.lower().strip()
                if not s:
                    continue
                if s in lower_index:
                    seeds.add(lower_index[s])
                    continue
                for name, i in index.items():
                    low = name.lower()
                    if s in low or low in s:
                        seeds.add(i)
            if not seeds:
                logger.info("retrieval.graph_signal method=2hop_fallback reason=no_seed_match")
                return self.get_related_chunks(seed_entity_names, corpus_id)

            # Undirected adjacency: typed relations + co-mentions.
            adjacency: dict[int, set[int]] = {i: set() for i in range(len(names))}
            rel_edges = self._run(
                """
                MATCH (a:Entity {corpus_id: $corpus_id})-[:RELATES_TO]->(b:Entity {corpus_id: $corpus_id})
                RETURN a.name AS a, b.name AS b
                """,
                corpus_id=corpus_id,
            )
            co_edges = self._run(
                """
                MATCH (a:Entity {corpus_id: $corpus_id})-[:MENTIONED_IN]->(c:Chunk)
                      <-[:MENTIONED_IN]-(b:Entity {corpus_id: $corpus_id})
                WHERE a.name < b.name
                RETURN a.name AS a, b.name AS b
                """,
                corpus_id=corpus_id,
            )
            for edge in list(rel_edges) + list(co_edges):
                ia, ib = index.get(str(edge.get("a", ""))), index.get(str(edge.get("b", "")))
                if ia is None or ib is None or ia == ib:
                    continue
                adjacency[ia].add(ib)
                adjacency[ib].add(ia)

            # Power iteration.
            n = len(names)
            personalization = [0.0] * n
            for i in seeds:
                personalization[i] = 1.0 / len(seeds)
            scores = list(personalization)
            for _ in range(max(1, iterations)):
                incoming = [0.0] * n
                for node, neighbours in adjacency.items():
                    if not neighbours or scores[node] == 0.0:
                        continue
                    share = scores[node] / len(neighbours)
                    for neighbour in neighbours:
                        incoming[neighbour] += share
                scores = [
                    damping * incoming[i] + (1.0 - damping) * personalization[i] for i in range(n)
                ]

            top = sorted(range(n), key=lambda i: scores[i], reverse=True)[:20]
            top = [i for i in top if scores[i] > 0.0]
            if not top:
                logger.info("retrieval.graph_signal method=2hop_fallback reason=zero_scores")
                return self.get_related_chunks(seed_entity_names, corpus_id)

            chunk_rows = self._run(
                """
                UNWIND $names AS name
                MATCH (e:Entity {name: name, corpus_id: $corpus_id})-[:MENTIONED_IN]->(c:Chunk)
                RETURN c.chunk_id AS chunk_id, c.source AS source, e.name AS entity
                """,
                names=[names[i] for i in top],
                corpus_id=corpus_id,
            )
            score_of = {names[i]: scores[i] for i in top}
            chunk_scores: dict[str, float] = {}
            chunk_sources: dict[str, str] = {}
            for row in chunk_rows:
                chunk_id = str(row.get("chunk_id", ""))
                if not chunk_id:
                    continue
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score_of.get(
                    str(row.get("entity", "")), 0.0
                )
                chunk_sources.setdefault(chunk_id, str(row.get("source", "")))
            if not chunk_scores:
                logger.info("retrieval.graph_signal method=2hop_fallback reason=no_chunks")
                return self.get_related_chunks(seed_entity_names, corpus_id)

            max_score = max(chunk_scores.values())
            logger.info(
                "retrieval.graph_signal method=pagerank seeds=%d entities=%d chunks=%d",
                len(seeds),
                n,
                len(chunk_scores),
            )
            ranked = sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True)[:50]
            return [
                {
                    "chunk_id": chunk_id,
                    "source": chunk_sources.get(chunk_id, ""),
                    "weight": round(score / max_score, 4) if max_score > 0 else 0.0,
                    "ppr_score": round(score, 6),
                }
                for chunk_id, score in ranked
            ]
        except Exception as exc:
            logger.warning("neo4j.pagerank_failed error=%s — 2-hop fallback", str(exc)[:200])
            logger.info("retrieval.graph_signal method=2hop_fallback reason=error")
            return self.get_related_chunks(seed_entity_names, corpus_id)

    def get_entity_neighbors(self, entity_name: str, corpus_id: str) -> list[str]:
        """Names of entities directly connected to ``entity_name``."""
        rows = self._run(
            """
            MATCH (e:Entity {corpus_id: $corpus_id})
            WHERE toLower(e.name) = toLower($name)
            MATCH (e)-[:RELATES_TO]-(nbr:Entity)
            RETURN DISTINCT nbr.name AS name
            LIMIT 100
            """,
            name=entity_name,
            corpus_id=corpus_id,
        )
        return [row["name"] for row in rows]

    # -------------------------------------------------------------- lifecycle
    def clear_corpus(self, corpus_id: str) -> None:
        """Delete every node (and attached edges) belonging to a corpus."""
        self._run(
            "MATCH (n {corpus_id: $corpus_id}) DETACH DELETE n",
            corpus_id=corpus_id,
        )
        logger.info("neo4j.corpus_cleared corpus_id=%s", corpus_id)

    def count_nodes(self, corpus_id: str | None = None) -> int:
        """Node count (per corpus, or database-wide when corpus_id is None)."""
        if corpus_id:
            rows = self._run(
                "MATCH (n {corpus_id: $corpus_id}) RETURN count(n) AS c", corpus_id=corpus_id
            )
        else:
            rows = self._run("MATCH (n) RETURN count(n) AS c")
        return int(rows[0]["c"]) if rows else 0

    def health_check(self) -> bool:
        """True iff the database answers a trivial query."""
        rows = self._run("RETURN 1 AS ok")
        return bool(rows) and rows[0].get("ok") == 1

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._driver = None
        # Allow reconnect attempts after an explicit close.
        self._connect_failed = False


# Shared instance: one connection pool per process. Import and use
# ``get_neo4j_store()`` rather than constructing Neo4jStore directly.
_shared_store: Neo4jStore | None = None
_shared_lock = threading.Lock()


def get_neo4j_store() -> Neo4jStore:
    global _shared_store
    if _shared_store is None:
        with _shared_lock:
            if _shared_store is None:
                _shared_store = Neo4jStore()
                atexit.register(_shared_store.close)
    return _shared_store
