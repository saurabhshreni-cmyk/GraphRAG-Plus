"""NetworkX-backed fallback for the v2 graph-exploration endpoints.

The four ``/graph/{corpus_id}/...`` endpoints (``full``, ``stats``,
``entity``, ``query-path``) are normally served from Neo4j. Neo4j is an
*optional* backend: every ingest also persists the same entities, relations
and mentions into the per-corpus NetworkX :class:`GraphStore`
(``graph.json``). When Neo4j is unavailable this module serves the identical
response shapes straight from that local graph, so the knowledge-graph
visualization works fully offline with no cloud dependency.

The functions here mirror the Cypher queries in ``api/main.py`` field-for-field
so the frontend cannot tell which backend answered.
"""

from __future__ import annotations

from typing import Any

from graphrag_plus.app.graph.store import GraphStore

# Node ids for entities are ``ent::<lowercased label>`` (see
# GraphStore.upsert_from_extractions). Chunk/Document nodes never use it.
_ENTITY_PREFIX = "ent::"

# Edge types that are structural, not entity-to-entity relations.
_MENTION_EDGE = "mentions"
_CONTAINS_EDGE = "contains"


def _is_entity(node_id: str) -> bool:
    return isinstance(node_id, str) and node_id.startswith(_ENTITY_PREFIX)


def _label_of(graph: Any, node_id: str) -> str:
    """Human-readable name for an entity node (falls back to the id tail)."""
    attrs = graph.nodes.get(node_id, {})
    label = attrs.get("label")
    if label:
        return str(label)
    return node_id[len(_ENTITY_PREFIX) :] if _is_entity(node_id) else str(node_id)


def _type_of(graph: Any, node_id: str) -> str:
    """Entity type, normalized to match the Neo4j ``coalesce(type,'OTHER')``.

    The generic ``Entity`` node_type (assigned when extraction produced no
    finer label) maps to ``OTHER`` so the frontend palette colours it the same
    way it would a Neo4j entity with no type.
    """
    node_type = graph.nodes.get(node_id, {}).get("node_type") or "OTHER"
    return "OTHER" if node_type == "Entity" else str(node_type)


def _entity_edges(graph: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """All entity→entity edges (typed relations + co-occurrence)."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for source, target, attrs in graph.edges(data=True):
        if _is_entity(source) and _is_entity(target):
            out.append((source, target, attrs))
    return out


def _relation_name(attrs: dict[str, Any]) -> str:
    return str(attrs.get("predicate") or attrs.get("edge_type") or "related_to")


def graph_full(graph_store: GraphStore, max_nodes: int = 500) -> dict[str, Any] | None:
    """Local equivalent of ``GET /graph/{cid}/full``.

    Returns ``None`` when the corpus has no entity nodes, so the caller can
    raise the same 404 the Neo4j path raises.
    """
    graph = graph_store.graph
    entity_ids = [n for n in graph.nodes if _is_entity(n)]
    if not entity_ids:
        return None

    max_nodes = max(1, min(max_nodes, 500))
    # Degree over the multigraph counts relation + mention + co-occur edges,
    # matching Neo4j's (e)-[r]-() degree.
    ranked = sorted(entity_ids, key=graph.degree, reverse=True)
    kept_ids = ranked[:max_nodes]
    kept = set(kept_ids)

    nodes = [
        {
            "id": _label_of(graph, nid),
            "label": _label_of(graph, nid),
            "type": _type_of(graph, nid),
            "description": str(graph.nodes[nid].get("description", "")),
            "connection_count": graph.degree(nid),
        }
        for nid in kept_ids
    ]

    # Deduplicate parallel edges (MultiDiGraph) into one line per
    # (source, target, relation), keeping the strongest confidence.
    edge_map: dict[tuple[str, str, str], float] = {}
    total_edges = 0
    for source, target, attrs in _entity_edges(graph):
        total_edges += 1
        if source not in kept or target not in kept:
            continue
        s, t = _label_of(graph, source), _label_of(graph, target)
        rel = _relation_name(attrs)
        weight = float(attrs.get("confidence", attrs.get("weight", 0.5)) or 0.5)
        key = (s, t, rel)
        if weight > edge_map.get(key, -1.0):
            edge_map[key] = weight
    edges = [
        {"source": s, "target": t, "relation": rel, "weight": w}
        for (s, t, rel), w in edge_map.items()
    ]

    type_counts: dict[str, int] = {}
    for nid in kept_ids:
        etype = _type_of(graph, nid)
        type_counts[etype] = type_counts.get(etype, 0) + 1

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(entity_ids),
            "total_edges": total_edges,
            "entity_types": type_counts,
            "returned_nodes": len(nodes),
        },
    }


def graph_stats(graph_store: GraphStore) -> dict[str, Any] | None:
    """Local equivalent of ``GET /graph/{cid}/stats``."""
    graph = graph_store.graph
    entity_ids = [n for n in graph.nodes if _is_entity(n)]
    chunk_count = sum(1 for _, a in graph.nodes(data=True) if a.get("node_type") == "Chunk")
    if not entity_ids and not chunk_count:
        return None

    nodes_by_type: dict[str, int] = {}
    for nid in entity_ids:
        etype = _type_of(graph, nid)
        nodes_by_type[etype] = nodes_by_type.get(etype, 0) + 1

    rels_by_type: dict[str, int] = {}
    for _s, _t, attrs in _entity_edges(graph):
        rel = _relation_name(attrs)
        rels_by_type[rel] = rels_by_type.get(rel, 0) + 1
    rels_by_type = dict(
        sorted(rels_by_type.items(), key=lambda kv: kv[1], reverse=True)[:25]
    )

    top_entities = sorted(entity_ids, key=graph.degree, reverse=True)[:10]
    top_connected = [
        {
            "name": _label_of(graph, nid),
            "type": _type_of(graph, nid),
            "degree": graph.degree(nid),
        }
        for nid in top_entities
    ]

    return {
        "corpus_id": graph_store.graph_path.parent.name,
        "nodes_by_type": dict(sorted(nodes_by_type.items(), key=lambda kv: kv[1], reverse=True)),
        "relationships_by_type": rels_by_type,
        "total_chunks": chunk_count,
        "top_connected_entities": top_connected,
    }


def graph_entity(
    graph_store: GraphStore,
    entity_name: str,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Local equivalent of ``GET /graph/{cid}/entity/{name}``.

    ``chunks`` is the corpus's chunk list (from ``chunks.json``); used to
    resolve chunk text/source for the mentions panel.
    """
    graph = graph_store.graph
    target_id = f"{_ENTITY_PREFIX}{entity_name.lower()}"
    if target_id not in graph:
        # Case-insensitive scan by label as a fallback.
        match = next(
            (n for n in graph.nodes if _is_entity(n) and _label_of(graph, n).lower() == entity_name.lower()),
            None,
        )
        if match is None:
            return None
        target_id = match

    canonical = _label_of(graph, target_id)
    meta = {
        "name": canonical,
        "type": _type_of(graph, target_id),
        "description": str(graph.nodes[target_id].get("description", "")),
        "confidence": float(graph.nodes[target_id].get("confidence", 0.0) or 0.0),
    }

    # Neighbours: entity→entity edges in either direction.
    neighbours: dict[tuple[str, str], dict[str, Any]] = {}
    for _source, target, attrs in graph.out_edges(target_id, data=True):
        if _is_entity(target):
            name = _label_of(graph, target)
            neighbours[(name, "out")] = {
                "name": name,
                "type": _type_of(graph, target),
                "relation": _relation_name(attrs),
                "direction": "out",
            }
    for source, _target, attrs in graph.in_edges(target_id, data=True):
        if _is_entity(source):
            name = _label_of(graph, source)
            neighbours[(name, "in")] = {
                "name": name,
                "type": _type_of(graph, source),
                "relation": _relation_name(attrs),
                "direction": "in",
            }

    # Chunks that mention this entity (in-edges of type ``mentions`` from
    # Chunk nodes). Resolve text/source from the chunk list.
    chunk_text: dict[str, dict[str, Any]] = {}
    if chunks:
        for c in chunks:
            cid = c.get("chunk_id")
            if cid:
                chunk_text[cid] = c
    mention_chunks: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for source, _t, attrs in graph.in_edges(target_id, data=True):
        if attrs.get("edge_type") == _MENTION_EDGE and not _is_entity(source):
            if source in seen_chunks:
                continue
            seen_chunks.add(source)
            info = chunk_text.get(source, {})
            mention_chunks.append(
                {
                    "chunk_id": source,
                    "text": str(info.get("text", "")),
                    "source": str(info.get("source", info.get("doc_id", ""))),
                }
            )
        if len(mention_chunks) >= 20:
            break

    return {
        "entity": meta,
        "neighbors": list(neighbours.values())[:50],
        "chunks": mention_chunks,
    }


def graph_query_path(
    graph_store: GraphStore, seed_names: list[str], query: str
) -> dict[str, Any]:
    """Local equivalent of ``GET /graph/{cid}/query-path``.

    ``seed_names`` are the query's candidate entity strings (spaCy entities or
    content tokens) computed by the caller — identical to the Neo4j path.
    """
    graph = graph_store.graph
    lower_seeds = [s.lower() for s in seed_names if s]

    seed_ids: list[str] = []
    seed_node_ids: set[str] = set()
    for nid in graph.nodes:
        if not _is_entity(nid):
            continue
        label = _label_of(graph, nid).lower()
        if any(s == label or s in label for s in lower_seeds):
            seed_ids.append(_label_of(graph, nid))
            seed_node_ids.add(nid)
        if len(seed_ids) >= 25:
            break

    edge_map: dict[tuple[str, str, str], bool] = {}
    for node_id in seed_node_ids:
        for source, target, attrs in graph.edges(node_id, data=True):
            if _is_entity(source) and _is_entity(target):
                key = (_label_of(graph, source), _label_of(graph, target), _relation_name(attrs))
                edge_map[key] = True
        for source, target, attrs in graph.in_edges(node_id, data=True):
            if _is_entity(source) and _is_entity(target):
                key = (_label_of(graph, source), _label_of(graph, target), _relation_name(attrs))
                edge_map[key] = True
        if len(edge_map) >= 100:
            break
    edges = [{"source": s, "target": t, "relation": rel} for (s, t, rel) in edge_map]

    neighbour_ids = sorted(
        ({e["target"] for e in edges} | {e["source"] for e in edges}) - set(seed_ids)
    )
    return {
        "query": query,
        "extracted_entities": seed_names,
        "seed_nodes": seed_ids,
        "neighbor_nodes": neighbour_ids,
        "edges": edges,
    }
