"""Entity resolution applied to the local NetworkX graph.

Neo4j is optional, and the free deployment runs without it — so the
"Apple / Apple Inc / Apple Incorporated collapse into one node" guarantee has
to hold on the local graph too, or the visualization and the graph retrieval
signal fragment exactly the way entity resolution exists to prevent.

Planning is shared with the Neo4j resolver
(:func:`~graphrag_plus.app.graph.entity_resolver.plan_entity_merges`); this
module only applies the plan to a :class:`GraphStore`.

Merging a duplicate into its canonical node:

* every incoming and outgoing edge is re-pointed at the canonical node,
  keeping its key, so chunk ``mentions`` links survive and the entity panel
  still shows the duplicate's source chunks;
* an edge the canonical node already has under the same key is kept rather
  than duplicated, and edges between the two merged nodes are dropped instead
  of becoming self-loops;
* the canonical node records the duplicate's label under ``aliases``;
* the duplicate node is removed.
"""

from __future__ import annotations

import time
from typing import Any

from graphrag_plus.app.graph.entity_resolver import plan_entity_merges
from graphrag_plus.app.graph.store import GraphStore
from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

_ENTITY_PREFIX = "ent::"


def _entity_rows(graph: Any) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Entity ``{"name","type"}`` rows plus a name -> node id map.

    Types follow the Neo4j convention (``Entity`` -> ``OTHER``) so both
    backends feed the planner identical inputs and therefore merge the same
    pairs.
    """
    rows: list[dict[str, str]] = []
    id_of: dict[str, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        if not (isinstance(node_id, str) and node_id.startswith(_ENTITY_PREFIX)):
            continue
        name = str(attrs.get("label") or node_id[len(_ENTITY_PREFIX) :])
        node_type = str(attrs.get("node_type") or "OTHER")
        rows.append({"name": name, "type": "OTHER" if node_type == "Entity" else node_type})
        id_of[name] = node_id
    return rows, id_of


def _merge_into(graph: Any, duplicate_id: str, canonical_id: str) -> None:
    """Fold ``duplicate_id`` into ``canonical_id`` in place on ``graph``."""
    for source, _target, key, attrs in list(graph.in_edges(duplicate_id, keys=True, data=True)):
        if source in (canonical_id, duplicate_id):
            continue
        if not graph.has_edge(source, canonical_id, key=key):
            graph.add_edge(source, canonical_id, key=key, **attrs)
    for _source, target, key, attrs in list(graph.out_edges(duplicate_id, keys=True, data=True)):
        if target in (canonical_id, duplicate_id):
            continue
        if not graph.has_edge(canonical_id, target, key=key):
            graph.add_edge(canonical_id, target, key=key, **attrs)

    # No type reconciliation is needed: the planner only merges same-type
    # entities, so duplicate and canonical already agree on type.
    dup_attrs = graph.nodes[duplicate_id]
    canon_attrs = graph.nodes[canonical_id]
    aliases = set(canon_attrs.get("aliases") or [])
    aliases.update(dup_attrs.get("aliases") or [])
    aliases.add(str(dup_attrs.get("label") or duplicate_id[len(_ENTITY_PREFIX) :]))
    aliases.discard(str(canon_attrs.get("label") or ""))
    canon_attrs["aliases"] = sorted(aliases)

    graph.remove_node(duplicate_id)


def resolve_local_graph(graph_store: GraphStore, embedder: Any) -> dict[str, Any]:
    """Deduplicate entity nodes in ``graph_store`` and persist the result.

    Returns the same stats shape as ``EntityResolver.resolve_corpus``. Never
    raises: resolution is an enhancement, and a failure must not block the
    ingest that triggered it.
    """
    start = time.perf_counter()
    result: dict[str, Any] = {"merged_count": 0, "canonical_pairs": [], "time_taken_s": 0.0}
    try:
        graph = graph_store.graph
        rows, id_of = _entity_rows(graph)
        plan = plan_entity_merges(rows, embedder)
        for merge in plan:
            duplicate_id = id_of.get(merge["duplicate"])
            canonical_id = id_of.get(merge["canonical"])
            if duplicate_id is None or canonical_id is None or duplicate_id == canonical_id:
                continue
            if duplicate_id not in graph or canonical_id not in graph:
                continue
            _merge_into(graph, duplicate_id, canonical_id)
            result["canonical_pairs"].append(merge)
            logger.info(
                "local_resolver.merged [%s] -> [%s] (similarity: %.3f)",
                merge["duplicate"],
                merge["canonical"],
                merge["similarity"],
            )
        result["merged_count"] = len(result["canonical_pairs"])
        if result["merged_count"]:
            graph_store.save()
    except Exception as exc:
        logger.warning("local_resolver.failed error=%s", str(exc)[:200])
    result["time_taken_s"] = round(time.perf_counter() - start, 2)
    return result


__all__ = ["resolve_local_graph"]
