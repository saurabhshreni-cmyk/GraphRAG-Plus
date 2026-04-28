"""NetworkX graph store."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from graphrag_plus.app.extraction.models import Entity, Relation
from graphrag_plus.app.ingestion.models import Chunk, Document
from graphrag_plus.app.utils.io_utils import dump_json, load_json


class GraphStore:
    """Persistent heterogeneous graph."""

    def __init__(self, graph_path: Path):
        self.graph_path = graph_path
        self.graph = nx.MultiDiGraph()
        self._load()

    def _load(self) -> None:
        payload = load_json(self.graph_path, default=None)
        if not payload:
            return
        for node in payload.get("nodes", []):
            attrs = dict(node)
            node_id = attrs.pop("id")
            self.graph.add_node(node_id, **attrs)
        for edge in payload.get("edges", []):
            attrs = dict(edge)
            source = attrs.pop("source")
            target = attrs.pop("target")
            # Preserve the deterministic edge key across persistence so that
            # repeated ingests don't bypass the dedupe check just because the
            # process restarted.
            edge_key = attrs.pop("key", None)
            if edge_key is None:
                self.graph.add_edge(source, target, **attrs)
            else:
                self.graph.add_edge(source, target, key=edge_key, **attrs)

    def save(self) -> None:
        """Persist graph to JSON. Includes edge keys so reload is idempotent."""
        nodes = [{"id": node_id, **attrs} for node_id, attrs in self.graph.nodes(data=True)]
        edges = []
        for source, target, edge_key, attrs in self.graph.edges(keys=True, data=True):
            edges.append({"source": source, "target": target, "key": edge_key, **attrs})
        dump_json(self.graph_path, {"nodes": nodes, "edges": edges})

    def upsert_from_extractions(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        entities: list[Entity],
        relations: list[Relation],
    ) -> tuple[list[str], list[str]]:
        """Update graph and return changed nodes/edges.

        Uses deterministic edge keys per ``(source, target, edge_type [, source_chunk])``
        so re-ingesting the same content updates existing edges instead of
        appending parallel ones to the underlying ``MultiDiGraph``. Without
        this, repeated ingests inflate edge counts and skew downstream graph
        statistics.
        """
        changed_nodes: list[str] = []
        changed_edges: list[str] = []
        for document in documents:
            self.graph.add_node(document.doc_id, node_type="Document", source=document.source)
            changed_nodes.append(document.doc_id)
        for chunk in chunks:
            self.graph.add_node(
                chunk.chunk_id, node_type="Chunk", doc_id=chunk.doc_id, timestamp=chunk.timestamp
            )
            self.graph.add_edge(chunk.doc_id, chunk.chunk_id, key="contains", edge_type="contains")
            changed_nodes.append(chunk.chunk_id)
            changed_edges.append(f"{chunk.doc_id}->{chunk.chunk_id}:contains")
        for entity in entities:
            entity_id = f"ent::{entity.text.lower()}"
            self.graph.add_node(entity_id, node_type=entity.entity_type, label=entity.text)
            self.graph.add_edge(
                entity.source_chunk_id,
                entity_id,
                key="mentions",
                edge_type="mentions",
                confidence=entity.confidence,
            )
            changed_nodes.append(entity_id)
            changed_edges.append(f"{entity.source_chunk_id}->{entity_id}:mentions")
        for relation in relations:
            subj_id = f"ent::{relation.subject.lower()}"
            obj_id = f"ent::{relation.obj.lower()}"
            self.graph.add_node(subj_id, node_type="Entity", label=relation.subject)
            self.graph.add_node(obj_id, node_type="Entity", label=relation.obj)
            edge_type = (
                relation.stance if relation.stance in {"supports", "contradicts"} else relation.predicate
            )
            # Include the source chunk in the key so the same predicate can
            # legitimately appear once per source without duplicating across
            # repeated ingests of the same source.
            edge_key = f"{edge_type}@{relation.source_chunk_id}"
            self.graph.add_edge(
                subj_id,
                obj_id,
                key=edge_key,
                edge_type=edge_type,
                predicate=relation.predicate,
                confidence=relation.confidence,
                source_chunk_id=relation.source_chunk_id,
                timestamp=relation.timestamp,
            )
            changed_nodes.extend([subj_id, obj_id])
            changed_edges.append(f"{subj_id}->{obj_id}:{edge_type}")
        self.save()
        return sorted(set(changed_nodes)), sorted(set(changed_edges))

    def neighbors(self, node_id: str) -> list[dict[str, str]]:
        """Get neighbors for node."""
        if node_id not in self.graph:
            return []
        rows = []
        for neighbor in self.graph.neighbors(node_id):
            rows.append({"id": neighbor, **self.graph.nodes[neighbor]})
        return rows

    def export_graphml(self, target_path: Path) -> None:
        """Export GraphML."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, target_path)

    def current_snapshot(self) -> dict[str, list[dict[str, str]]]:
        """Return graph payload for versioning and visualization."""
        nodes = [{"id": node_id, **attrs} for node_id, attrs in self.graph.nodes(data=True)]
        edges = [
            {"source": s, "target": t, "key": k, **a} for s, t, k, a in self.graph.edges(keys=True, data=True)
        ]
        return {"nodes": nodes, "edges": edges}
