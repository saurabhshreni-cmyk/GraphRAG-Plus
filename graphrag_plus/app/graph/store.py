"""NetworkX graph store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

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
            self.graph.add_edge(source, target, **attrs)

    def save(self) -> None:
        """Persist graph to JSON."""
        nodes = [{"id": node_id, **attrs} for node_id, attrs in self.graph.nodes(data=True)]
        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            edges.append({"source": source, "target": target, **attrs})
        dump_json(self.graph_path, {"nodes": nodes, "edges": edges})

    def upsert_from_extractions(
        self,
        documents: List[Document],
        chunks: List[Chunk],
        entities: List[Entity],
        relations: List[Relation],
    ) -> Tuple[List[str], List[str]]:
        """Update graph and return changed nodes/edges."""
        changed_nodes: List[str] = []
        changed_edges: List[str] = []
        for document in documents:
            self.graph.add_node(document.doc_id, node_type="Document", source=document.source)
            changed_nodes.append(document.doc_id)
        for chunk in chunks:
            self.graph.add_node(chunk.chunk_id, node_type="Chunk", doc_id=chunk.doc_id, timestamp=chunk.timestamp)
            self.graph.add_edge(chunk.doc_id, chunk.chunk_id, edge_type="contains")
            changed_nodes.append(chunk.chunk_id)
            changed_edges.append(f"{chunk.doc_id}->{chunk.chunk_id}:contains")
        for entity in entities:
            entity_id = f"ent::{entity.text.lower()}"
            self.graph.add_node(entity_id, node_type=entity.entity_type, label=entity.text)
            self.graph.add_edge(entity.source_chunk_id, entity_id, edge_type="mentions", confidence=entity.confidence)
            changed_nodes.append(entity_id)
            changed_edges.append(f"{entity.source_chunk_id}->{entity_id}:mentions")
        for relation in relations:
            subj_id = f"ent::{relation.subject.lower()}"
            obj_id = f"ent::{relation.obj.lower()}"
            self.graph.add_node(subj_id, node_type="Entity", label=relation.subject)
            self.graph.add_node(obj_id, node_type="Entity", label=relation.obj)
            edge_type = relation.stance if relation.stance in {"supports", "contradicts"} else relation.predicate
            self.graph.add_edge(
                subj_id,
                obj_id,
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

    def neighbors(self, node_id: str) -> List[Dict[str, str]]:
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

    def current_snapshot(self) -> Dict[str, List[Dict[str, str]]]:
        """Return graph payload for versioning."""
        nodes = [{"id": node_id, **attrs} for node_id, attrs in self.graph.nodes(data=True)]
        edges = [{"source": s, "target": t, **a} for s, t, a in self.graph.edges(data=True)]
        return {"nodes": nodes, "edges": edges}

