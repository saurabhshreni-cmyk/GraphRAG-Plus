"""Tests for the NetworkX fallback that serves the v2 graph endpoints when
Neo4j is unavailable (graphrag_plus.app.graph.local_graph)."""

from __future__ import annotations

from pathlib import Path

from graphrag_plus.app.extraction.models import Entity, Relation
from graphrag_plus.app.graph import local_graph as lg
from graphrag_plus.app.graph.store import GraphStore
from graphrag_plus.app.ingestion.models import Chunk, Document


def _seed_store(tmp_path: Path) -> tuple[GraphStore, list[dict]]:
    """Build a tiny corpus graph: Google (ORG) --associated_with--> Bing (ORG)."""
    store = GraphStore(tmp_path / "graph.json")
    documents = [Document(doc_id="doc1", source="wiki", text="...", metadata={})]
    chunks = [
        Chunk(chunk_id="doc1_ch0", doc_id="doc1", text="Google and Bing are search engines.", start=0, end=35, timestamp=None)
    ]
    entities = [
        Entity(text="Google", entity_type="Org", confidence=0.9, method="ner", source_chunk_id="doc1_ch0"),
        Entity(text="Bing", entity_type="Org", confidence=0.9, method="ner", source_chunk_id="doc1_ch0"),
    ]
    relations = [
        Relation(
            subject="Google",
            predicate="associated_with",
            obj="Bing",
            stance="neutral",
            confidence=0.8,
            method="llm",
            source_chunk_id="doc1_ch0",
        )
    ]
    store.upsert_from_extractions(documents, chunks, entities, relations)
    chunk_dicts = [{"chunk_id": c.chunk_id, "text": c.text, "doc_id": c.doc_id, "source": "wiki"} for c in chunks]
    return store, chunk_dicts


def test_graph_full_returns_typed_nodes_and_edges(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    result = lg.graph_full(store)
    assert result is not None
    labels = {n["label"] for n in result["nodes"]}
    assert {"Google", "Bing"} <= labels
    # The relation-created nodes must keep their specific ORG type, not the
    # generic "Entity"/"OTHER" (regression guard for the type-preservation fix).
    types = {n["label"]: n["type"] for n in result["nodes"]}
    assert types["Google"] == "Org"
    assert any(e["relation"] == "associated_with" for e in result["edges"])
    assert result["stats"]["returned_nodes"] == len(result["nodes"])


def test_graph_full_none_for_empty_corpus(tmp_path: Path) -> None:
    empty = GraphStore(tmp_path / "empty.json")
    assert lg.graph_full(empty) is None


def test_graph_stats_shape(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    stats = lg.graph_stats(store)
    assert stats is not None
    assert stats["total_chunks"] == 1
    assert stats["nodes_by_type"].get("Org") == 2
    assert "associated_with" in stats["relationships_by_type"]
    assert stats["top_connected_entities"][0]["name"] in {"Google", "Bing"}


def test_graph_entity_neighbours_and_chunks(tmp_path: Path) -> None:
    store, chunks = _seed_store(tmp_path)
    result = lg.graph_entity(store, "google", chunks)
    assert result is not None
    assert result["entity"]["name"] == "Google"
    assert result["entity"]["type"] == "Org"
    nbr_names = {n["name"] for n in result["neighbors"]}
    assert "Bing" in nbr_names
    assert result["chunks"] and result["chunks"][0]["text"].startswith("Google and Bing")


def test_graph_entity_missing_returns_none(tmp_path: Path) -> None:
    store, chunks = _seed_store(tmp_path)
    assert lg.graph_entity(store, "Nonexistent", chunks) is None


def test_query_path_seeds_and_edges(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    result = lg.graph_query_path(store, ["Google"], "tell me about Google")
    assert "Google" in result["seed_nodes"]
    assert "Bing" in result["neighbor_nodes"]
    assert any(e["source"] == "Google" and e["target"] == "Bing" for e in result["edges"])
