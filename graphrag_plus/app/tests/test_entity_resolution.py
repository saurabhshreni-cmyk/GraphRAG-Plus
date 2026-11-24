"""Tests for entity resolution: the shared planner and the local applier.

The planner had no tests before it was extracted from the Neo4j resolver. The
local applier is what keeps "Apple / Apple Inc / Apple Incorporated collapse
into one node" true on the free deployment, which has no Neo4j.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphrag_plus.app.graph.entity_resolver import (
    _EMBEDDING_THRESHOLD,
    _UNCALIBRATED_EMBEDDING_THRESHOLD,
    embedding_threshold_for,
    plan_entity_merges,
)
from graphrag_plus.app.graph.local_resolver import resolve_local_graph
from graphrag_plus.app.graph.store import GraphStore


def _rows(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"name": name, "type": etype} for name, etype in pairs]


class _Embedder:
    """Fake embedder mapping names to fixed vectors."""

    def __init__(self, vectors: dict[str, list[float]], model_name: str = "BAAI/bge-large-en-v1.5"):
        self._vectors = vectors
        self.model_name = model_name

    def available(self) -> bool:
        return True

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors.get(t, [0.0, 0.0, 1.0]) for t in texts]


class TestThresholdSelection:
    def test_bge_uses_the_measured_threshold(self, monkeypatch):
        monkeypatch.delenv("GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD", raising=False)
        assert embedding_threshold_for(_Embedder({})) == _EMBEDDING_THRESHOLD

    def test_uncalibrated_backends_get_the_stricter_bar(self, monkeypatch):
        monkeypatch.delenv("GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD", raising=False)
        remote = _Embedder({}, model_name="gemini:gemini-embedding-001")
        assert embedding_threshold_for(remote) == _UNCALIBRATED_EMBEDDING_THRESHOLD

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD", "0.88")
        assert embedding_threshold_for(_Embedder({}, model_name="jina:x")) == pytest.approx(0.88)

    def test_bad_override_falls_back(self, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD", "not-a-number")
        assert embedding_threshold_for(_Embedder({})) == _EMBEDDING_THRESHOLD


class TestPlanEntityMerges:
    def test_collapses_a_transitive_alias_chain_to_the_longest_name(self):
        plan = plan_entity_merges(
            _rows(("Apple", "ORG"), ("Apple Inc", "ORG"), ("Apple Incorporated", "ORG")), None
        )
        assert {m["canonical"] for m in plan} == {"Apple Incorporated"}
        assert {m["duplicate"] for m in plan} == {"Apple", "Apple Inc"}

    def test_a_product_is_not_folded_into_the_company(self):
        # Regression, observed live on the deployed preview: the bare "Apple"
        # is a token-prefix of both "Apple Incorporated" and "Apple Music",
        # and union-find used to chain all three into one node.
        plan = plan_entity_merges(
            _rows(
                ("Apple", "ORG"),
                ("Apple Inc", "ORG"),
                ("Apple Incorporated", "ORG"),
                ("Apple Music", "ORG"),
            ),
            None,
        )
        merged = {m["duplicate"] for m in plan}
        assert merged == {"Apple", "Apple Inc"}
        assert all(m["canonical"] == "Apple Incorporated" for m in plan)
        assert "Apple Music" not in merged

    def test_legal_suffix_variants_merge_without_a_bridging_name(self):
        # Previously "Apple Inc" and "Apple Incorporated" only joined through a
        # bare "Apple"; normalization links them directly.
        plan = plan_entity_merges(_rows(("Apple Inc", "ORG"), ("Apple Incorporated", "ORG")), None)
        assert plan == [{"duplicate": "Apple Inc", "canonical": "Apple Incorporated", "similarity": 1.0}]

    def test_ambiguous_prefix_does_not_merge(self):
        # "Google" prefixes two different products; it can't safely be either.
        plan = plan_entity_merges(
            _rows(("Google", "ORG"), ("Google Cloud", "ORG"), ("Google Maps", "ORG")), None
        )
        assert plan == []

    def test_unambiguous_prefix_still_merges(self):
        plan = plan_entity_merges(_rows(("Tim", "PERSON"), ("Tim Cook", "PERSON")), None)
        assert [(m["duplicate"], m["canonical"]) for m in plan] == [("Tim", "Tim Cook")]

    def test_spelling_variants_still_merge(self):
        plan = plan_entity_merges(_rows(("Microsoft Corp.", "ORG"), ("Microsoft Corporation", "ORG")), None)
        assert len(plan) == 1

    def test_never_merges_across_types(self):
        assert plan_entity_merges(_rows(("Apple", "ORG"), ("Apple", "PRODUCT")), None) == []

    def test_short_names_are_too_ambiguous_to_merge(self):
        assert plan_entity_merges(_rows(("AI", "OTHER"), ("Al", "OTHER")), None) == []

    def test_distinct_entities_stay_separate(self):
        assert plan_entity_merges(_rows(("Google", "ORG"), ("Microsoft", "ORG")), None) == []

    def test_embedding_pass_merges_zero_overlap_aliases(self, monkeypatch):
        monkeypatch.delenv("GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD", raising=False)
        embedder = _Embedder({"IBM": [1.0, 0.0, 0.0], "International Business Machines": [0.99, 0.01, 0.0]})
        plan = plan_entity_merges(_rows(("IBM", "ORG"), ("International Business Machines", "ORG")), embedder)
        assert plan == [
            {
                "duplicate": "IBM",
                "canonical": "International Business Machines",
                "similarity": pytest.approx(plan[0]["similarity"]),
            }
        ]

    def test_uncalibrated_backend_does_not_merge_at_the_bge_bar(self, monkeypatch):
        # 0.93 clears the bge threshold (0.92) but not the uncalibrated one
        # (0.95): a hosted backend must not inherit a bar measured elsewhere.
        # Names share no tokens, so only the embedding pass can link them.
        monkeypatch.delenv("GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD", raising=False)
        rows = _rows(("Orion", "ORG"), ("Helios Systems Group", "ORG"))
        vectors = {"Orion": [1.0, 0.0], "Helios Systems Group": [0.93, (1 - 0.93**2) ** 0.5]}
        remote = _Embedder(vectors, model_name="gemini:gemini-embedding-001")
        assert plan_entity_merges(rows, remote) == []
        assert len(plan_entity_merges(rows, _Embedder(vectors))) == 1

    def test_a_failing_embedder_leaves_string_merges_intact(self):
        class _Broken:
            model_name = "x"

            def available(self):
                return True

            def embed_batch(self, texts):
                raise RuntimeError("quota exhausted")

        plan = plan_entity_merges(_rows(("Apple", "ORG"), ("Apple Inc", "ORG")), _Broken())
        assert [m["canonical"] for m in plan] == ["Apple Inc"]

    def test_fewer_than_two_entities_is_a_no_op(self):
        assert plan_entity_merges(_rows(("Apple", "ORG")), None) == []


@pytest.fixture
def graph_store(tmp_path: Path) -> GraphStore:
    """A graph where three Apple aliases each have their own links."""
    store = GraphStore(tmp_path / "graph.json")
    g = store.graph
    g.add_node("doc1", node_type="Document")
    for chunk in ("c1", "c2", "c3"):
        g.add_node(chunk, node_type="Chunk", doc_id="doc1")
    g.add_node("ent::apple", node_type="ORG", label="Apple")
    g.add_node("ent::apple inc", node_type="ORG", label="Apple Inc")
    g.add_node("ent::apple incorporated", node_type="ORG", label="Apple Incorporated")
    # Same surface form, different type: must NOT join the ORG cluster.
    g.add_node("ent::apple watch", node_type="PRODUCT", label="Apple Watch")
    g.add_node("ent::beats", node_type="ORG", label="Beats")
    g.add_node("ent::tim cook", node_type="PERSON", label="Tim Cook")
    g.add_edge("c1", "ent::apple", key="mentions", edge_type="mentions")
    g.add_edge("c2", "ent::apple inc", key="mentions", edge_type="mentions")
    g.add_edge("c3", "ent::apple incorporated", key="mentions", edge_type="mentions")
    g.add_edge("ent::apple", "ent::beats", key="acquired@c1", edge_type="acquired", predicate="acquired")
    g.add_edge("ent::tim cook", "ent::apple inc", key="leads@c2", edge_type="leads", predicate="leads")
    # An edge between two aliases must not become a self-loop.
    g.add_edge("ent::apple", "ent::apple inc", key="same_as@c1", edge_type="same_as")
    store.save()
    return store


class TestResolveLocalGraph:
    def test_collapses_aliases_into_one_node(self, graph_store):
        stats = resolve_local_graph(graph_store, None)
        g = graph_store.graph
        assert stats["merged_count"] == 2
        assert "ent::apple" not in g
        assert "ent::apple inc" not in g
        assert "ent::apple incorporated" in g

    def test_relations_are_repointed_in_both_directions(self, graph_store):
        resolve_local_graph(graph_store, None)
        g = graph_store.graph
        assert g.has_edge("ent::apple incorporated", "ent::beats", key="acquired@c1")
        assert g.has_edge("ent::tim cook", "ent::apple incorporated", key="leads@c2")

    def test_chunk_mentions_survive_so_source_chunks_still_show(self, graph_store):
        resolve_local_graph(graph_store, None)
        mentioning = {u for u, _ in graph_store.graph.in_edges("ent::apple incorporated")}
        assert {"c1", "c2", "c3"} <= mentioning

    def test_no_self_loops_from_alias_to_alias_edges(self, graph_store):
        resolve_local_graph(graph_store, None)
        assert not graph_store.graph.has_edge("ent::apple incorporated", "ent::apple incorporated")

    def test_canonical_records_its_aliases(self, graph_store):
        resolve_local_graph(graph_store, None)
        attrs = graph_store.graph.nodes["ent::apple incorporated"]
        assert attrs["aliases"] == ["Apple", "Apple Inc"]
        assert attrs["node_type"] == "ORG"

    def test_unrelated_and_differently_typed_entities_are_untouched(self, graph_store):
        resolve_local_graph(graph_store, None)
        assert "ent::beats" in graph_store.graph
        assert "ent::tim cook" in graph_store.graph
        assert "ent::apple watch" in graph_store.graph

    def test_result_is_persisted(self, graph_store):
        resolve_local_graph(graph_store, None)
        reloaded = GraphStore(graph_store.graph_path)
        assert "ent::apple" not in reloaded.graph
        assert "ent::apple incorporated" in reloaded.graph

    def test_is_idempotent(self, graph_store):
        resolve_local_graph(graph_store, None)
        assert resolve_local_graph(graph_store, None)["merged_count"] == 0
