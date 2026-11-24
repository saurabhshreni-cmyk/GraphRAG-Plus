"""Measure relevance floors and the entity-merge threshold for an embedding backend.

The retrieval floors in ``app/retrieval/service.py`` (0.45 / 0.55) and the
entity-resolution threshold (0.92) were measured on ``BAAI/bge-large-en-v1.5``.
Cosine distributions differ between models, so a different backend - such as
the hosted Gemini or Jina API the deployment uses - needs its own numbers.
This script measures them and prints the environment variables to set.

Usage (from the repo root, with the backend's key in the environment)::

    $env:GEMINI_API_KEY = "..."            # or JINA_API_KEY
    $env:GRAPHRAG_EMBEDDING_BACKEND = "remote"
    $env:PYTHONPATH = "."
    python graphrag_plus/scripts/calibrate_backend.py

It makes three small embedding calls (well inside any free tier).

Method: embed on-topic and off-topic queries against a fixed passage set and
take each query's best passage cosine. A usable backend separates the two
groups; the floors are placed inside the gap - MIN 35% of the way up from the
highest off-topic score, STRONG 65% - so off-topic queries fall below MIN with
margin while on-topic ones clear STRONG. The merge threshold sits midway
between the most similar distinct pair and the least similar true alias, and
never below 0.92 (the bar measured on bge-large).
"""

from __future__ import annotations

import sys

import numpy as np
from graphrag_plus.app.embeddings.embedder import get_embedder

PASSAGES = [
    "A knowledge graph represents entities as nodes and the relationships between them as edges.",
    "Graph databases such as Neo4j store nodes, relationships and their properties natively.",
    "Retrieval-augmented generation grounds a language model's answer in retrieved documents.",
    "Entity resolution merges different surface forms of the same real-world entity into one node.",
    "Personalized PageRank ranks graph nodes by their proximity to a set of seed nodes.",
    "Hybrid retrieval combines keyword search like BM25 with dense vector similarity.",
]
ON_TOPIC = [
    "What is a knowledge graph?",
    "How does retrieval augmented generation work?",
    "Which database stores graph relationships?",
    "How are duplicate entities merged?",
    "What does Personalized PageRank do?",
    "How is BM25 combined with vector search?",
]
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the 1998 football World Cup?",
    "What is the boiling point of water at sea level?",
    "How many legs does a spider have?",
    "What is the best way to learn the guitar?",
]
ALIAS_PAIRS = [
    ("IBM", "International Business Machines"),
    ("Microsoft", "Microsoft Corporation"),
    ("NYC", "New York City"),
    ("Google", "Google LLC"),
]
DISTINCT_PAIRS = [
    ("Google", "Microsoft"),
    ("Apple", "Samsung"),
    ("Paris", "London"),
    ("Neo4j", "PostgreSQL"),
]

_MIN_GAP_FRACTION = 0.35
_STRONG_GAP_FRACTION = 0.65
# Never recommend a merge bar looser than the 0.92 measured on bge-large. On
# that model alias and distinct name pairs overlap (alias 0.68-0.87, distinct
# 0.58-0.80), so the embedding pass is a narrow safety net and the string /
# token-prefix passes do most of the merging; loosening it buys false merges.
_MERGE_THRESHOLD_FLOOR = 0.92


def _unit(rows: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _best_scores(embedder, queries: list[str], passage_matrix: np.ndarray) -> list[float]:
    query_matrix = _unit([embedder.embed_query(q) for q in queries])
    return (query_matrix @ passage_matrix.T).max(axis=1).tolist()


def main() -> int:
    embedder = get_embedder()
    if not embedder.available():
        print("No embedding backend is available. Set GEMINI_API_KEY or JINA_API_KEY.")
        return 1
    print(f"Backend: {embedder.model_name}  (dim {embedder.dimension})\n")

    passages = embedder.embed_batch(PASSAGES)
    if len(passages) != len(PASSAGES):
        print("Embedding call failed - check the key and quota.")
        return 1
    passage_matrix = _unit(passages)

    on = _best_scores(embedder, ON_TOPIC, passage_matrix)
    off = _best_scores(embedder, OFF_TOPIC, passage_matrix)
    on_min, off_max = min(on), max(off)
    gap = on_min - off_max
    print(f"On-topic best cosine:  {min(on):.3f} - {max(on):.3f}")
    print(f"Off-topic best cosine: {min(off):.3f} - {max(off):.3f}")

    exit_code = 0
    if gap <= 0:
        print("\nWARNING: on- and off-topic scores overlap; no floor separates them.")
        print("Keep the defaults and rely on the term-overlap gate.")
        exit_code = 2
        min_floor = strong_floor = None
    else:
        min_floor = off_max + _MIN_GAP_FRACTION * gap
        strong_floor = off_max + _STRONG_GAP_FRACTION * gap
        print(f"Separation gap:        {gap:.3f}")

    names = sorted({n for pair in ALIAS_PAIRS + DISTINCT_PAIRS for n in pair})
    name_vectors = embedder.embed_batch(names)
    if len(name_vectors) != len(names):
        print("Embedding call for entity names failed.")
        return 1
    index = {name: i for i, name in enumerate(names)}
    name_matrix = _unit(name_vectors)

    def _cos(a: str, b: str) -> float:
        return float(name_matrix[index[a]] @ name_matrix[index[b]])

    alias = [_cos(a, b) for a, b in ALIAS_PAIRS]
    distinct = [_cos(a, b) for a, b in DISTINCT_PAIRS]
    print(f"\nAlias pair cosine:     {min(alias):.3f} - {max(alias):.3f}")
    print(f"Distinct pair cosine:  {min(distinct):.3f} - {max(distinct):.3f}")
    if max(distinct) >= min(alias):
        merge_threshold = max(_MERGE_THRESHOLD_FLOOR, max(distinct) + 0.02)
        print("Aliases and distinct names overlap; using a conservative threshold.")
    else:
        merge_threshold = max(_MERGE_THRESHOLD_FLOOR, (max(distinct) + min(alias)) / 2)

    print("\n--- Set these on the deployment (Vercel -> Settings -> Environment Variables) ---")
    if min_floor is not None and strong_floor is not None:
        print(f"GRAPHRAG_MIN_COSINE_SEMANTIC={min_floor:.2f}")
        print(f"GRAPHRAG_STRONG_COSINE_SEMANTIC={strong_floor:.2f}")
    print(f"GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD={merge_threshold:.2f}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
