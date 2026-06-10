"""Corpus isolation: each ingestion lives in its own ``corpus_id`` namespace.

Without this, a user who ingests "LSTM (ML)" and then "Quantum entanglement
(physics)" would see both topics' entities mixed into a single graph and
retrieved together for any query — silently producing cross-domain
hallucinations. The :class:`CorpusManager` keeps each corpus's chunks,
entities, relations, and graph completely separate on disk and in memory.
"""

from graphrag_plus.app.corpus.manager import CorpusManager
from graphrag_plus.app.corpus.models import CorpusBundle, CorpusMeta

__all__ = ["CorpusBundle", "CorpusManager", "CorpusMeta"]
