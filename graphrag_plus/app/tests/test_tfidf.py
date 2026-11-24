"""Tests for the dependency-free TF-IDF index.

The equivalence test is the point of this file: it pins the NumPy
implementation to the ``scikit-learn`` behaviour it replaced, so the relevance
floors in ``service.py`` keep the meaning they were measured with. sklearn is a
dev-only dependency (it is deliberately absent from the deployed bundle), so
that test skips when it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphrag_plus.app.retrieval.tfidf import ENGLISH_STOP_WORDS, TfidfIndex, tokenize

DOCS = [
    "A knowledge graph stores entities and the relationships between them.",
    "Knowledge graphs power semantic search and question answering systems.",
    "Retrieval augmented generation grounds a language model in retrieved text.",
    "The capital of France is Paris, a city on the river Seine.",
    "Graph databases such as Neo4j store nodes and edges with properties.",
]


class TestTokenize:
    def test_lowercases_and_strips_punctuation(self):
        assert tokenize("Knowledge-Graph, stores!") == ["knowledge", "graph", "stores"]

    def test_drops_stopwords(self):
        # "the", "of", "and" are all in the vendored list.
        assert tokenize("the capital of France and Spain") == ["capital", "france", "spain"]

    def test_drops_single_character_tokens(self):
        assert tokenize("a b ab") == ["ab"]

    def test_handles_empty_and_none_like_input(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_underscores_stay_within_one_token(self):
        assert tokenize("corpus_demo_nova") == ["corpus_demo_nova"]


class TestStopWords:
    def test_has_the_sklearn_list_size(self):
        # sklearn's ENGLISH_STOP_WORDS has exactly 318 entries; a mismatch here
        # means the vendored copy drifted.
        assert len(ENGLISH_STOP_WORDS) == 318

    def test_is_immutable(self):
        assert isinstance(ENGLISH_STOP_WORDS, frozenset)


class TestTfidfIndex:
    def test_empty_corpus_is_empty_and_scores_zero(self):
        index = TfidfIndex()
        index.fit([])
        assert index.is_empty
        assert index.cosine("anything").shape == (0,)

    def test_identical_text_scores_highest(self):
        index = TfidfIndex()
        index.fit(DOCS)
        scores = index.cosine("knowledge graph entities relationships")
        assert scores.argmax() == 0

    def test_off_topic_query_scores_all_zero(self):
        index = TfidfIndex()
        index.fit(DOCS)
        # No in-vocabulary term: every document must score exactly zero so the
        # relevance floor rejects them rather than ranking noise.
        scores = index.cosine("photosynthesis chlorophyll")
        assert np.all(scores == 0.0)

    def test_stopword_only_query_scores_all_zero(self):
        index = TfidfIndex()
        index.fit(DOCS)
        assert np.all(index.cosine("tell me about the") == 0.0)

    def test_scores_are_bounded_cosines(self):
        index = TfidfIndex()
        index.fit(DOCS)
        scores = index.cosine("graph databases nodes edges")
        assert scores.shape == (len(DOCS),)
        assert np.all(scores >= -1e-9)
        assert np.all(scores <= 1.0 + 1e-9)

    def test_refit_replaces_previous_corpus(self):
        index = TfidfIndex()
        index.fit(DOCS)
        index.fit(["only one document about turbines"])
        assert index.cosine("turbines").shape == (1,)

    def test_document_with_only_stopwords_does_not_break_scoring(self):
        index = TfidfIndex()
        index.fit(["the and of", "turbines spin"])
        scores = index.cosine("turbines")
        assert scores[0] == 0.0
        assert scores[1] > 0.0


class TestSklearnEquivalence:
    """Pin the implementation to the sklearn behaviour it replaced."""

    @staticmethod
    def _sklearn_cosines(docs: list[str], query: str) -> np.ndarray:
        sklearn_text = pytest.importorskip(
            "sklearn.feature_extraction.text",
            reason="sklearn is a dev-only dependency; absent from the deployed bundle",
        )
        vectorizer = sklearn_text.TfidfVectorizer(
            lowercase=True,
            stop_words=list(sklearn_text.ENGLISH_STOP_WORDS),
            token_pattern=r"(?u)\b[A-Za-z0-9_]{2,}\b",
        )
        matrix = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([query])
        return np.asarray((matrix @ query_vec.T).todense()).ravel()

    @pytest.mark.parametrize(
        "query",
        [
            "knowledge graph",
            "what is a knowledge graph",
            "capital of France",
            "graph databases nodes edges properties",
            "retrieval augmented generation language model",
            "semantic search question answering",
        ],
    )
    def test_cosines_match_sklearn(self, query):
        expected = self._sklearn_cosines(DOCS, query)
        index = TfidfIndex()
        index.fit(DOCS)
        actual = index.cosine(query)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)

    def test_ranking_matches_sklearn_on_a_larger_corpus(self):
        docs = DOCS * 7 + [f"filler document number {i} about unrelated matters" for i in range(20)]
        query = "knowledge graph entities"
        expected = self._sklearn_cosines(docs, query)
        index = TfidfIndex()
        index.fit(docs)
        np.testing.assert_allclose(index.cosine(query), expected, rtol=1e-9, atol=1e-12)

    def test_stopword_list_matches_sklearn_exactly(self):
        sklearn_text = pytest.importorskip(
            "sklearn.feature_extraction.text",
            reason="sklearn is a dev-only dependency; absent from the deployed bundle",
        )
        assert frozenset(sklearn_text.ENGLISH_STOP_WORDS) == ENGLISH_STOP_WORDS
