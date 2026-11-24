"""Dependency-free TF-IDF cosine index.

Replaces ``sklearn.feature_extraction.text.TfidfVectorizer`` for the one way
this project used it: scoring every chunk against a query by cosine
similarity.

Why not sklearn? It pulls in SciPy, and the two together are ~160 MB - more
than half the serverless bundle budget, spent on a single fallback signal.
This module reproduces the scoring sklearn produced (asserted in
``test_tfidf.py``) using only NumPy, so the measured relevance floors in
:mod:`graphrag_plus.app.retrieval.service` keep their meaning.

Equivalence with ``TfidfVectorizer(lowercase=True, stop_words=...,
token_pattern=r"(?u)\\b[A-Za-z0-9_]{2,}\\b")``:

* ``smooth_idf=True``    -> ``idf(t) = ln((1 + n) / (1 + df(t))) + 1``
* ``sublinear_tf=False`` -> term frequency is the raw in-document count
* ``norm="l2"``          -> each document vector is L2-normalized
* query terms outside the fitted vocabulary are dropped

Scoring runs off an inverted index, so a query touches only the documents that
contain one of its terms instead of a full dense matrix multiply. At this
project's scale that is both smaller and faster than the path it replaces.
"""

from __future__ import annotations

import math
import re

import numpy as np

# Vendored verbatim from scikit-learn
# (``sklearn.feature_extraction.text.ENGLISH_STOP_WORDS``, BSD-3-Clause).
# Reproduced rather than re-derived because this project's retrieval gates were
# measured against this exact list - substituting a different one would
# silently shift every threshold in service.py.
ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
    (
        "a",
        "about",
        "above",
        "across",
        "after",
        "afterwards",
        "again",
        "against",
        "all",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "amoungst",
        "amount",
        "an",
        "and",
        "another",
        "any",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anywhere",
        "are",
        "around",
        "as",
        "at",
        "back",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "below",
        "beside",
        "besides",
        "between",
        "beyond",
        "bill",
        "both",
        "bottom",
        "but",
        "by",
        "call",
        "can",
        "cannot",
        "cant",
        "co",
        "con",
        "could",
        "couldnt",
        "cry",
        "de",
        "describe",
        "detail",
        "do",
        "done",
        "down",
        "due",
        "during",
        "each",
        "eg",
        "eight",
        "either",
        "eleven",
        "else",
        "elsewhere",
        "empty",
        "enough",
        "etc",
        "even",
        "ever",
        "every",
        "everyone",
        "everything",
        "everywhere",
        "except",
        "few",
        "fifteen",
        "fifty",
        "fill",
        "find",
        "fire",
        "first",
        "five",
        "for",
        "former",
        "formerly",
        "forty",
        "found",
        "four",
        "from",
        "front",
        "full",
        "further",
        "get",
        "give",
        "go",
        "had",
        "has",
        "hasnt",
        "have",
        "he",
        "hence",
        "her",
        "here",
        "hereafter",
        "hereby",
        "herein",
        "hereupon",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "hundred",
        "i",
        "ie",
        "if",
        "in",
        "inc",
        "indeed",
        "interest",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "keep",
        "last",
        "latter",
        "latterly",
        "least",
        "less",
        "ltd",
        "made",
        "many",
        "may",
        "me",
        "meanwhile",
        "might",
        "mill",
        "mine",
        "more",
        "moreover",
        "most",
        "mostly",
        "move",
        "much",
        "must",
        "my",
        "myself",
        "name",
        "namely",
        "neither",
        "never",
        "nevertheless",
        "next",
        "nine",
        "no",
        "nobody",
        "none",
        "noone",
        "nor",
        "not",
        "nothing",
        "now",
        "nowhere",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "part",
        "per",
        "perhaps",
        "please",
        "put",
        "rather",
        "re",
        "same",
        "see",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "serious",
        "several",
        "she",
        "should",
        "show",
        "side",
        "since",
        "sincere",
        "six",
        "sixty",
        "so",
        "some",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhere",
        "still",
        "such",
        "system",
        "take",
        "ten",
        "than",
        "that",
        "the",
        "their",
        "them",
        "themselves",
        "then",
        "thence",
        "there",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "thereupon",
        "these",
        "they",
        "thick",
        "thin",
        "third",
        "this",
        "those",
        "though",
        "three",
        "through",
        "throughout",
        "thru",
        "thus",
        "to",
        "together",
        "too",
        "top",
        "toward",
        "towards",
        "twelve",
        "twenty",
        "two",
        "un",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "very",
        "via",
        "was",
        "we",
        "well",
        "were",
        "what",
        "whatever",
        "when",
        "whence",
        "whenever",
        "where",
        "whereafter",
        "whereas",
        "whereby",
        "wherein",
        "whereupon",
        "wherever",
        "whether",
        "which",
        "while",
        "whither",
        "who",
        "whoever",
        "whole",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "yet",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    )
)

# Matches sklearn's ``token_pattern`` semantics for this configuration: runs of
# 2+ word characters. ``findall`` of ``+`` plus a length filter and
# ``\b[A-Za-z0-9_]{2,}\b`` agree on every input this sees.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# 1-character tokens carry no retrieval signal, and sklearn's ``{2,}`` dropped
# them as well.
_MIN_TOKEN_LEN = 2


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word characters, drop stopwords and 1-char tokens.

    Used for both indexing and querying so BM25 and TF-IDF always agree on the
    vocabulary - otherwise filler words like "about" or "tell" can drive
    matches for off-topic queries.
    """
    tokens = (tok.lower() for tok in _TOKEN_RE.findall(text or ""))
    return [tok for tok in tokens if len(tok) >= _MIN_TOKEN_LEN and tok not in ENGLISH_STOP_WORDS]


class TfidfIndex:
    """Fitted TF-IDF index over a fixed document set.

    Call :meth:`fit` again to rebuild for a new document set; the index is not
    incrementally updatable.
    """

    __slots__ = ("_doc_norms", "_idf", "_n_docs", "_postings")

    def __init__(self) -> None:
        self._n_docs: int = 0
        # term -> idf weight
        self._idf: dict[str, float] = {}
        # term -> (document indices, L2-normalized tf-idf weights)
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._doc_norms: np.ndarray = np.zeros(0, dtype=np.float64)

    # --------------------------------------------------------------- lifecycle
    def fit(self, texts: list[str]) -> None:
        """Build the index from ``texts``. Empty input yields an empty index."""
        self._n_docs = len(texts)
        self._idf = {}
        self._postings = {}
        self._doc_norms = np.zeros(self._n_docs, dtype=np.float64)
        if not texts:
            return

        # Per-document term counts and corpus document frequencies.
        counts: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for text in texts:
            term_freq: dict[str, int] = {}
            for token in tokenize(text):
                term_freq[token] = term_freq.get(token, 0) + 1
            counts.append(term_freq)
            for term in term_freq:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        # Smoothed idf, matching sklearn's smooth_idf=True.
        n_docs = self._n_docs
        self._idf = {term: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0 for term, df in doc_freq.items()}

        # Unnormalized weights, then the per-document L2 norm.
        raw: list[dict[str, float]] = []
        for doc_idx, term_freq in enumerate(counts):
            weights = {term: count * self._idf[term] for term, count in term_freq.items()}
            raw.append(weights)
            self._doc_norms[doc_idx] = math.sqrt(sum(w * w for w in weights.values()))

        # Invert into per-term postings carrying already-normalized weights, so
        # scoring never visits a document that shares no term with the query.
        staging: dict[str, list[tuple[int, float]]] = {}
        for doc_idx, weights in enumerate(raw):
            norm = float(self._doc_norms[doc_idx])
            if norm <= 0.0:
                continue
            for term, weight in weights.items():
                staging.setdefault(term, []).append((doc_idx, weight / norm))
        self._postings = {
            term: (
                np.fromiter((d for d, _ in rows), dtype=np.int64, count=len(rows)),
                np.fromiter((w for _, w in rows), dtype=np.float64, count=len(rows)),
            )
            for term, rows in staging.items()
        }

    # ------------------------------------------------------------------ scoring
    @property
    def is_empty(self) -> bool:
        """True when nothing is indexed, so callers can skip this signal."""
        return self._n_docs == 0 or not self._postings

    def cosine(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` against every document.

        Returns a ``float64`` array of length ``n_docs`` - all zeros when the
        query shares no in-vocabulary term with the corpus.
        """
        scores = np.zeros(self._n_docs, dtype=np.float64)
        if self.is_empty:
            return scores

        # Query vector: same weighting, restricted to the fitted vocabulary.
        q_freq: dict[str, int] = {}
        for token in tokenize(query):
            if token in self._idf:
                q_freq[token] = q_freq.get(token, 0) + 1
        if not q_freq:
            return scores
        q_weights = {term: count * self._idf[term] for term, count in q_freq.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm <= 0.0:
            return scores

        for term, weight in q_weights.items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_idx, doc_weights = posting
            scores[doc_idx] += doc_weights * (weight / q_norm)
        return scores


__all__ = ["ENGLISH_STOP_WORDS", "TfidfIndex", "tokenize"]
