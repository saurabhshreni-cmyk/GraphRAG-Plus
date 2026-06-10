"""Keyword-frequency domain classifier.

The classifier scans the corpus text once, counts hits against a small
per-domain vocabulary, and returns the highest-scoring domain — or
``"general"`` when no domain dominates. It's deliberately rule-based so
it stays fast, deterministic, and testable; swapping in an embedding-
based classifier later is a one-function change behind ``detect_domain``.

Currently recognized domains:

* ``machine_learning``
* ``physics``
* ``finance``
* ``biology``
* ``chemistry``
* ``mathematics``
* ``computer_science``
* ``general`` (fallback)
"""

from __future__ import annotations

import re

# Each domain has a vocabulary of distinctive (low-collision) terms. We
# avoid generic words that overlap multiple domains ("model", "system").
# Longer/multi-word terms are preferred where possible.
_DOMAIN_VOCAB: dict[str, frozenset[str]] = {
    "machine_learning": frozenset(
        {
            "neural network",
            "deep learning",
            "machine learning",
            "lstm",
            "rnn",
            "cnn",
            "transformer",
            "attention",
            "backpropagation",
            "gradient descent",
            "embedding",
            "convolutional",
            "recurrent",
            "reinforcement learning",
            "supervised",
            "unsupervised",
            "training set",
            "loss function",
            "softmax",
            "relu",
            "sigmoid",
            "feature vector",
            "classifier",
            "regression",
            "overfitting",
            "regularization",
            "batch",
            "epoch",
            "dropout",
            "tensor",
            "gradient",
        }
    ),
    "physics": frozenset(
        {
            "quantum",
            "entanglement",
            "particle",
            "wavefunction",
            "relativity",
            "spacetime",
            "electron",
            "photon",
            "neutron",
            "proton",
            "boson",
            "fermion",
            "string theory",
            "thermodynamics",
            "magnetic field",
            "electromagnetic",
            "kinetic energy",
            "potential energy",
            "gravity",
            "gravitational",
            "black hole",
            "schrödinger",
            "schrodinger",
            "heisenberg",
            "uncertainty principle",
            "quantum field",
            "lagrangian",
            "hamiltonian",
            "spin",
            "superposition",
            "decoherence",
        }
    ),
    "finance": frozenset(
        {
            "depreciation",
            "amortization",
            "balance sheet",
            "income statement",
            "cash flow",
            "asset",
            "liability",
            "equity",
            "revenue",
            "expense",
            "accrual",
            "dividend",
            "bond",
            "stock",
            "portfolio",
            "interest rate",
            "yield",
            "valuation",
            "discounted cash flow",
            "ebitda",
            "gross margin",
            "operating income",
            "net income",
            "shareholder",
            "capital",
            "leverage",
            "hedge",
            "derivative",
            "futures",
            "options",
            "compound interest",
        }
    ),
    "biology": frozenset(
        {
            "dna",
            "rna",
            "protein",
            "enzyme",
            "cell",
            "organism",
            "evolution",
            "species",
            "gene",
            "genome",
            "chromosome",
            "mitochondria",
            "ribosome",
            "membrane",
            "tissue",
            "organ",
            "ecosystem",
            "photosynthesis",
            "metabolism",
            "mutation",
            "natural selection",
            "homeostasis",
            "synapse",
            "neuron",
            "antibody",
            "antigen",
            "virus",
            "bacteria",
        }
    ),
    "chemistry": frozenset(
        {
            "molecule",
            "atom",
            "ion",
            "covalent",
            "ionic bond",
            "compound",
            "reaction",
            "catalyst",
            "ph",
            "acid",
            "base",
            "oxidation",
            "reduction",
            "electron pair",
            "electrolyte",
            "isotope",
            "periodic table",
            "valence",
            "stoichiometry",
            "polymer",
            "monomer",
            "hydrocarbon",
            "organic chemistry",
            "inorganic",
            "solution",
            "solvent",
            "solute",
        }
    ),
    "mathematics": frozenset(
        {
            "theorem",
            "proof",
            "integral",
            "derivative",
            "differential",
            "matrix",
            "vector space",
            "linear algebra",
            "calculus",
            "topology",
            "manifold",
            "polynomial",
            "eigenvalue",
            "eigenvector",
            "function",
            "domain",
            "codomain",
            "continuity",
            "limit",
            "infinity",
            "set theory",
            "group theory",
            "ring",
            "field",
            "modulo",
            "modular",
        }
    ),
    "computer_science": frozenset(
        {
            "algorithm",
            "complexity",
            "data structure",
            "graph theory",
            "binary tree",
            "linked list",
            "hash table",
            "compiler",
            "interpreter",
            "operating system",
            "concurrency",
            "thread",
            "process",
            "kernel",
            "syscall",
            "tcp",
            "http",
            "rest api",
            "database",
            "sql",
            "nosql",
            "indexing",
            "btree",
            "cache",
            "garbage collection",
        }
    ),
}

# Whole-word/phrase patterns per domain, compiled once at import time.
# Substring counting is NOT safe here: "ion" would match "integration" and
# "ph" would match "Phase", so every business document scored as chemistry.
_DOMAIN_PATTERNS: dict[str, list[tuple[re.Pattern[str], int]]] = {
    domain: [
        (
            re.compile(r"\b" + re.escape(term) + r"\b"),
            # Weight by term length so multi-word terms outweigh single-word.
            max(1, len(term) // 4),
        )
        for term in vocab
    ]
    for domain, vocab in _DOMAIN_VOCAB.items()
}


def detect_domain(text: str, *, min_hits: int = 3) -> str:
    """Return the dominant domain label for ``text`` or ``"general"``.

    Strategy:
      * Lowercase the text once.
      * Count whole-word (boundary-anchored) hits per domain.
      * Multi-word terms count more (a hit on "neural network" is
        stronger than a hit on "atom") because we score by total
        characters matched, not raw count.
      * Require ``min_hits`` total weight before declaring a winner.
    """
    if not text or not text.strip():
        return "general"
    haystack = text.lower()
    scores: dict[str, int] = {}
    for domain, patterns in _DOMAIN_PATTERNS.items():
        score = 0
        for pattern, weight in patterns:
            count = len(pattern.findall(haystack))
            if count:
                score += count * weight
        scores[domain] = score
    if not scores:
        return "general"
    best_domain, best_score = max(scores.items(), key=lambda p: p[1])
    if best_score < min_hits:
        return "general"
    # Require a clear margin over the runner-up so weak signals don't
    # promote a tie to a confident classification.
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    if best_score < second + 2:
        return "general"
    return best_domain
