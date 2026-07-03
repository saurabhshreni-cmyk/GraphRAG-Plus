"""Grounded answer generation.

Two strategies, both grounded in retrieved evidence and **intent-aware**:

* **Extractive (default)** — deterministic, fast, never hallucinates.
  The shape of the answer adapts to the detected query intent:

    - DEFINITION  → 1-2 sentence concise definition
    - LIST        → enumerated bullet list mined from list markers
    - COMPARISON  → bullet-style comparison of the two queried terms
    - EXPLANATION → multi-sentence paragraph synthesized from top chunks
    - FACTUAL     → original behaviour (top 2 sentences by relevance)

* **LLM (opt-in via ``llm_enabled``)** — pass the same evidence into a
  pluggable :class:`LLMClient` to produce a concise summary. If the
  client errors, we fall back to extractive.

Trust-aware behaviour:

* When ``evidence`` is empty we never call the LLM; the failure handler
  upstream classifies the query as ``NO_EVIDENCE``.
* The LLM only sees the retrieved snippets — there is no out-of-band
  context that could leak ungrounded facts.
* Quality filter discards sentences that don't share any content tokens
  with the question, so multi-chunk synthesis can't drift off-topic.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Protocol

# Sentinel strings the LLM is instructed to emit when evidence doesn't cover
# the question. Detecting them lets us fall back to extractive output instead
# of returning a verbatim refusal that the user can't act on.
LLM_ABSTAIN_TOKEN = "I cannot answer based on the provided context."
LLM_ABSTAIN_TOKEN_ALT = "I don't have enough information to answer this."

# Stopwords used for scoring sentences against the query. Kept in sync with
# the retrieval service philosophy: we want content overlap, not boilerplate
# overlap.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "if",
        "then",
        "else",
        "than",
        "what",
        "which",
        "who",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "tell",
        "about",
        "explain",
        "show",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_SENTENCE_VERB_RE = re.compile(
    r"\b("
    r"is|are|was|were|be|being|been|"
    r"refers?|means?|defines?|describes?|represents?|"
    r"uses?|includes?|contains?|consists?|comprises?|"
    r"has|have|had|allows?|enables?|requires?|involves?|"
    r"provides?|produces?|creates?|causes?|occurs?|happens?|"
    r"acquires?|acquired|merges?|merged|cancels?|canceled|continues?|continued|"
    r"supports?|supported|contradicts?|contradicted|"
    r"measures?|computes?|calculates?|records?|recognizes?|"
    r"classifies?|categorizes?|divides?|groups?|"
    r"controls?|stores?|maintains?|models?|learns?|"
    r"can|may|might|must|should|will|would"
    r")\b",
    re.IGNORECASE,
)
_MEANINGFUL_SHORT_STARTS = frozenset({"ai", "ml", "nlp", "qa", "rl", "on", "in", "at"})
_CUT_FIRST_WORDS = frozenset({"ant", "ing", "ed"})


class LLMClient(Protocol):
    """Minimal completion interface for any backend.

    A concrete implementation receives the user question plus retrieval
    context and returns a single answer string. Implementations are
    responsible for their own timeouts / retries.
    """

    def complete(self, question: str, context: str) -> str: ...


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text or "") if tok.lower() not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    # Normalize whitespace and split on .?! followed by whitespace + capital.
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    # Keep the original sentence-ending punctuation by re-joining single
    # trailing characters that the regex may have stripped.
    return [p.strip() for p in parts if p.strip()]


def _rank_sentences(question: str, sentences: list[str]) -> list[tuple[float, str]]:
    """Score each sentence by query-term recall (Jaccard-ish)."""
    q_tokens = _tokens(question)
    if not q_tokens:
        return [(0.0, s) for s in sentences]
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        s_tokens = _tokens(sentence)
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens)
        if overlap == 0:
            continue
        # Recall (overlap / |q|) weighted by precision (overlap / |s|),
        # which prefers tight, on-topic sentences over rambling ones.
        recall = overlap / len(q_tokens)
        precision = overlap / len(s_tokens)
        scored.append((recall + 0.3 * precision, sentence))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


# Fragment / mid-sentence detection -------------------------------------------
# Catches the broken-prefix pattern that surfaced in earlier validations:
# chunks that start with ``"ant Error Carousel..."`` or ``"s interpreted as..."``
# because the previous chunk ended mid-word. We refuse to surface these as
# answers — the user sees a complete sentence or no answer at all.

_MIN_SENTENCE_WORDS = 5


def _is_clean_sentence(sentence: str) -> bool:
    """True iff ``sentence`` is complete enough to show to a user.

    This gate is intentionally conservative. A candidate must look like a
    real sentence, contain a verb, and end with terminal punctuation before it
    can reach answer generation.
    """
    s = _hard_clean_sentence(sentence)
    if not s:
        return False
    if not re.search(r"[.!?][\")\]]?$", s):
        return False
    if not _starts_readably(s):
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", s)
    if len(words) < _MIN_SENTENCE_WORDS:
        return False
    if _looks_like_cut_first_word(words[0]):
        return False
    return bool(_SENTENCE_VERB_RE.search(s))


def _starts_readably(text: str) -> bool:
    first = text[0]
    if first.isupper() or first.isdigit():
        return True
    if first in "\"'([":
        return len(text) > 1 and (text[1].isupper() or text[1].isdigit())
    return False


def _looks_like_cut_first_word(word: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", word)
    if not cleaned:
        return True
    lower = cleaned.lower()
    if lower in _MEANINGFUL_SHORT_STARTS:
        return False
    if cleaned.isupper() and 2 <= len(cleaned) <= 10:
        return False
    if len(cleaned) < 3:
        return True
    return lower in _CUT_FIRST_WORDS


def _hard_clean_sentence(sentence: str) -> str:
    """Trim whitespace and remove obvious broken leading tokens."""
    s = re.sub(r"\s+", " ", sentence or "").strip()
    s = s.strip(" \t\r\n,;:")
    while s:
        first_word = re.match(r"^[A-Za-z']+", s)
        if not first_word:
            break
        word = first_word.group(0)
        if word[0].isupper() or word.isupper() or word.lower() in _MEANINGFUL_SHORT_STARTS:
            break
        if len(word) < 3 or word.lower() in _CUT_FIRST_WORDS:
            s = s[first_word.end() :].lstrip(" ,;:-")
            continue
        break
    return s


def _clean_sentences(text: str) -> list[str]:
    """Split + filter to only well-formed sentences."""
    return [
        cleaned
        for sentence in _split_sentences(text)
        if _is_clean_sentence(cleaned := _hard_clean_sentence(sentence))
    ]


# Generic enumeration-introducing verbs (don't require the "types of" prefix).
# These fire on phrases like "Methods include straight-line, declining balance, ..."
_ENUM_INTRO_RE = re.compile(
    r"\b(?:include[ds]?|including|are|consist[s]?\s+of|comprise[ds]?|"
    r"namely|such\s+as|classified\s+(?:into|as)|categori[sz]ed\s+as|"
    r"divided\s+into|grouped\s+(?:into|as))\s*[:\s]",
    re.IGNORECASE,
)
_BULLET_LINE_RE = re.compile(r"(?m)^\s*(?:[-*•·]|\d+[.)]|[a-z][.)])\s+(\S.+)$")
# Splits an inline enumeration into items. We only split on "," and ";" plus
# explicit "or"/"and" with surrounding whitespace; the per-item filter
# downstream removes anything that looks non-noun-phrase.
_INLINE_LIST_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|\bor\b)\s*", re.IGNORECASE)


def _extract_definition_subject(question: str) -> str:
    """Pull the term being defined out of a "what is X?" / "define X" query."""
    q = question.strip().rstrip("?.").strip()
    # "what is the X" / "what is an X" / "what is X"
    m = re.search(r"^\s*what\s+(?:is|are)\s+(?:the|a|an)?\s*(.+)$", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^\s*define\s+(.+)$", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^\s*(?:meaning|definition)\s+of\s+(.+)$", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return q


def _is_initialism(text: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", text or "")
    return 2 <= len(compact) <= 10 and compact.upper() == compact and any(ch.isalpha() for ch in compact)


def _initialism(text: str) -> str:
    pieces = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", text or "")
    initials: list[str] = []
    for piece in pieces:
        for subpiece in piece.split("-"):
            if subpiece:
                initials.append(subpiece[0].upper())
    return "".join(initials)


def _definition_sentence_for_subject(sentence: str, subject: str) -> str | None:
    """Return a definition sentence for ``subject`` or None.

    For definition queries, merely mentioning the subject is not enough. The
    sentence must say that the subject "is/are" something or "refers to"
    something. If the evidence expands an acronym, rewrite the leading
    expansion back to the queried term so the answer directly addresses the
    user's wording.
    """
    subject_clean = subject.strip()
    if not subject_clean:
        return None

    direct = re.compile(
        rf"\b{re.escape(subject_clean)}\b\s+(?:is|are|refers?\s+to)\b",
        re.IGNORECASE,
    )
    if direct.search(sentence):
        return sentence

    if not _is_initialism(subject_clean):
        return None

    target = re.sub(r"[^A-Za-z0-9]", "", subject_clean).upper()
    words = list(re.finditer(r"[A-Za-z]+(?:-[A-Za-z]+)*", sentence))
    for start in range(len(words)):
        for end in range(start + 2, min(len(words), start + 6) + 1):
            phrase = " ".join(match.group(0) for match in words[start:end])
            if _initialism(phrase) != target:
                continue
            phrase_start = words[start].start()
            phrase_end = words[end - 1].end()
            after = sentence[phrase_end:]
            if not re.match(r"\s+(?:is|are|refers?\s+to)\b", after, re.IGNORECASE):
                continue
            rewritten = f"{sentence[:phrase_start]}{subject_clean}{sentence[phrase_end:]}"
            return re.sub(r"\s+", " ", rewritten).strip()
    return None


# --- list item validation --------------------------------------------------


def _is_valid_list_item(item: str) -> bool:
    """True iff ``item`` looks like a real noun phrase, not a clause fragment.

    This keeps naive comma/and/or splits
    produce items like ``"not based on time"`` or ``"but on a level of
    Annuity"`` from prose like ``"...method based on units, not based on
    time, but on a level of Annuity"``.

    Rules:
      * 3 ≤ length ≤ 60 characters.
      * 1 ≤ tokens ≤ 5.
      * Must start with an alphabetic character.
      * First token must NOT be a verb / preposition / conjunction /
        determiner — these signal a clause fragment.
      * Must contain at least one alphabetic word ≥ 3 letters.
    """
    if not item:
        return False
    text = item.strip(" .;,()-")
    tokens = text.split()
    lowered = text.lower()
    first = tokens[0].lower() if tokens else ""
    is_single_generic = len(tokens) == 1 and tokens[0].strip(".,;:()[]").lower() in _GENERIC_LIST_ITEMS

    checks = (
        3 <= len(text) <= 60,
        bool(tokens) and len(tokens) <= 5,
        text[0].isalpha(),
        lowered not in _GENERIC_LIST_ITEMS,
        lowered not in _STOPWORDS,
        first not in _LIST_ITEM_BAD_STARTS,
        not is_single_generic,
        not _SENTENCE_VERB_RE.search(text),
        bool(re.search(r"\b[A-Za-z]{3,}\b", text)),
    )
    return all(checks)


def _list_query_head(question: str) -> str | None:
    match = re.search(
        r"\b(types?|kinds?|examples?|methods?|categories|forms?|varieties|classes|approaches)\b",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None
    head = match.group(1).lower()
    singular_map = {
        "types": "type",
        "kinds": "kind",
        "examples": "example",
        "methods": "method",
        "categories": "category",
        "forms": "form",
        "varieties": "variety",
        "classes": "class",
        "approaches": "approach",
    }
    return singular_map.get(head, head)


def _cohere_list_candidates(question: str, candidates: list[str]) -> list[str]:
    """Keep the most coherent subset when prose yields mixed-quality items.

    Inline enumerations often contain a few true list members plus stray noun
    phrases from the surrounding clause. When two or more candidates share the
    same trailing headword (for example ``"... method"``), that cluster is
    usually the real list and the outliers are noise.
    """
    if len(candidates) < 3:
        return candidates

    strict_heads = {"type", "kind", "method", "category", "form", "variety", "class", "approach"}
    if _list_query_head(question) not in strict_heads:
        return candidates

    head_counts = Counter(item.split()[-1].lower() for item in candidates if item.split())
    dominant_head, dominant_count = head_counts.most_common(1)[0]
    if dominant_count < 2:
        return candidates
    if dominant_count < max(2, len(candidates) // 2):
        return candidates
    return [item for item in candidates if item.split() and item.split()[-1].lower() == dominant_head]


# Words that can't legitimately START a noun-phrase list item. If a "list
# item" begins with one of these, it's almost certainly a clause fragment
# carved out of prose.
_LIST_ITEM_BAD_STARTS = frozenset(
    {
        # connectives / prepositions
        "and", "or", "but", "nor", "so", "yet",
        "in", "on", "at", "to", "for", "with", "by", "from",
        "as", "of", "into", "onto", "upon", "about",
        "between", "among", "through", "during", "before", "after",
        # verbs / participles that signal clause continuations
        "based", "used", "consisting", "containing", "having",
        "including", "excluding", "applied", "computed", "derived",
        # determiners / pronouns
        "this", "that", "these", "those", "it", "they", "them",
        # negations / connectives
        "not", "no", "neither", "either",
        # generic clause heads
        "however", "although", "because", "since", "though",
        "while", "whereas", "whenever",
    }
)  # fmt: skip

_GENERIC_LIST_ITEMS = frozenset(
    {
        "type",
        "types",
        "kind",
        "kinds",
        "method",
        "methods",
        "system",
        "systems",
        "model",
        "models",
        "data",
        "asset",
        "assets",
        "item",
        "items",
        "example",
        "examples",
        "category",
        "categories",
        "form",
        "forms",
        "thing",
        "things",
    }
)


class AnswerGenerator:
    """Deterministic generator with optional LLM-backed synthesis."""

    def __init__(self, llm_enabled: bool, llm_client: LLMClient | None = None):
        self.llm_enabled = llm_enabled
        self.llm_client = llm_client

    # --------------------------------------------------------- extractive paths

    @staticmethod
    def _extractive_answer(
        question: str,
        evidence: list[dict[str, object]],
        intent: str | None = None,
        comparison_terms: tuple[str, str] | None = None,
    ) -> str:
        """Route to the right extractive strategy for the detected intent."""
        if not evidence:
            return "Evidence was found but insufficient for a complete answer."

        if intent == "definition":
            return AnswerGenerator._definition_answer(question, evidence)
        if intent == "list":
            return AnswerGenerator._list_answer(question, evidence)
        if intent == "comparison":
            return AnswerGenerator._comparison_answer(question, evidence, comparison_terms)
        if intent == "explanation":
            return AnswerGenerator._explanation_answer(question, evidence)
        return AnswerGenerator._factual_answer(question, evidence)

    # --- intent-specific extractive strategies --------------------------------

    @staticmethod
    def _evidence_text(item: dict[str, object]) -> str:
        """Return the chunk's *full* text when available, falling back to
        the truncated 300-char snippet.

        Using the full text lets us extract complete sentences instead of
        cutting off at the snippet boundary.
        """
        full = str(item.get("full_text", "") or "")
        if full.strip():
            return full
        return str(item.get("snippet", "") or "")

    @staticmethod
    def _factual_answer(question: str, evidence: list[dict[str, object]]) -> str:
        """Top question-relevant *complete* sentence from the best chunk."""
        for item in evidence[:3]:
            text = AnswerGenerator._evidence_text(item)
            sentences = _clean_sentences(text)
            ranked = _rank_sentences(question, sentences)
            if ranked:
                # One high-quality sentence beats two of mixed quality.
                return _ensure_complete(ranked[0][1])
        fallback = AnswerGenerator._fallback_sentence_answer(question, evidence)
        if fallback:
            return fallback
        return "Evidence was found but insufficient for a complete answer."

    @staticmethod
    def _fallback_sentence_answer(question: str, evidence: list[dict[str, object]]) -> str | None:
        """Merge the top two valid sentences when no single ranked answer wins."""
        candidates: list[str] = []
        seen: set[str] = set()
        for item in evidence[:3]:
            text = AnswerGenerator._evidence_text(item)
            for sentence in _clean_sentences(text):
                key = sentence.lower().rstrip(".!?")
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(sentence)
                if len(candidates) >= 2:
                    return " ".join(_ensure_complete(s) for s in candidates)
        return None

    @staticmethod
    def _definition_answer(question: str, evidence: list[dict[str, object]]) -> str:
        """One clean sentence answering "what is X?"-style queries.

        Strategy for clean definition generation:
          1. Pull the subject term from the question (strip "what is" / "define").
          2. Walk the top-3 chunks, splitting into clean sentences only.
          3. Score each sentence by:
               * +5 if it contains an explicit definition pattern
                 ("X is a", "X refers to", "X is defined as", ...).
               * +2 if it mentions the subject term.
               * + token-overlap recall.
          4. Prefer the FIRST high-scoring sentence in the chunk (encyclopedic
             definitions almost always lead the article).
          5. Return a single, complete sentence — no concatenation.
        """
        subject = _extract_definition_subject(question)
        q_tokens = _tokens(question)

        best: tuple[float, str] | None = None
        for chunk_idx, item in enumerate(evidence[:3]):
            text = AnswerGenerator._evidence_text(item)
            sentences = _clean_sentences(text)
            for sent_idx, sentence in enumerate(sentences):
                definition_sentence = _definition_sentence_for_subject(sentence, subject)
                if definition_sentence is None:
                    continue
                score = 0.0
                score += 7.0
                # Token overlap with the original question.
                s_tokens = _tokens(definition_sentence)
                if q_tokens and s_tokens:
                    score += len(q_tokens & s_tokens) / max(1, len(q_tokens))
                # Earlier sentences in earlier chunks win on ties — that's
                # where canonical definitions live.
                score -= 0.05 * (chunk_idx * 5 + sent_idx)
                if score <= 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, definition_sentence)

        if best:
            return _ensure_complete(best[1])
        fallback = AnswerGenerator._fallback_sentence_answer(question, evidence)
        if fallback:
            return fallback
        return "Evidence was found but insufficient for a complete answer."

    @staticmethod
    def _list_answer(question: str, evidence: list[dict[str, object]]) -> str:
        """Mine list items from across top-k chunks with strict validation.

        Strategy for list extraction:
          1. Pass A: scan top-5 chunks for visible bullet / numbered lines.
             These are the highest-fidelity items because the source
             explicitly marked them.
          2. Pass B: scan for inline enumerations introduced by an explicit
             marker (``"include"``, ``"are"``, ``"such as"``, ``"consist
             of"``). Split on commas / "and" / "or" and validate each
             piece as a noun phrase via :func:`_is_valid_list_item`.
          3. Pass C (fallback): scan capitalized noun phrases across all
             chunks and return the top-N most frequent. Catches articles
             that don't use explicit list markers.
        """
        items: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> bool:
            normalized = candidate.strip(" .;,:-")
            key = normalized.lower()
            if not key or key in seen:
                return False
            if not _is_valid_list_item(normalized):
                return False
            seen.add(key)
            items.append(normalized)
            return True

        # ---- Pass A: bullet / numbered lines -----------------------------
        for ev in evidence[:5]:
            text = AnswerGenerator._evidence_text(ev)
            for match in _BULLET_LINE_RE.finditer(text):
                add(match.group(1))
                if len(items) >= 12:
                    break
            if len(items) >= 12:
                break

        # ---- Pass B: inline enumeration after a marker -------------------
        if len(items) < 3:
            for ev in evidence[:5]:
                text = AnswerGenerator._evidence_text(ev)
                for marker in _ENUM_INTRO_RE.finditer(text):
                    tail = text[marker.end() :]
                    # Stop at the first sentence boundary so we don't run
                    # past the enumeration into the next idea.
                    sentence_end = re.search(r"[.!?](?:\s|$)", tail)
                    if sentence_end:
                        tail = tail[: sentence_end.start()]
                    pieces = [piece.strip(" .;,:-") for piece in _INLINE_LIST_SPLIT_RE.split(tail)]
                    pieces = [piece for piece in pieces if _is_valid_list_item(piece)]
                    pieces = _cohere_list_candidates(question, pieces)
                    for piece in pieces:
                        add(piece)
                        if len(items) >= 12:
                            break
                    if len(items) >= 12:
                        break
                if len(items) >= 12:
                    break

        # ---- Pass C (fallback): top capitalized noun phrases -------------
        if len(items) < 3:
            phrase_counts: Counter[str] = Counter()
            for ev in evidence[:5]:
                text = AnswerGenerator._evidence_text(ev)
                for phrase in re.findall(
                    r"\b[A-Z][a-zA-Z]{2,}(?:[\s\-][A-Z]?[a-zA-Z]{2,}){0,3}\b",
                    text,
                ):
                    if _is_valid_list_item(phrase):
                        phrase_counts[phrase] += 1
            for phrase, count in phrase_counts.most_common(20):
                if count >= 2:
                    add(phrase)
                if len(items) >= 8:
                    break

        if not items:
            # No list structure detected at all — fall through to factual.
            return AnswerGenerator._factual_answer(question, evidence)

        # Render as enumerated list.
        head = _list_intro(question)
        body = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
        return f"{head}\n{body}"

    @staticmethod
    def _comparison_answer(
        question: str,
        evidence: list[dict[str, object]],
        comparison_terms: tuple[str, str] | None,
    ) -> str:
        """Build a bullet-style A vs B comparison."""
        if comparison_terms is None:
            return AnswerGenerator._factual_answer(question, evidence)
        term_a, term_b = comparison_terms
        a_lines: list[str] = []
        b_lines: list[str] = []
        seen_a: set[str] = set()
        seen_b: set[str] = set()

        for item in evidence[:8]:
            text = AnswerGenerator._evidence_text(item)
            for sentence in _split_sentences(text):
                low = sentence.lower()
                has_a = term_a.lower() in low
                has_b = term_b.lower() in low
                key = low[:80]
                # Sentences that only mention A → A's column; same for B.
                # Sentences that mention BOTH go to the "intersection" pool
                # but we'll emit them under whichever side has fewer entries
                # to keep balance.
                if has_a and not has_b and key not in seen_a and len(a_lines) < 3:
                    a_lines.append(sentence.strip())
                    seen_a.add(key)
                elif has_b and not has_a and key not in seen_b and len(b_lines) < 3:
                    b_lines.append(sentence.strip())
                    seen_b.add(key)
                elif has_a and has_b:
                    if len(a_lines) <= len(b_lines) and key not in seen_a and len(a_lines) < 3:
                        a_lines.append(sentence.strip())
                        seen_a.add(key)
                    elif key not in seen_b and len(b_lines) < 3:
                        b_lines.append(sentence.strip())
                        seen_b.add(key)
            if len(a_lines) >= 3 and len(b_lines) >= 3:
                break

        if not a_lines and not b_lines:
            return AnswerGenerator._factual_answer(question, evidence)

        out_lines: list[str] = [f"Comparison of {term_a} and {term_b}:"]
        out_lines.append(f"\n{term_a}:")
        for line in a_lines or ["(no specific evidence found)"]:
            out_lines.append(f"  - {line}")
        out_lines.append(f"\n{term_b}:")
        for line in b_lines or ["(no specific evidence found)"]:
            out_lines.append(f"  - {line}")
        return "\n".join(out_lines)

    @staticmethod
    def _explanation_answer(question: str, evidence: list[dict[str, object]]) -> str:
        """Multi-chunk synthesis into a 2-4 sentence paragraph.

        * Filters fragment / mid-sentence garbage via :func:`_clean_sentences`.
        * De-dupes near-identical sentences (often introduced by paragraph-
          level chunk overlap).
        * Caps at 3 sentences so the answer reads as a paragraph, not a wall
          of text.
        """
        ranked: list[tuple[float, str]] = []
        seen_keys: set[str] = set()
        for item in evidence[:5]:
            text = AnswerGenerator._evidence_text(item)
            sentences = _clean_sentences(text)
            for score, sentence in _rank_sentences(question, sentences):
                key = sentence.lower().rstrip(".!?")[:80]
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                ranked.append((score, sentence))
        ranked.sort(key=lambda p: p[0], reverse=True)
        top = [_ensure_complete(s) for _, s in ranked[:3]]
        if not top:
            return AnswerGenerator._factual_answer(question, evidence)
        return " ".join(top)

    # ---------------------------------------------------------------- LLM path
    # How many evidence chunks feed the LLM prompt. 5 (not 3) because the
    # answer for aggregate questions ("which companies…") often lives in
    # rank-4/5 chunks; the client still caps total context characters.
    _CONTEXT_CHUNKS = 5

    @staticmethod
    def _build_context(evidence: list[dict[str, object]]) -> str:
        """Top-N strongest chunks, deduped sentence-by-sentence.

        Sorted-by-final-score happens upstream (ScoringModule), so we just
        take the prefix here. We split into sentences so duplicate sentences
        across overlapping chunks (common when ingestion produced
        near-identical paragraphs) don't show up twice in the prompt.

        Uses the chunk's FULL text (the client caps total context length) —
        the 300-char snippet routinely truncates mid-answer, which made the
        LLM abstain on questions the evidence could actually answer.
        """
        seen: set[str] = set()
        rendered: list[str] = []
        for idx, item in enumerate(evidence[: AnswerGenerator._CONTEXT_CHUNKS]):
            source = str(item.get("source_id", "?"))
            snippet = str(item.get("full_text") or item.get("snippet") or "").strip()
            if not snippet:
                continue
            kept_sentences: list[str] = []
            for sentence in _clean_sentences(snippet):
                key = sentence.lower().rstrip(".!?").strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                kept_sentences.append(sentence)
            if kept_sentences:
                rendered.append(f"[{idx + 1}] ({source}) {' '.join(kept_sentences)}")
        return "\n".join(rendered)

    def _llm_answer(self, question: str, evidence: list[dict[str, object]]) -> str:
        client = self.llm_client
        if client is None:
            # No client wired — fall through to deterministic placeholder so
            # behaviour stays predictable in test environments.
            best = str(evidence[0].get("snippet", "") or "")[:240]
            return f"LLM-style synthesis: {best}".strip()
        context = self._build_context(evidence)
        return client.complete(question, context).strip()

    # --------------------------------------------------------- quality filter
    @staticmethod
    def _llm_answer_passes_quality(question: str, answer: str, context: str = "") -> bool:
        """Reject LLM output that is neither question-relevant nor grounded.

        Two ways to pass:
        1. Shares a content token with the question, OR
        2. Is grounded in the evidence: at least half of the answer's content
           tokens appear in the context. This admits legitimate entity-list
           answers ("Google, Microsoft, LinkedIn…") that paraphrase the
           question entirely, while still rejecting free-associated output
           whose vocabulary matches neither question nor evidence.
        """
        q_tokens = _tokens(question)
        a_tokens = _tokens(answer)
        if not q_tokens:
            return bool(a_tokens)
        if q_tokens & a_tokens:
            return True
        if context and a_tokens:
            c_tokens = _tokens(context)
            grounded = len(a_tokens & c_tokens)
            if grounded >= 3 and grounded / len(a_tokens) >= 0.5:
                return True
        # Single-content-token questions ("NetworkX?") with a substantive
        # answer pass — retrieval already enforced topical chunks.
        return len(q_tokens) <= 1 and len(a_tokens) >= 6

    # ----------------------------------------------------------------- public
    def generate(  # noqa: PLR0911 -- early returns mirror the documented gate flow
        self,
        question: str,
        evidence: list[dict[str, object]],
        confidence: float,
        answer_threshold: float,
        *,
        intent: str | None = None,
        comparison_terms: tuple[str, str] | None = None,
    ) -> tuple[str, bool, bool]:
        """Generate answer text. Returns (answer, used_llm, llm_failed).

        * ``used_llm`` — True iff the LLM client produced the final answer.
        * ``llm_failed`` — True iff the LLM was attempted and raised; in
          that case we transparently fall back to extractive output.
        * ``intent`` / ``comparison_terms`` — drive intent-aware extractive
          paths. ``None`` means "use legacy factual behaviour".
        """
        if not evidence:
            # NO_EVIDENCE handling stays in the failure classifier; we just
            # emit a stable abstain string and never call the LLM.
            return (
                "I cannot answer reliably because no evidence was found.",
                False,
                False,
            )

        extractive = self._extractive_answer(
            question, evidence, intent=intent, comparison_terms=comparison_terms
        )

        # Flow (upgraded — LLM-first):
        #     evidence empty    -> abstain, never call LLM (handled above)
        #     llm_enabled=False -> extractive
        #     llm_enabled=True  -> LLM synthesis, extractive fallback on any
        #                          error / timeout / abstain / quality reject.
        # ``confidence`` / ``answer_threshold`` no longer gate the LLM call;
        # they still drive the failure classifier upstream.
        _ = confidence, answer_threshold
        if not self.llm_enabled:
            return extractive, False, False
        try:
            llm_answer = self._llm_answer(question, evidence)
        except Exception:
            # LLM raised (timeout / connection / decode) -> extractive fallback,
            # surface llm_failed=True in the flags for observability.
            return extractive, False, True

        if not llm_answer:
            # Empty response -> soft failure, fall back to extractive.
            return extractive, False, True

        # The LLM was instructed to emit a fixed abstain string when the
        # context doesn't cover the question. We have evidence (this branch
        # only runs when len(evidence) > 0), so an abstain means the model
        # disagreed with retrieval. Trust retrieval and fall back to the
        # extractive sentence-rank answer.
        lowered_answer = llm_answer.lower()
        for abstain in (LLM_ABSTAIN_TOKEN, LLM_ABSTAIN_TOKEN_ALT):
            if abstain.lower().rstrip(".") in lowered_answer:
                return extractive, False, True

        # Quality filter: reject completions that are neither on-topic nor
        # grounded in the retrieved evidence.
        if not self._llm_answer_passes_quality(question, llm_answer, self._build_context(evidence)):
            return extractive, False, True

        return llm_answer, True, False

    # ------------------------------------------------- reasoning verification
    _verifier = None  # class-level lazy singleton (shared across corpora)

    @classmethod
    def _get_verifier(cls):
        if cls._verifier is None:
            from graphrag_plus.app.generation.reasoning_verifier import ReasoningVerifier

            cls._verifier = ReasoningVerifier()
        return cls._verifier

    @staticmethod
    def _verifier_enabled() -> bool:
        import os

        return os.environ.get("GRAPHRAG_REASONING_VERIFIER", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }

    def generate_verified(
        self,
        question: str,
        evidence: list[dict[str, object]],
        confidence: float,
        answer_threshold: float,
        *,
        intent: str | None = None,
        comparison_terms: tuple[str, str] | None = None,
    ) -> tuple[str, bool, bool, object | None]:
        """:meth:`generate` plus a DeepSeek R1 verification pass on LLM drafts.

        Returns ``(answer, used_llm, llm_failed, verification)`` where
        ``verification`` is a
        :class:`~graphrag_plus.app.generation.reasoning_verifier.VerificationResult`
        or ``None`` when verification didn't run (extractive answer, verifier
        disabled, or model unavailable). Never raises past :meth:`generate`.
        """
        answer, used_llm, llm_failed = self.generate(
            question,
            evidence,
            confidence,
            answer_threshold,
            intent=intent,
            comparison_terms=comparison_terms,
        )
        # Verify only genuine LLM drafts: extractive output is deterministic
        # and already evidence-bound, so a reasoning pass adds latency for no
        # grounding benefit.
        if not used_llm or not self._verifier_enabled():
            return answer, used_llm, llm_failed, None
        try:
            verifier = self._get_verifier()
            if not verifier.available():
                return answer, used_llm, llm_failed, None
            verification = verifier.verify(question, answer, evidence)
            return verification.final_answer, used_llm, llm_failed, verification
        except Exception:  # defensive: verification must never break answering
            return answer, used_llm, llm_failed, None

    def generate_result(
        self,
        question: str,
        evidence: list[dict[str, object]],
        confidence: float,
        answer_threshold: float,
        *,
        intent: str | None = None,
        comparison_terms: tuple[str, str] | None = None,
    ) -> "AnswerResult":
        """Structured variant of :meth:`generate` returning an AnswerResult.

        Confidence combines the calibrated retrieval confidence with a bonus
        for how many independent retrieval signals (BM25 / semantic / graph)
        contributed to the evidence — answers backed by multiple signals are
        more trustworthy than single-signal hits.
        """
        from graphrag_plus.app.models.schemas import AnswerResult

        answer, used_llm, llm_failed, verification = self.generate_verified(
            question,
            evidence,
            confidence,
            answer_threshold,
            intent=intent,
            comparison_terms=comparison_terms,
        )

        signals: set[str] = set()
        for item in evidence:
            if float(item.get("raw_bm25", 0.0) or 0.0) > 0:
                signals.add("bm25")
            if float(item.get("raw_cosine", 0.0) or 0.0) > 0:
                signals.add("semantic")
            if float(item.get("raw_graph", item.get("graph_score", 0.0)) or 0.0) > 0:
                signals.add("graph")
        signal_bonus = 0.05 * max(0, len(signals) - 1)
        final_confidence = min(1.0, max(0.0, confidence + signal_bonus))

        sources = list(dict.fromkeys(str(item.get("source_id", "")) for item in evidence if item.get("source_id")))
        reasoning_bits = [
            f"signals={'+'.join(sorted(signals)) or 'none'}",
            f"generator={'llm' if used_llm else 'extractive'}",
        ]
        if llm_failed:
            reasoning_bits.append("llm_failed=fallback_to_extractive")

        return AnswerResult(
            answer=answer,
            confidence=final_confidence,
            sources=sources,
            entities_used=[],
            reasoning="; ".join(reasoning_bits),
            verified_by_reasoning=bool(verification and getattr(verification, "verified", False)),
            reasoning_summary=str(getattr(verification, "reasoning_summary", "") or ""),
            answer_changed_by_reasoning=bool(verification and getattr(verification, "changed", False)),
        )


# --- module-level helpers ---------------------------------------------------


def _ensure_complete(sentence: str) -> str:
    """Make ``sentence`` end with a complete-sentence punctuation mark.

    Strips trailing whitespace, then appends a period if the last visible
    character isn't ``.``, ``!``, ``?``, or a closing quote/paren that
    follows one of those. Prevents the "mid-sentence answer" failure where
    a chunk's last sentence got cut off by snippet truncation.
    """
    text = sentence.strip()
    if not text:
        return text
    # Already ends with terminal punctuation (allow trailing brackets/quotes).
    if re.search(r"[.!?][\")\]]?$", text):
        return text
    return text + "."


def _list_intro(question: str) -> str:
    """Pick a sensible heading for a list answer.

    Tries to mirror the user's phrasing — "Types of X" stays "Types of X:" —
    so the output reads as a direct response, not boilerplate.
    """
    cleaned = question.strip().rstrip("?.")
    match = re.search(
        r"\b(types?|kinds?|examples?|methods?|categories|forms?|varieties|classes|approaches)\s+of\s+(.+)$",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        head = match.group(1).capitalize()
        subject = match.group(2).strip()
        return f"{head} of {subject}:"
    return "Items:"
