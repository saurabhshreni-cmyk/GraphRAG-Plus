r"""Content-aware chunking.

The chunker splits documents in three priority tiers so that retrieval
gets coherent units that don't break definitions or list enumerations
mid-sentence:

1. **Heading boundaries** — Markdown ``#`` / ``##`` lines and ALL-CAPS
   short lines are treated as section starts. Each section's first chunk
   is emitted independently so headings stay attached to their paragraph.
2. **Paragraph packing** — paragraphs (split on blank lines) are packed
   into chunks up to ``chunk_size`` characters. List blocks (consecutive
   bullet / numbered lines) are kept atomic — never split across chunks
   even if oversized.
3. **Sentence packing fallback** — for paragraphs longer than
   ``chunk_size`` we pack at sentence boundaries (``[.!?]\s+``) instead
   of fixed character offsets, so definitions don't get cleaved.

The previous fixed-window implementation is preserved as a final safety
net for content that has none of the above structure (e.g. a single
30 000-character blob with no whitespace, or a single oversized sentence).
"""

from __future__ import annotations

import re

from graphrag_plus.app.ingestion.models import Chunk, Document

TIMESTAMP_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Markdown-ish heading patterns + ALL-CAPS-LINE heuristic.
_HEADING_RE = re.compile(r"^(?:#{1,6}\s+.+|[A-Z][A-Z0-9 \-:&]{4,80})$")
# Bullet / numbered list line.
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•·]|\d+[.)]|[a-z][.)])\s+\S")
# Sentence boundary: ".!?" followed by whitespace + capital/digit.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split documents into content-aware chunks with timestamp hints.

    Returns one ``Chunk`` per logical unit. ``chunk_overlap`` is honoured at
    paragraph-boundary level (we re-include the last paragraph of the
    previous chunk in the next chunk) rather than character-level, which
    matches the way humans skim documents.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        text = doc.text or ""
        if not text.strip():
            continue
        for idx, payload in enumerate(_split_document(text, chunk_size, chunk_overlap)):
            chunk_text, start, end = payload
            timestamp_match = TIMESTAMP_PATTERN.search(chunk_text)
            timestamp = timestamp_match.group(1) if timestamp_match else None
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_ch_{idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start=start,
                    end=end,
                    timestamp=timestamp,
                )
            )
    return chunks


# --- splitting internals ----------------------------------------------------


def _split_document(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    """Return ``[(chunk_text, start_offset, end_offset), ...]`` for ``text``.

    Honours: heading breaks, paragraph packing, atomic list blocks,
    sentence-aware fallback for oversized paragraphs.
    """
    units = _logical_units(text)
    if not units:
        return _fixed_window(text, chunk_size, chunk_overlap)

    # Pack units into chunks, respecting chunk_size budget.
    packed: list[tuple[str, int, int]] = []
    buffer: list[tuple[str, int, int]] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        merged_text = "\n\n".join(piece[0] for piece in buffer).strip()
        if merged_text:
            packed.append((merged_text, buffer[0][1], buffer[-1][2]))
        buffer = []
        buffer_len = 0

    for unit_text, unit_start, unit_end, unit_kind in units:
        unit_size = len(unit_text)

        # Atomic units (lists, headings) get flushed standalone if they're
        # large enough to be their own chunk.
        if unit_kind in {"list", "heading"} and unit_size > chunk_size:
            flush()
            packed.append((unit_text.strip(), unit_start, unit_end))
            continue

        # Headings start a new chunk so they stay attached to their section.
        if unit_kind == "heading" and buffer:
            flush()

        # If adding this unit would exceed budget, flush first.
        if buffer and buffer_len + unit_size + 2 > chunk_size:
            flush()

        # Oversized regular paragraph → sentence-pack it.
        if unit_size > chunk_size:
            flush()
            for sent_text, s_start, s_end in _sentence_pack(unit_text, unit_start, chunk_size):
                packed.append((sent_text, s_start, s_end))
            continue

        buffer.append((unit_text, unit_start, unit_end))
        buffer_len += unit_size + 2

    flush()

    # Apply paragraph-level overlap: each non-first chunk gets the trailing
    # sentences of the previous chunk re-prepended (up to ``chunk_overlap``
    # characters). Keeps continuity without char-level cleaving.
    if chunk_overlap > 0 and len(packed) > 1:
        with_overlap: list[tuple[str, int, int]] = [packed[0]]
        for i in range(1, len(packed)):
            prev_text, _, _ = packed[i - 1]
            cur_text, cur_start, cur_end = packed[i]
            tail = _trailing_chars(prev_text, chunk_overlap)
            if tail and not cur_text.startswith(tail):
                merged = f"{tail}\n\n{cur_text}"
                with_overlap.append((merged, cur_start, cur_end))
            else:
                with_overlap.append(packed[i])
        return with_overlap
    return packed


def _logical_units(text: str) -> list[tuple[str, int, int, str]]:
    """Break ``text`` into ``(unit_text, start, end, kind)`` tuples.

    Kinds: ``"heading"``, ``"list"``, ``"paragraph"``. List blocks group
    consecutive bullet / numbered lines so they stay atomic.
    """
    units: list[tuple[str, int, int, str]] = []
    lines = text.split("\n")
    cursor = 0
    line_offsets: list[tuple[int, int]] = []
    for line in lines:
        line_offsets.append((cursor, cursor + len(line)))
        cursor += len(line) + 1  # +1 for the newline we split on

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        l_start, _ = line_offsets[i]

        # Heading: standalone short line that's a markdown heading or all-caps.
        if _HEADING_RE.match(stripped) and len(stripped) <= 120:
            _, l_end = line_offsets[i]
            units.append((stripped, l_start, l_end, "heading"))
            i += 1
            continue

        # List block: gather all consecutive list lines.
        if _LIST_LINE_RE.match(line):
            j = i
            while j < n and (
                _LIST_LINE_RE.match(lines[j]) or (lines[j].strip() and lines[j].startswith((" ", "\t")))
            ):
                j += 1
            block = "\n".join(lines[i:j]).strip()
            _, l_end = line_offsets[j - 1]
            units.append((block, l_start, l_end, "list"))
            i = j
            continue

        # Paragraph: gather lines until blank line or list/heading.
        j = i
        while j < n:
            if not lines[j].strip():
                break
            if _LIST_LINE_RE.match(lines[j]) and j != i:
                break
            if _HEADING_RE.match(lines[j].strip()) and len(lines[j].strip()) <= 120 and j != i:
                break
            j += 1
        para = " ".join(ln.strip() for ln in lines[i:j] if ln.strip()).strip()
        _, l_end = line_offsets[j - 1]
        if para:
            units.append((para, l_start, l_end, "paragraph"))
        i = j
    return units


def _sentence_pack(text: str, base_offset: int, chunk_size: int) -> list[tuple[str, int, int]]:
    """Pack sentences into chunks ≤ ``chunk_size`` characters.

    Boundaries are detected via ``[.!?]\\s+[A-Z0-9]``; offsets are relative
    to ``base_offset`` so the chunk's ``start``/``end`` still index into
    the original document.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    if not sentences:
        return [(text.strip(), base_offset, base_offset + len(text))]

    packs: list[tuple[str, int, int]] = []
    buf: list[str] = []
    buf_len = 0
    cursor = base_offset
    pack_start = base_offset

    for sent in sentences:
        sent_size = len(sent)
        if buf and buf_len + sent_size + 1 > chunk_size:
            merged = " ".join(buf).strip()
            packs.append((merged, pack_start, cursor))
            buf = []
            buf_len = 0
            pack_start = cursor

        if sent_size > chunk_size:
            # Single sentence still too big — split on character windows so
            # we don't return a giant unsplittable chunk. Honours the
            # ``chunk_size`` budget, with a tiny overlap so context isn't
            # cleaved at an arbitrary character.
            if buf:
                merged = " ".join(buf).strip()
                packs.append((merged, pack_start, cursor))
                buf = []
                buf_len = 0
            for window_text, w_start, w_end in _fixed_window(sent, chunk_size, max(2, chunk_size // 5)):
                packs.append((window_text, cursor + w_start, cursor + w_end))
            cursor += sent_size + 1
            pack_start = cursor
            continue

        buf.append(sent)
        buf_len += sent_size + 1
        cursor += sent_size + 1

    if buf:
        merged = " ".join(buf).strip()
        packs.append((merged, pack_start, cursor))

    return packs


def _trailing_chars(text: str, n: int) -> str:
    """Return last ``n`` characters aligned to sentence-start when possible.

    Falls back to word-boundary alignment so the overlap never starts
    mid-word.
    """
    if len(text) <= n:
        return text
    tail = text[-n:]
    # Prefer aligning to the next sentence boundary inside the tail.
    sentence_match = re.search(r"(?<=[.!?])\s+", tail)
    if sentence_match:
        tail = tail[sentence_match.end() :]
    else:
        # Otherwise align to the next whitespace so we don't cleave a word.
        ws_match = re.search(r"\s", tail)
        if ws_match:
            tail = tail[ws_match.end() :]
    return tail.strip()


def _fixed_window(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    """Last-resort character-window fallback for content with no structure.

    Aligns ``start`` and ``end`` to word boundaries (whitespace) so chunks
    never begin or end mid-word. Without this alignment, downstream
    sentence extraction sees fragments like ``"ant Error Carousel"``
    (cut from "Constant Error Carousel") and surfaces them as broken
    answers.
    """
    chunks: list[tuple[str, int, int]] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        # Pull the end LEFT to the most recent whitespace so we don't cleave
        # a word — but only when there's still meaningful content remaining.
        if end < n:
            ws = text.rfind(" ", start + chunk_size // 2, end)
            if ws > start:
                end = ws
        chunks.append((text[start:end].strip(), start, end))
        if end >= n:
            break
        # Push next start RIGHT past whitespace so we begin on a word.
        next_start = max(0, end - chunk_overlap)
        if next_start < n:
            ws = text.find(" ", next_start)
            if ws != -1 and ws < end:
                next_start = ws + 1
        start = next_start
    return chunks
