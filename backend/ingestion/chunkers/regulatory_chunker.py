"""Regulatory document chunker for OSHA and similar standards documents.

OSHA documents use a mix of numeric section headers (``1.``, ``1.1``,
``1.1.1``), Roman-numeral headers (``I.``, ``II.``, ``III.``), and ALL-CAPS
headings.  This chunker splits at *top-level* section boundaries only —
numeric top-level sections are those whose label has no dot after the first
numeral (``1.``, ``2.`` etc.), and Roman-numeral sections (``I.``, ``II.``
etc.) are always top-level.  ALL-CAPS headings on their own line are also
treated as top-level boundaries.

Chunking strategy
-----------------
1. Extract a document title from the first non-blank line of the text.
2. Walk the text line-by-line, collecting lines into the current section
   buffer.  Start a new section whenever a top-level header line is found.
3. If a completed section exceeds ``_MAX_CHUNK_CHARS`` (1 500) characters,
   split it at paragraph boundaries (blank lines), falling back to line and
   then word boundaries so the ceiling always holds.  Each sub-chunk inherits
   the parent section's metadata.
4. Any trailing content after the last header becomes its own chunk.

PDF-sourced text
----------------
The fallback in step 3 is load-bearing rather than defensive.  ``pypdf``
emits a newline per *rendered* line and virtually no blank lines (OSHA 3150
extracts to 4 blank lines across 574), so blank-line paragraph splitting finds
no boundaries in PDF text and would return the entire section unsplit.  Chunks
of 45,000+ characters then reach the generator and crowd everything else out
of its context.  Line-boundary packing is the right unit for this text because
rendered lines are the only structure PDF extraction preserves.

Each chunk carries:
    - ``section_number``: the parsed header label (e.g. ``"1."``, ``"II."``,
      ``"DEFINITIONS"``), or ``""`` for preamble text.
    - ``section_title``: the heading text following the section number, or
      the ALL-CAPS heading itself.
    - ``document_title``: title extracted from the first line of the document.
    - ``has_numbered_list``: ``True`` if the chunk body contains at least one
      numbered list item (line starting with a digit followed by ``.`` or
      ``)``, e.g. ``1.`` or ``1)``).
"""

import re

from generation.schemas import Chunk
from ingestion.chunkers.base import BaseChunker

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Top-level numeric header: "1.", "12." — digit(s) followed by a period then
# whitespace or end-of-line.  Must NOT match sub-section labels like "1.1".
_NUMERIC_TOP_LEVEL: re.Pattern[str] = re.compile(
    r"^(\d+)\.\s+(.*)$"
)
_NUMERIC_SUB_SECTION: re.Pattern[str] = re.compile(r"^\d+\.\d")

# Roman-numeral top-level header: "I.", "II.", "III.", "IV.", "V.", etc.
_ROMAN_TOP_LEVEL: re.Pattern[str] = re.compile(
    r"^((?:X{0,3})(?:IX|IV|V?I{0,3}))\.\s+(.*)",
    re.IGNORECASE,
)

# ALL-CAPS heading: a line of 4+ characters where every alphabetic character
# is uppercase, containing at least one letter, with no lowercase letters.
# Allows spaces, digits, hyphens, parentheses, and common punctuation.
_ALL_CAPS_HEADING: re.Pattern[str] = re.compile(
    r"^[A-Z][A-Z0-9 \t\-/()&:,.]{3,}$"
)

# Numbered list item: line starting with "1." / "1)" / "(1)"
_NUMBERED_LIST_ITEM: re.Pattern[str] = re.compile(
    r"^\s*(?:\(\d+\)|\d+[.)]\s)"
)

# Maximum characters per output chunk before paragraph-splitting is applied.
_MAX_CHUNK_CHARS: int = 1500


# ---------------------------------------------------------------------------
# Helper dataclass (plain dict equivalent — no external deps)
# ---------------------------------------------------------------------------


class _Section:
    """Holds the accumulated lines and metadata for one top-level section."""

    __slots__ = ("number", "title", "lines")

    def __init__(self, number: str, title: str) -> None:
        self.number = number
        self.title = title
        self.lines: list[str] = []

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class RegulatoryChunker(BaseChunker):
    """Chunker for OSHA regulatory documents and similar standards texts.

    Receives already-extracted plain text (PDF extraction is handled by the
    ingestion pipeline before this class is called) and splits it at
    top-level section boundaries.  Oversized sections are further split at
    paragraph boundaries so that no chunk exceeds ``_MAX_CHUNK_CHARS``.
    """

    def chunk(self, content: str, metadata: dict) -> list[Chunk]:
        """Split regulatory document text into section-boundary chunks.

        Args:
            content: Full plain-text content of the document, already
                extracted from the source PDF or .txt file.
            metadata: Dict containing at minimum ``document_id``,
                ``document_type``, and ``source_path``.

        Returns:
            Ordered list of :class:`~generation.schemas.Chunk` objects.
        """
        document_id: str = metadata["document_id"]
        document_type: str = metadata["document_type"]
        source_path: str = metadata["source_path"]

        text = self.clean_whitespace(content)
        document_title = _extract_title(text)
        sections = _split_into_sections(text)

        chunks: list[Chunk] = []
        chunk_index = 0

        for section in sections:
            body = section.text.strip()
            if not body:
                continue

            sub_texts = _split_at_paragraphs(body, _MAX_CHUNK_CHARS)
            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue
                extra: dict = {
                    "section_number": section.number,
                    "section_title": section.title,
                    "document_title": document_title,
                    "has_numbered_list": _has_numbered_list(sub_text),
                }
                chunks.append(
                    self.build_chunk(
                        content=sub_text,
                        document_id=document_id,
                        document_type=document_type,
                        source_path=source_path,
                        chunk_index=chunk_index,
                        extra_metadata=extra,
                    )
                )
                chunk_index += 1

        return chunks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_title(text: str) -> str:
    """Return the first non-blank line of the document as the title.

    Args:
        text: Cleaned full document text.

    Returns:
        The first non-blank line, or an empty string if the document is blank.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _is_top_level_header(line: str) -> tuple[str, str] | None:
    """Test whether a line is a top-level section header.

    Checks numeric (``1.``), Roman-numeral (``II.``), and ALL-CAPS heading
    patterns in that order.  Sub-section labels (``1.1``, ``1.1.1``) are
    excluded from numeric matching.

    Args:
        line: A single stripped line of text.

    Returns:
        A ``(number, title)`` tuple if the line is a top-level header, or
        ``None`` if it is not.
    """
    stripped = line.strip()

    # Numeric top-level — exclude sub-sections like "1.1"
    if not _NUMERIC_SUB_SECTION.match(stripped):
        m = _NUMERIC_TOP_LEVEL.match(stripped)
        if m:
            return m.group(1) + ".", m.group(2).strip()

    # Roman-numeral top-level
    m = _ROMAN_TOP_LEVEL.match(stripped)
    if m:
        label = m.group(1).upper() + "."
        return label, m.group(2).strip()

    # ALL-CAPS heading (at least 4 characters, no lowercase letters)
    if _ALL_CAPS_HEADING.match(stripped) and stripped != stripped.lower():
        return "", stripped

    return None


def _split_into_sections(text: str) -> list[_Section]:
    """Walk the document text and group lines into top-level sections.

    Args:
        text: Cleaned full document text.

    Returns:
        Ordered list of :class:`_Section` objects, one per top-level
        section plus an optional preamble section at index 0.
    """
    sections: list[_Section] = []
    current = _Section(number="", title="preamble")

    for line in text.splitlines():
        header = _is_top_level_header(line)
        if header is not None:
            # Save the current section (even if empty — filtered later)
            sections.append(current)
            number, title = header
            current = _Section(number=number, title=title)
        else:
            current.lines.append(line)

    sections.append(current)
    return sections


def _split_at_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split an oversized section into chunks of at most ``max_chars``.

    Splits on blank-line paragraph boundaries first, since those are the most
    meaningful break points, and greedily packs the resulting paragraphs.

    Any paragraph that alone exceeds ``max_chars`` is broken down further by
    :func:`_split_long_block` rather than emitted oversized.  That fallback is
    not an edge case: text extracted from PDFs contains a newline per *visual*
    line but almost no blank lines (OSHA 3150 yields 4 blank lines across 574),
    so paragraph splitting finds nothing to split on and the whole section
    arrives here as one "paragraph".  Without the fallback the character
    ceiling silently does not apply to any PDF-sourced document — sections of
    45,000+ characters reach the generator, swamping its context with mostly
    irrelevant text.

    Args:
        text: Section body text to split.
        max_chars: Maximum character count per output chunk.

    Returns:
        List of text strings, each no longer than ``max_chars``.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    # Guarantee every unit is within the cap before packing.
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(_split_long_block(para, max_chars))

    result: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for unit in units:
        unit_len = len(unit)
        # +2 for the "\n\n" separator that would join them
        if buffer and buffer_len + 2 + unit_len > max_chars:
            result.append("\n\n".join(buffer))
            buffer = []
            buffer_len = 0
        buffer.append(unit)
        buffer_len += (2 if buffer_len else 0) + unit_len

    if buffer:
        result.append("\n\n".join(buffer))

    return result


def _split_long_block(text: str, max_chars: int) -> list[str]:
    """Break a paragraph with no blank lines into <= max_chars pieces.

    Packs whole lines greedily, because PDF-extracted text preserves one line
    per rendered line and those are the only structural boundaries available.
    A single line longer than ``max_chars`` is wrapped at word boundaries, and
    a single word longer than the cap is sliced at character boundaries so the
    ceiling always holds.

    No text is discarded: concatenating the returned pieces reproduces the
    input's word sequence.

    Args:
        text: An over-long block containing no blank-line boundaries.
        max_chars: Maximum character count per output piece.

    Returns:
        Ordered list of pieces, each at most ``max_chars`` long.
    """
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            units.append(line)
        else:
            units.extend(_wrap_words(line, max_chars))

    pieces: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for unit in units:
        added = len(unit) + (1 if buffer else 0)
        if buffer and buffer_len + added > max_chars:
            pieces.append("\n".join(buffer))
            buffer = [unit]
            buffer_len = len(unit)
        else:
            buffer.append(unit)
            buffer_len += added
    if buffer:
        pieces.append("\n".join(buffer))
    return pieces


def _wrap_words(text: str, max_chars: int) -> list[str]:
    """Wrap a single over-long line at word boundaries without losing text.

    Args:
        text: The over-long line.
        max_chars: Maximum character count per output piece.

    Returns:
        Ordered list of pieces, each at most ``max_chars`` long.
    """
    pieces: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for word in text.split():
        if len(word) > max_chars:
            if buffer:
                pieces.append(" ".join(buffer))
                buffer = []
                buffer_len = 0
            for start in range(0, len(word), max_chars):
                pieces.append(word[start : start + max_chars])
            continue
        added = len(word) + (1 if buffer else 0)
        if buffer and buffer_len + added > max_chars:
            pieces.append(" ".join(buffer))
            buffer = [word]
            buffer_len = len(word)
        else:
            buffer.append(word)
            buffer_len += added
    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def _has_numbered_list(text: str) -> bool:
    """Return True if any line in text looks like a numbered list item.

    Args:
        text: Chunk body text.

    Returns:
        ``True`` if at least one line matches ``_NUMBERED_LIST_ITEM``.
    """
    return any(_NUMBERED_LIST_ITEM.match(line) for line in text.splitlines())
