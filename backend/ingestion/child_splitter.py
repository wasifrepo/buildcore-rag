"""Child-chunk splitter for small-to-big (parent-child) retrieval.

BuildCore indexes documents at two granularities:

* **Parent chunks** are the structure-aware units produced by the five
  type-specific chunkers (a full SOP section, a whole contract clause, one
  email message, a complete numbered procedure, a checklist row group).  They
  carry enough surrounding context to answer a question and are what the
  generator ultimately reads.
* **Child chunks** are 2-3 sentence windows carved out of each parent.  They
  are the units that get *embedded* and *BM25-indexed*, so vector and keyword
  matching happen against small, topically-focused text where the signal is
  not diluted by an entire section.

At retrieval time a child hit is resolved back to its parent (see
``retrieval/_parenting.py``) so the pipeline matches precisely but generates
with full context.  This mirrors Azure AI Search "index projections", where a
child document carries a parent key and the parent's fields — keeping the
local ``LocalRetriever`` and the production ``AzureAISearchRetriever`` behaving
the same way.

Losslessness
------------
Children are capped at ``CHILD_MAX_CHARS`` characters.  A single sentence that
exceeds the cap is *hard-wrapped* at word boundaries into multiple children
rather than truncated, so no source text is ever dropped.  Because every child
is comfortably below the embedding model's token limit, the ingestion pipeline
never needs to truncate before embedding.

Configuration (environment variables)
-------------------------------------
* ``CHILD_SENTENCES``  — sentences per child window (default ``3``).
* ``CHILD_OVERLAP``    — sentences shared between adjacent windows (default ``1``).
* ``CHILD_MAX_CHARS``  — hard character ceiling per child (default ``1200``).
"""

import os
import re

from generation.schemas import Chunk
from ingestion.chunkers.base import BaseChunker

# ---------------------------------------------------------------------------
# Defaults (overridable via environment variables)
# ---------------------------------------------------------------------------

_DEFAULT_CHILD_SENTENCES: int = 3
_DEFAULT_CHILD_OVERLAP: int = 1
_DEFAULT_CHILD_MAX_CHARS: int = 1200

# Sentence boundary: a ., ! or ? followed by whitespace and the start of a new
# sentence (an optional opening quote/bracket then an uppercase letter or
# digit).  The lookbehind keeps the terminator with the preceding sentence and
# the lookahead avoids splitting decimals and section numbers like "3.2".
_SENTENCE_BOUNDARY: re.Pattern[str] = re.compile(
    r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_child_chunks(parent: Chunk) -> list[Chunk]:
    """Split a parent chunk into child chunks with parent linkage metadata.

    Each returned child carries, in its ``metadata`` dict, everything needed to
    reconstruct the parent at retrieval time:

    * ``parent_id`` — the parent chunk's ID.
    * ``parent_content`` — the parent's full text (returned to the generator).
    * ``parent_index`` — the parent's ``chunk_index`` within its document.
    * ``child_index`` — this child's zero-based position within the parent.

    The parent's structural metadata (``section_title``, ``clause_id``,
    ``sender``, ``step_number`` …) is copied onto every child so that it is
    available on the reconstructed parent without a second lookup.

    Args:
        parent: A structure-aware :class:`~generation.schemas.Chunk` produced
            by one of the type-specific chunkers.

    Returns:
        Ordered list of child :class:`~generation.schemas.Chunk` objects.  A
        parent that is already short enough yields a single child whose text
        equals the parent's content.  Never returns an empty list for a
        non-empty parent.
    """
    sentences_per_child, overlap, max_chars = _load_config()

    sentences = split_into_sentences(parent.content)
    child_texts = group_into_children(
        sentences, sentences_per_child, overlap, max_chars
    )
    if not child_texts:
        # Defensive fallback: a parent with no detectable sentences (e.g. a
        # single unpunctuated identifier) still gets one child equal to itself.
        child_texts = [parent.content]

    # Parent bookkeeping copied onto every child, minus the parent's own
    # chunk_index (re-exposed as parent_index so reconstruction is unambiguous).
    base_metadata = {
        key: value
        for key, value in parent.metadata.items()
        if key != "chunk_index"
    }
    parent_index = parent.metadata.get("chunk_index", 0)

    children: list[Chunk] = []
    for child_index, text in enumerate(child_texts):
        metadata = dict(base_metadata)
        metadata["parent_id"] = parent.chunk_id
        metadata["parent_content"] = parent.content
        metadata["parent_index"] = parent_index
        metadata["child_index"] = child_index

        child_id = BaseChunker.generate_chunk_id(
            parent.document_id, f"{parent.chunk_id}::{child_index}::{text}"
        )
        children.append(
            Chunk(
                chunk_id=child_id,
                document_id=parent.document_id,
                document_type=parent.document_type,
                content=text,
                metadata=metadata,
            )
        )
    return children


def split_into_sentences(text: str) -> list[str]:
    """Split text into an ordered list of sentence-like units.

    Splitting is line-aware: each non-blank line is treated as at least one
    unit (so line-oriented content such as checklist rows and numbered
    procedure steps stays intact), and prose lines are further divided at
    sentence boundaries.  Blank lines are discarded.

    Args:
        text: The parent chunk content to split.

    Returns:
        List of non-empty, stripped sentence strings in reading order.
    """
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for part in _SENTENCE_BOUNDARY.split(line):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def group_into_children(
    sentences: list[str],
    sentences_per_child: int,
    overlap: int,
    max_chars: int,
) -> list[str]:
    """Group sentences into overlapping child windows bounded by a char cap.

    Sentences are packed into sliding windows of ``sentences_per_child``
    sentences with ``overlap`` sentences shared between adjacent windows (which
    preserves context across window boundaries and improves recall).  A window
    whose joined text would exceed ``max_chars`` is greedily repacked into
    smaller pieces, and any single sentence longer than ``max_chars`` is
    hard-wrapped at word boundaries — nothing is ever truncated.

    Args:
        sentences: Ordered sentence list from :func:`split_into_sentences`.
        sentences_per_child: Target number of sentences per window (>= 1).
        overlap: Sentences shared between consecutive windows (>= 0, and
            strictly less than ``sentences_per_child``).
        max_chars: Hard character ceiling per child.

    Returns:
        Ordered list of child text strings, each at most ``max_chars`` long.
    """
    # Guarantee no single sentence exceeds the cap before windowing.
    normalised: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            normalised.append(sentence)
        else:
            normalised.extend(_hard_wrap(sentence, max_chars))

    if not normalised:
        return []

    step = max(1, sentences_per_child - max(0, overlap))
    children: list[str] = []
    index = 0
    total = len(normalised)
    while index < total:
        window = normalised[index : index + sentences_per_child]
        children.extend(_pack_within_limit(window, max_chars))
        if index + sentences_per_child >= total:
            break
        index += step
    return children


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_config() -> tuple[int, int, int]:
    """Read child-splitting parameters from the environment.

    Returns:
        Tuple ``(sentences_per_child, overlap, max_chars)`` with defaults
        applied and sanity-clamped so that ``overlap < sentences_per_child``
        and both are non-negative.
    """
    sentences_per_child = int(
        os.environ.get("CHILD_SENTENCES", _DEFAULT_CHILD_SENTENCES)
    )
    overlap = int(os.environ.get("CHILD_OVERLAP", _DEFAULT_CHILD_OVERLAP))
    max_chars = int(os.environ.get("CHILD_MAX_CHARS", _DEFAULT_CHILD_MAX_CHARS))

    sentences_per_child = max(1, sentences_per_child)
    overlap = min(max(0, overlap), sentences_per_child - 1)
    max_chars = max(1, max_chars)
    return sentences_per_child, overlap, max_chars


def _pack_within_limit(sentences: list[str], max_chars: int) -> list[str]:
    """Join sentences into as few pieces as possible without exceeding max_chars.

    Args:
        sentences: Sentences (each already <= ``max_chars``) to join.
        max_chars: Hard character ceiling per output piece.

    Returns:
        List of joined strings, each at most ``max_chars`` long.
    """
    pieces: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for sentence in sentences:
        added = len(sentence) + (1 if buffer else 0)
        if buffer and buffer_len + added > max_chars:
            pieces.append(" ".join(buffer))
            buffer = [sentence]
            buffer_len = len(sentence)
        else:
            buffer.append(sentence)
            buffer_len += added
    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Split an over-long string into <= max_chars pieces without losing text.

    Splits primarily at word boundaries.  A single word longer than
    ``max_chars`` (e.g. a very long identifier) is sliced at character
    boundaries so the ceiling always holds.  Concatenating the returned pieces
    with single spaces reproduces the original word sequence.

    Args:
        text: The over-long string to wrap.
        max_chars: Hard character ceiling per piece.

    Returns:
        Ordered list of pieces, each at most ``max_chars`` long.
    """
    pieces: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for word in text.split():
        # A single word longer than the cap is emitted in char-sized slices.
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
