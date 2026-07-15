"""Shared helpers for parent-child (small-to-big) retrieval.

The ChromaDB collection stores *child* chunks (2-3 sentence windows), but the
retrieval pipeline reasons about *parent* chunks (the structure-aware units the
generator reads).  This module reconstructs a parent
:class:`~generation.schemas.Chunk` from a child result row using the linkage
metadata written by ``ingestion.child_splitter.build_child_chunks``.

Both the dense and sparse retrievers use these helpers so that child→parent
collapsing behaves identically regardless of which retrieval signal produced
the hit.
"""

import json

from generation.schemas import Chunk

# Child-only bookkeeping keys that must not leak onto the reconstructed parent.
_CHILD_ONLY_KEYS: frozenset[str] = frozenset(
    {"parent_id", "parent_content", "child_index"}
)


def try_deserialise_list(value: object) -> object:
    """Deserialise a JSON-array string back to a Python list, else pass through.

    The ingestion pipeline serialises ``list`` metadata values to JSON strings
    before upserting into ChromaDB (see ``pipeline._flatten_metadata``).  This
    reverses that transformation for string values that look like JSON arrays.

    Args:
        value: Metadata value as stored in ChromaDB.

    Returns:
        The parsed list if ``value`` is a valid JSON array string, otherwise
        ``value`` unchanged.
    """
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return value


def parent_from_child(
    child_id: str,
    child_content: str,
    flat_meta: dict,
    score: float,
    kind: str,
) -> Chunk:
    """Reconstruct the parent :class:`~generation.schemas.Chunk` for a child hit.

    Uses the ``parent_id`` / ``parent_content`` / ``parent_index`` linkage
    fields written at ingestion time.  If those fields are absent (e.g. an
    index built before the parent-child migration), the child is treated as its
    own parent so retrieval degrades gracefully rather than failing.

    Args:
        child_id: The matched child chunk's ID (ChromaDB row ID).
        child_content: The matched child's text.
        flat_meta: The child's flat ChromaDB metadata dict.
        score: Relevance score for the hit — cosine similarity for dense,
            normalised BM25 for sparse.
        kind: ``"dense"`` or ``"sparse"``; selects which score field to set on
            the returned parent chunk.

    Returns:
        A parent :class:`~generation.schemas.Chunk` whose ``chunk_id`` is the
        parent ID and whose ``content`` is the parent's full text, with the
        appropriate score field populated.
    """
    parent_id = flat_meta.get("parent_id") or child_id
    content = flat_meta.get("parent_content") or child_content
    document_id = flat_meta.get("document_id", "")
    document_type = flat_meta.get("document_type", "")

    metadata: dict = {}
    for key, value in flat_meta.items():
        if key in ("document_id", "document_type") or key in _CHILD_ONLY_KEYS:
            continue
        if key == "parent_index":
            metadata["chunk_index"] = value
            continue
        metadata[key] = try_deserialise_list(value)

    chunk = Chunk(
        chunk_id=parent_id,
        document_id=document_id,
        document_type=document_type,
        content=content,
        metadata=metadata,
    )
    if kind == "dense":
        chunk.dense_score = score
    elif kind == "sparse":
        chunk.sparse_score = score
    return chunk
