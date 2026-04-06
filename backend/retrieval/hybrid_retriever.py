"""Hybrid retriever — Reciprocal Rank Fusion of dense and sparse results.

Merges the outputs of :class:`~retrieval.dense_retriever.DenseRetriever` and
:class:`~retrieval.sparse_retriever.SparseRetriever` using **Reciprocal Rank
Fusion** (RRF).  RRF is a parameter-light rank aggregation algorithm that
has been shown to outperform score-based fusion across a wide range of
retrieval benchmarks without requiring score normalisation or calibration
between the two systems.

RRF formula
-----------
For each chunk that appears in one or both ranked lists:

    rrf_score(d) = Σ  1 / (k + rank_i(d))
                   i

where ``rank_i(d)`` is the 1-based position of document ``d`` in ranked list
``i``, ``k = 60`` is the standard smoothing constant, and the sum runs over
all lists in which ``d`` appears.  Documents that appear in both lists
receive a higher combined score than documents in only one list.

The RRF score is stored in the returned chunk's ``dense_score`` field so that
it is available to the reranker and trace logger without adding a new schema
field.  The per-source scores (``dense_score`` from the dense retriever,
``sparse_score`` from the sparse retriever) are preserved on the merged chunk
where available.

Document-type filtering
------------------------
The :class:`~generation.schemas.QueryAnalysis` object produced by the query
analyser sometimes specifies a document type restriction in its
``retrieval_strategy`` field (e.g. ``"filter by document_type=contract"``).
This module parses that string for a ``document_type=<value>`` pattern and,
when found, applies a post-hoc filter to the merged results to keep only
chunks of the specified type.
"""

import re

from generation.schemas import Chunk, QueryAnalysis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard RRF smoothing constant.  A value of 60 is widely used in the
# literature and is robust across a variety of retrieval settings.
_RRF_K: int = 60

# Pattern to extract an explicit document_type filter from the query
# analyser's retrieval_strategy string.
# Matches e.g. "document_type=contract", "document_type=safety_sop"
_DOC_TYPE_FILTER_PATTERN: re.Pattern[str] = re.compile(
    r"document_type=([a-z_]+)"
)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def merge_results(
    dense_chunks: list[Chunk],
    sparse_chunks: list[Chunk],
    query_analysis: QueryAnalysis,
    rrf_k: int = _RRF_K,
) -> list[Chunk]:
    """Merge dense and sparse retrieval results using Reciprocal Rank Fusion.

    Accepts pre-retrieved ranked lists from both retrievers, applies RRF
    scoring, deduplicates by ``chunk_id``, optionally applies a document-type
    filter derived from the query analysis, and returns the merged list sorted
    by RRF score descending.

    Args:
        dense_chunks: Ranked list of chunks from the dense retriever, sorted
            by ``dense_score`` descending.
        sparse_chunks: Ranked list of chunks from the sparse retriever, sorted
            by ``sparse_score`` descending.
        query_analysis: The :class:`~generation.schemas.QueryAnalysis` result
            from the query analyser.  Its ``retrieval_strategy`` field is
            inspected for an explicit ``document_type=<value>`` filter.
        rrf_k: RRF smoothing constant.  Defaults to ``60``.  Increasing this
            value reduces the influence of top-ranked documents; decreasing it
            emphasises them.

    Returns:
        Merged, deduplicated list of :class:`~generation.schemas.Chunk`
        objects sorted by RRF score descending.  Each chunk's ``dense_score``
        field is set to the computed RRF score.  The original ``dense_score``
        and ``sparse_score`` values from the respective retrievers are
        preserved in the ``metadata`` dict under the keys
        ``"_orig_dense_score"`` and ``"_orig_sparse_score"`` for trace
        inspection.
    """
    # --- Build lookup tables keyed by chunk_id ---
    # Use dicts to deduplicate and allow O(1) score/content access.
    dense_map: dict[str, Chunk] = {c.chunk_id: c for c in dense_chunks}
    sparse_map: dict[str, Chunk] = {c.chunk_id: c for c in sparse_chunks}
    all_ids: set[str] = set(dense_map) | set(sparse_map)

    # --- Assign 1-based ranks in each list ---
    dense_rank: dict[str, int] = {
        chunk.chunk_id: rank + 1 for rank, chunk in enumerate(dense_chunks)
    }
    sparse_rank: dict[str, int] = {
        chunk.chunk_id: rank + 1 for rank, chunk in enumerate(sparse_chunks)
    }

    # --- Compute RRF scores ---
    rrf_scores: dict[str, float] = {}
    for chunk_id in all_ids:
        score = 0.0
        if chunk_id in dense_rank:
            score += 1.0 / (rrf_k + dense_rank[chunk_id])
        if chunk_id in sparse_rank:
            score += 1.0 / (rrf_k + sparse_rank[chunk_id])
        rrf_scores[chunk_id] = score

    # --- Build merged Chunk objects ---
    merged: list[Chunk] = []
    for chunk_id in all_ids:
        # Prefer the dense result for content/metadata; fall back to sparse.
        base: Chunk = dense_map.get(chunk_id) or sparse_map[chunk_id]
        orig_dense = dense_map[chunk_id].dense_score if chunk_id in dense_map else None
        orig_sparse = sparse_map[chunk_id].sparse_score if chunk_id in sparse_map else None

        # Preserve original scores in metadata for trace transparency.
        augmented_meta = {**base.metadata}
        if orig_dense is not None:
            augmented_meta["_orig_dense_score"] = orig_dense
        if orig_sparse is not None:
            augmented_meta["_orig_sparse_score"] = orig_sparse

        merged.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=base.document_id,
                document_type=base.document_type,
                content=base.content,
                metadata=augmented_meta,
                # Store RRF score as dense_score so the reranker and trace
                # logger have a single pre-rerank quality signal.
                dense_score=rrf_scores[chunk_id],
                sparse_score=orig_sparse,
                rerank_score=None,
            )
        )

    # --- Apply document-type filter if specified in retrieval strategy ---
    doc_type_filter = _extract_doc_type_filter(query_analysis.retrieval_strategy)
    if doc_type_filter:
        filtered = [c for c in merged if c.document_type == doc_type_filter]
        # Fall back to the full list if the filter removes everything —
        # better to return all results than an empty list.
        merged = filtered if filtered else merged

    merged.sort(key=lambda c: c.dense_score or 0.0, reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_doc_type_filter(retrieval_strategy: str) -> str | None:
    """Parse the query analyser's retrieval strategy for a document-type hint.

    Looks for a ``document_type=<value>`` substring that the query analyser
    includes when the query clearly targets a single document type.

    Args:
        retrieval_strategy: Free-text retrieval strategy string from
            :class:`~generation.schemas.QueryAnalysis`.

    Returns:
        The document type value string (e.g. ``"contract"``) if found,
        otherwise ``None``.
    """
    match = _DOC_TYPE_FILTER_PATTERN.search(retrieval_strategy)
    return match.group(1) if match else None
