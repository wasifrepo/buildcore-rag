"""Local hybrid retriever — ChromaDB dense + BM25 sparse + RRF fusion.

This is the development / self-hosted implementation of the :class:`Retriever`
interface.  It composes the three hand-built retrieval layers:

1. :class:`~retrieval.dense_retriever.DenseRetriever` — vector search over the
   child chunks in ChromaDB, collapsed to parents.
2. :class:`~retrieval.sparse_retriever.SparseRetriever` — BM25 keyword search
   over the same child corpus, collapsed to parents.
3. :func:`~retrieval.hybrid_retriever.merge_results` — Reciprocal Rank Fusion
   of the two parent lists.

It is the default backend (``RETRIEVER_BACKEND=local``) and the reference
against which the production
:class:`~retrieval.azure_ai_search_retriever.AzureAISearchRetriever` is
evaluated for parity.
"""

import re

from generation.schemas import Chunk, QueryAnalysis
from retrieval.base import Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import merge_results
from retrieval.sparse_retriever import SparseRetriever

# Pattern matching a ``document_type=<value>`` hint in the query analyser's
# retrieval_strategy string (e.g. "document_type=contract").
_DOC_TYPE_FILTER_RE: re.Pattern[str] = re.compile(r"document_type=([a-z_]+)")


class LocalRetriever(Retriever):
    """Hybrid retriever backed by local ChromaDB + BM25.

    Instantiate once at application startup and share across requests: the
    ChromaDB connection and the in-memory BM25 index are built in ``__init__``.

    Args:
        dense: Optional pre-built :class:`DenseRetriever`.  Constructed with
            environment defaults if omitted.
        sparse: Optional pre-built :class:`SparseRetriever`.  Constructed with
            environment defaults if omitted.
    """

    def __init__(
        self,
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
    ) -> None:
        """Build (or accept) the dense and sparse retrievers.

        Args:
            dense: Optional dense retriever instance.
            sparse: Optional sparse retriever instance.
        """
        self._dense = dense or DenseRetriever()
        self._sparse = sparse or SparseRetriever()

    def retrieve(
        self,
        queries: list[str],
        original_query: str,
        query_analysis: QueryAnalysis,
        top_k: int | None = None,
    ) -> list[Chunk]:
        """Run dense + sparse retrieval and fuse the results with RRF.

        Dense retrieval embeds every query variant; sparse retrieval uses the
        original query only (expanded variants over-count keyword matches).
        A ``document_type=`` hint in the query analysis is applied as a dense
        ``where`` filter.

        Args:
            queries: Original query plus expanded variants (for dense search).
            original_query: Unmodified user query (for BM25 search).
            query_analysis: Drives the optional document-type filter and RRF
                merge behaviour.
            top_k: Optional per-signal candidate cap before fusion.

        Returns:
            RRF-fused list of parent :class:`~generation.schemas.Chunk` objects
            sorted by fused score descending.
        """
        doc_type_filter = self._extract_doc_type_filter(
            query_analysis.retrieval_strategy
        )
        dense_chunks = self._dense.search(queries, top_k, doc_type_filter)
        sparse_chunks = self._sparse.search(original_query, top_k)
        return merge_results(dense_chunks, sparse_chunks, query_analysis)

    def refresh(self) -> None:
        """Rebuild the in-memory BM25 index after re-ingestion."""
        self._sparse.rebuild_index()

    @staticmethod
    def _extract_doc_type_filter(retrieval_strategy: str) -> str | None:
        """Parse a ``document_type=<value>`` hint from a retrieval strategy.

        Args:
            retrieval_strategy: Free-text strategy string from the query
                analyser.

        Returns:
            The document type value (e.g. ``"contract"``) if present, else
            ``None``.
        """
        match = _DOC_TYPE_FILTER_RE.search(retrieval_strategy)
        return match.group(1) if match else None
