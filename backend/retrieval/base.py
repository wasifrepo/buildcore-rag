"""Retrieval adapter interfaces for the BuildCore RAG pipeline.

The pipeline depends on two abstractions so that the retrieval substrate can be
swapped without touching orchestration code:

* :class:`Retriever` — turns a set of query strings into a ranked list of
  candidate *parent* chunks (hybrid dense + sparse fusion).  The local
  implementation composes ChromaDB, BM25, and RRF; the production
  implementation delegates to Azure AI Search (vector + keyword hybrid).
* :class:`Reranker` — reorders candidate chunks for a single query.  The local
  implementation is the cross-encoder in ``retrieval.reranker``; in production
  this role can be filled by the Azure AI Search semantic ranker.

Keeping these as explicit interfaces is the "adapter" seam that lets the same
pipeline run against a hand-built local stack in development and a managed Azure
backend in production, with the evaluation harness proving parity between them.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from generation.schemas import Chunk, QueryAnalysis


class Retriever(ABC):
    """Abstract hybrid retriever returning ranked parent chunks for a query.

    Implementations own whatever combination of dense, sparse, and fusion
    logic their backend requires; the pipeline only relies on
    :meth:`retrieve`.
    """

    @abstractmethod
    def retrieve(
        self,
        queries: list[str],
        original_query: str,
        query_analysis: QueryAnalysis,
        top_k: int | None = None,
    ) -> list[Chunk]:
        """Return ranked candidate parent chunks for a query.

        Args:
            queries: The original query plus expanded variants.  Backends that
                perform dense retrieval typically embed all of these; sparse
                retrieval typically uses ``original_query`` only.
            original_query: The unmodified user query, preferred for keyword
                retrieval and as the reranker input.
            query_analysis: The :class:`~generation.schemas.QueryAnalysis` for
                this query, consulted for document-type filter hints.
            top_k: Optional cap on candidates per retrieval signal before
                fusion.  ``None`` lets the backend use its configured default.

        Returns:
            Ranked list of parent :class:`~generation.schemas.Chunk` objects,
            highest-ranked first, ready to be reranked.
        """

    def refresh(self) -> None:
        """Reload any in-memory index after re-ingestion.

        Backends with an in-process index (e.g. the local BM25 index) override
        this to rebuild it.  Stateless/managed backends inherit this no-op.
        """


@runtime_checkable
class Reranker(Protocol):
    """Structural type for a query-aware reranker.

    Any object exposing a compatible ``rerank`` method satisfies this
    protocol; the concrete
    :class:`~retrieval.reranker.CrossEncoderReranker` conforms without needing
    to inherit from it.
    """

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[Chunk]:
        """Reorder ``chunks`` by relevance to ``query`` and return the top-k."""
        ...
