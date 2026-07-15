"""Azure AI Search retriever — production hybrid backend.

The production implementation of the :class:`Retriever` interface, selected
with ``RETRIEVER_BACKEND=azure_ai_search``.  It delegates to a single managed
service what the local stack builds by hand:

===========================  ==========================================
Local (``LocalRetriever``)   Azure AI Search
===========================  ==========================================
ChromaDB vector search       HNSW vector search over ``contentVector``
rank-bm25 keyword search     BM25 over the ``content`` field
``hybrid_retriever`` RRF     native hybrid fusion (also RRF)
MiniLM cross-encoder         managed semantic ranker
===========================  ==========================================

One request does all of it.  Passing ``search_text`` *and* ``vector_queries``
to a single ``search`` call makes Azure run both signals and fuse them with
RRF server-side; adding ``query_type="semantic"`` layers the semantic ranker on
top of the fused list.  The local pipeline needs four components and two
round-trips to achieve the same shape.

Multi-query expansion maps cleanly onto this: every expanded variant becomes
its own ``VectorizedQuery`` in the same request, and Azure fuses all of them
together with the keyword hits.  That replaces the local retriever's
best-score-per-variant merge loop.

Small-to-big collapsing
-----------------------
The index stores *child* chunks (see ``ingestion.azure_index``).  Search
matches children; this retriever collapses hits back to parents by
``parent_id``, keeping the best-scoring child per parent, exactly as
``LocalRetriever`` does via ``retrieval._parenting``.  Because many children
map to one parent, it over-fetches (``top_k × CHILD_FETCH_MULTIPLIER``) so
enough distinct parents survive the collapse — the same compensation the local
dense retriever applies.

Score mapping
-------------
Azure fuses dense and sparse internally and does not expose the two
contributions separately, so a per-signal breakdown is not recoverable.  The
fused ``@search.score`` is stored in ``dense_score``, which matches the
convention the local pipeline already uses (``hybrid_retriever.merge_results``
also writes its RRF score into ``dense_score``), and ``sparse_score`` is left
``None``.  When the semantic ranker runs, its ``@search.rerankerScore`` is
stored in ``rerank_score`` — the same field the local cross-encoder writes —
so traces and the UI render identically across backends.

Reranking
---------
When ``AZURE_SEARCH_SEMANTIC_CONFIG`` is set, results are already semantically
reranked on arrival and the pipeline's rerank step should be a pass-through
(set ``RERANKER_BACKEND=passthrough``; see ``retrieval.reranker_factory``).
Leave the semantic config unset to fall back to the local cross-encoder.

Configuration (environment variables)
-------------------------------------
* ``AZURE_SEARCH_ENDPOINT``        — ``https://<name>.search.windows.net``.
* ``AZURE_SEARCH_INDEX``           — index name (default ``"buildcore"``).
* ``AZURE_SEARCH_API_KEY``         — query key; omit to use Managed Identity.
* ``AZURE_SEARCH_SEMANTIC_CONFIG`` — semantic configuration name; enables the
  managed semantic ranker when set.
* ``AZURE_SEARCH_VECTOR_FIELD``    — child embedding field.
* ``AZURE_SEARCH_PARENT_ID_FIELD`` — parent key field.
* ``CHILD_FETCH_MULTIPLIER``       — child over-fetch factor (default ``4``).
* ``TOP_K_DENSE``                  — parent budget (default ``20``).
"""

import json
import logging
import os

from azure.search.documents.models import VectorizedQuery

from common.llm_client import embed_texts, get_embedding_model
from generation.schemas import Chunk, QueryAnalysis
from ingestion.azure_index import (
    get_parent_id_field,
    get_search_client,
    get_vector_field,
)
from retrieval.base import Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K: int = 20
_DEFAULT_CHILD_FETCH_MULTIPLIER: int = 4


class AzureAISearchRetriever(Retriever):
    """Hybrid retriever backed by Azure AI Search.

    Instantiate once at application startup and share across requests; the
    underlying ``SearchClient`` holds a connection pool and is thread-safe.

    Args:
        semantic_config: Semantic configuration name.  Falls back to
            ``AZURE_SEARCH_SEMANTIC_CONFIG``.  When ``None``, the semantic
            ranker is not requested and plain hybrid fusion is returned.
    """

    def __init__(self, semantic_config: str | None = None) -> None:
        """Build the search client and resolve field/ranking configuration.

        Args:
            semantic_config: Optional semantic configuration name override.
        """
        self._client = get_search_client()
        self._semantic_config = semantic_config or os.environ.get(
            "AZURE_SEARCH_SEMANTIC_CONFIG"
        )
        self._vector_field = get_vector_field()
        self._parent_id_field = get_parent_id_field()
        self._embed_model = get_embedding_model()

    def retrieve(
        self,
        queries: list[str],
        original_query: str,
        query_analysis: QueryAnalysis,
        top_k: int | None = None,
    ) -> list[Chunk]:
        """Run one hybrid Azure AI Search query and collapse children to parents.

        Embeds every query variant, issues a single request combining those
        vector queries with a BM25 search on ``original_query``, optionally
        applies the semantic ranker, then collapses the matched children to
        their parents.

        Args:
            queries: Original query plus expanded variants (each embedded).
            original_query: Unmodified user query, used for the keyword leg.
            query_analysis: Consulted for a ``document_type`` filter hint.
            top_k: Optional cap on distinct parents returned.  Falls back to
                ``TOP_K_DENSE``.

        Returns:
            Parent :class:`~generation.schemas.Chunk` objects sorted by
            relevance descending, at most ``top_k`` long.
        """
        resolved_top_k = top_k or int(os.environ.get("TOP_K_DENSE", _DEFAULT_TOP_K))
        child_fetch = resolved_top_k * int(
            os.environ.get("CHILD_FETCH_MULTIPLIER", _DEFAULT_CHILD_FETCH_MULTIPLIER)
        )

        vector_queries = [
            VectorizedQuery(
                vector=vector,
                k_nearest_neighbors=child_fetch,
                fields=self._vector_field,
            )
            for vector in embed_texts(queries, model=self._embed_model)
        ]

        search_kwargs: dict = {
            "search_text": original_query,
            "vector_queries": vector_queries,
            "top": child_fetch,
            "select": [
                "id",
                "content",
                "parent_content",
                "parent_index",
                "document_id",
                "document_type",
                "metadata_json",
                self._parent_id_field,
            ],
        }

        if query_analysis.document_type_filter:
            # Enum values are fixed identifiers, but escape defensively anyway:
            # OData string literals double a single quote to escape it.
            escaped = query_analysis.document_type_filter.value.replace("'", "''")
            search_kwargs["filter"] = f"document_type eq '{escaped}'"

        if self._semantic_config:
            search_kwargs["query_type"] = "semantic"
            search_kwargs["semantic_configuration_name"] = self._semantic_config

        results = self._client.search(**search_kwargs)
        return self._collapse_to_parents(results, resolved_top_k)

    def _collapse_to_parents(self, results, top_k: int) -> list[Chunk]:
        """Collapse child search results into ranked parent chunks.

        Keeps the best-scoring child per parent.  Ranking prefers the semantic
        reranker score when present and falls back to the fused hybrid score,
        so ordering is consistent whether or not the semantic ranker ran.

        Args:
            results: Iterable of child result rows from the search call.
            top_k: Maximum number of distinct parents to return.

        Returns:
            Parent chunks sorted by relevance descending, at most ``top_k``.
        """
        best: dict[str, tuple[float, Chunk]] = {}

        for row in results:
            search_score = row.get("@search.score") or 0.0
            reranker_score = row.get("@search.rerankerScore")
            # Semantic scores (0–4) and RRF scores are on different scales;
            # never mix them within one ranking pass.
            ranking_score = (
                reranker_score if reranker_score is not None else search_score
            )

            parent_id = row.get(self._parent_id_field) or row.get("id")
            existing = best.get(parent_id)
            if existing is not None and existing[0] >= ranking_score:
                continue

            chunk = Chunk(
                chunk_id=parent_id,
                document_id=row.get("document_id", ""),
                document_type=row.get("document_type", ""),
                content=row.get("parent_content") or row.get("content", ""),
                metadata=self._rebuild_metadata(row),
                dense_score=search_score,
                rerank_score=reranker_score,
            )
            best[parent_id] = (ranking_score, chunk)

        ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]

    @staticmethod
    def _rebuild_metadata(row) -> dict:
        """Rehydrate a parent's metadata dict from a child result row.

        Reverses the JSON encoding applied by
        :func:`ingestion.azure_index._to_search_document` and restores
        ``chunk_index`` from ``parent_index``, so the resulting chunk carries
        the same metadata shape the local backend produces.

        Args:
            row: A child search result row.

        Returns:
            The parent chunk's metadata dict.
        """
        raw = row.get("metadata_json")
        metadata: dict = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                logger.warning(
                    "Malformed metadata_json on document '%s'; ignoring.",
                    row.get("id"),
                )

        parent_index = row.get("parent_index")
        if parent_index is not None:
            metadata["chunk_index"] = parent_index
        return metadata
