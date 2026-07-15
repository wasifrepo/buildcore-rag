"""Dense retriever for the BuildCore RAG pipeline.

Embeds one or more query strings using OpenAI ``text-embedding-3-small``,
queries the ChromaDB vector store for the nearest neighbours of each query,
and returns a deduplicated, score-ranked list of
:class:`~generation.schemas.Chunk` objects with ``dense_score`` populated.

Multi-query deduplication
--------------------------
When the query expander supplies expanded variants, all queries are embedded
and queried independently.  Results are merged by ``chunk_id``; when the same
chunk appears in results for multiple query variants the *highest* cosine
similarity score across all variants is retained (optimistic deduplication).
The final list is sorted by ``dense_score`` descending.

ChromaDB distance → similarity
-------------------------------
The collection is created with ``hnsw:space: cosine``, so ChromaDB returns
distances in the range ``[0, 2]`` where ``0`` means identical and ``2`` means
maximally dissimilar.  The conversion used here is:

    dense_score = 1.0 − distance

For OpenAI embeddings (which are unit-normalised), this yields values in
``[−1, 1]`` that are monotonically equivalent to cosine similarity.

Small-to-big (parent-child) collapsing
--------------------------------------
The ChromaDB collection stores *child* chunks (2-3 sentence windows).  Vector
search matches children, but this retriever returns *parent* chunks: each child
hit is resolved to its parent via ``retrieval._parenting.parent_from_child`` and
duplicates are collapsed by parent ID, keeping the best cosine similarity seen
across the parent's children.  Because many children map to one parent, the
retriever over-fetches children (``top_k × CHILD_FETCH_MULTIPLIER``) so that
enough distinct parents survive the collapse.

Metadata reconstruction
------------------------
The ingestion pipeline serialises ``list`` metadata values to JSON strings
before upserting into ChromaDB (see ``pipeline._flatten_metadata``).  The
parent-reconstruction helper reverses that transformation: any string metadata
value that parses as a JSON array is converted back to a Python list.
"""

import os

import chromadb
from common.llm_client import embed_texts, get_embedding_model
from generation.schemas import Chunk
from retrieval._parenting import parent_from_child

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K: int = 20
_DEFAULT_EMBED_MODEL: str = "text-embedding-3-small"
# Children fetched per query variant = top_k × this multiplier, so that after
# collapsing children to parents there are still ~top_k distinct parents.
_DEFAULT_CHILD_FETCH_MULTIPLIER: int = 4


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class DenseRetriever:
    """Retrieves chunks from ChromaDB using dense vector similarity search.

    Maintains an OpenAI client for embedding and a ChromaDB collection
    handle for querying.  Both are initialised once in ``__init__`` and
    reused across calls; the class is safe to instantiate once at application
    startup and share across requests.

    Args:
        chroma_persist_dir: Filesystem path for ChromaDB storage.  Defaults
            to the ``CHROMA_PERSIST_DIR`` environment variable.
        chroma_collection_name: ChromaDB collection name.  Defaults to the
            ``CHROMA_COLLECTION_NAME`` environment variable.
    """

    def __init__(
        self,
        chroma_persist_dir: str | None = None,
        chroma_collection_name: str | None = None,
    ) -> None:
        """Initialise the OpenAI client and attach to the ChromaDB collection.

        Args:
            chroma_persist_dir: Path to the ChromaDB persistence directory.
                Falls back to the ``CHROMA_PERSIST_DIR`` environment variable,
                then to ``./data/chroma``.
            chroma_collection_name: Name of the ChromaDB collection to query.
                Falls back to ``CHROMA_COLLECTION_NAME``, then ``"buildcore"``.
        """
        self._embed_model = get_embedding_model()

        persist_dir = chroma_persist_dir or os.environ.get(
            "CHROMA_PERSIST_DIR", "./data/chroma"
        )
        collection_name = chroma_collection_name or os.environ.get(
            "CHROMA_COLLECTION_NAME", "buildcore"
        )
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(
        self,
        queries: list[str],
        top_k: int | None = None,
        document_type_filter: str | None = None,
    ) -> list[Chunk]:
        """Embed each query, search ChromaDB, and return deduplicated results.

        Embeds all queries in a single batched API call, queries ChromaDB
        independently for each query embedding, then merges results across
        all queries by keeping the highest ``dense_score`` per ``chunk_id``.

        Args:
            queries: One or more query strings to embed and search.  Typically
                the original query plus the expanded variants produced by
                :func:`~retrieval.query_expander.expand_query`.
            top_k: Maximum number of results to return per query variant
                before deduplication.  Defaults to the ``TOP_K_DENSE``
                environment variable (default ``20``).
            document_type_filter: If provided, restricts the search to chunks
                whose ``document_type`` metadata field equals this value
                (e.g. ``"contract"``, ``"safety_sop"``).  Maps to a ChromaDB
                ``where`` filter clause.

        Returns:
            Deduplicated list of parent :class:`~generation.schemas.Chunk`
            objects sorted by ``dense_score`` descending, at most
            ``resolved_top_k`` long.  Each parent's ``dense_score`` is the best
            cosine similarity observed across its children and query variants.
        """
        resolved_top_k = top_k or int(os.environ.get("TOP_K_DENSE", _DEFAULT_TOP_K))
        child_fetch = resolved_top_k * int(
            os.environ.get("CHILD_FETCH_MULTIPLIER", _DEFAULT_CHILD_FETCH_MULTIPLIER)
        )

        # Embed all queries in one API call
        embeddings = embed_texts(queries, model=self._embed_model)

        where_filter: dict | None = None
        if document_type_filter:
            where_filter = {"document_type": {"$eq": document_type_filter}}

        # Query ChromaDB for each embedding, collapse child hits to parents, and
        # keep the best score per parent across all query variants.
        best: dict[str, Chunk] = {}  # parent_id → best parent Chunk seen so far

        for embedding in embeddings:
            query_kwargs: dict = {
                "query_embeddings": [embedding],
                "n_results": child_fetch,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                query_kwargs["where"] = where_filter

            result = self._collection.query(**query_kwargs)

            ids: list[str] = result["ids"][0]
            documents: list[str] = result["documents"][0]
            metadatas: list[dict] = result["metadatas"][0]
            distances: list[float] = result["distances"][0]

            for child_id, content, flat_meta, distance in zip(
                ids, documents, metadatas, distances
            ):
                dense_score = 1.0 - distance
                parent = parent_from_child(
                    child_id, content, flat_meta, dense_score, "dense"
                )
                existing = best.get(parent.chunk_id)
                if existing is not None and (existing.dense_score or 0.0) >= dense_score:
                    continue
                best[parent.chunk_id] = parent

        ranked = sorted(
            best.values(), key=lambda c: c.dense_score or 0.0, reverse=True
        )
        return ranked[:resolved_top_k]
