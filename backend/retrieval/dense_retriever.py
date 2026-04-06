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

Metadata reconstruction
------------------------
The ingestion pipeline serialises ``list`` metadata values to JSON strings
before upserting into ChromaDB (see ``pipeline._flatten_metadata``).  This
module reverses that transformation: any string metadata value that parses as
a JSON array is converted back to a Python list.
"""

import json
import os

import chromadb
from openai import OpenAI

from generation.schemas import Chunk

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K: int = 20
_DEFAULT_EMBED_MODEL: str = "text-embedding-3-small"


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
        self._embed_model = os.environ.get("EMBEDDING_MODEL", _DEFAULT_EMBED_MODEL)
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
            Deduplicated list of :class:`~generation.schemas.Chunk` objects
            sorted by ``dense_score`` descending.  Each chunk's ``dense_score``
            is the best cosine similarity observed across all query variants.
        """
        resolved_top_k = top_k or int(os.environ.get("TOP_K_DENSE", _DEFAULT_TOP_K))

        # Embed all queries in one API call
        embed_response = self._openai.embeddings.create(
            model=self._embed_model,
            input=queries,
        )
        embeddings = [item.embedding for item in embed_response.data]

        where_filter: dict | None = None
        if document_type_filter:
            where_filter = {"document_type": {"$eq": document_type_filter}}

        # Query ChromaDB for each embedding and merge results
        best: dict[str, Chunk] = {}  # chunk_id → best Chunk seen so far

        for embedding in embeddings:
            query_kwargs: dict = {
                "query_embeddings": [embedding],
                "n_results": resolved_top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                query_kwargs["where"] = where_filter

            result = self._collection.query(**query_kwargs)

            ids: list[str] = result["ids"][0]
            documents: list[str] = result["documents"][0]
            metadatas: list[dict] = result["metadatas"][0]
            distances: list[float] = result["distances"][0]

            for chunk_id, content, flat_meta, distance in zip(
                ids, documents, metadatas, distances
            ):
                dense_score = 1.0 - distance
                if chunk_id in best and best[chunk_id].dense_score >= dense_score:
                    continue
                chunk = _build_chunk_from_chroma(
                    chunk_id=chunk_id,
                    content=content,
                    flat_meta=flat_meta,
                    dense_score=dense_score,
                )
                best[chunk_id] = chunk

        return sorted(best.values(), key=lambda c: c.dense_score or 0.0, reverse=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_chunk_from_chroma(
    chunk_id: str,
    content: str,
    flat_meta: dict,
    dense_score: float,
) -> Chunk:
    """Reconstruct a :class:`~generation.schemas.Chunk` from a ChromaDB result row.

    Extracts ``document_id`` and ``document_type`` from the flat metadata,
    deserialises any JSON-array strings back to Python lists, and packages
    the remainder of the metadata into the chunk's ``metadata`` dict.

    Args:
        chunk_id: ChromaDB document ID (equals the chunk's SHA-256-derived ID).
        content: Raw chunk text as stored in ChromaDB.
        flat_meta: Flat metadata dict as returned by ChromaDB.  List values
            are stored as JSON strings and are deserialised here.
        dense_score: Pre-computed cosine similarity score (``1.0 − distance``).

    Returns:
        A fully populated :class:`~generation.schemas.Chunk` instance.
    """
    document_id: str = flat_meta.get("document_id", "")
    document_type: str = flat_meta.get("document_type", "")

    # Reconstruct chunk metadata — exclude the top-level fields that are
    # stored separately on the Chunk model itself.
    metadata: dict = {}
    for key, value in flat_meta.items():
        if key in ("document_id", "document_type"):
            continue
        metadata[key] = _try_deserialise_list(value)

    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_type=document_type,
        content=content,
        metadata=metadata,
        dense_score=dense_score,
    )


def _try_deserialise_list(value: object) -> object:
    """Attempt to deserialise a JSON-array string back to a Python list.

    The ingestion pipeline serialises list metadata values to JSON strings
    (e.g. ``'["item1", "item2"]'``).  This function reverses that
    transformation for string values that begin with ``[``.  Non-matching
    values are returned unchanged.

    Args:
        value: Metadata value as stored in ChromaDB.

    Returns:
        The original Python list if ``value`` is a valid JSON array string,
        otherwise ``value`` unchanged.
    """
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return value
