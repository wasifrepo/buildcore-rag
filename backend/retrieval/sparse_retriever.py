"""Sparse (BM25) retriever for the BuildCore RAG pipeline.

Implements keyword-based retrieval using the ``rank-bm25`` library.  At
initialisation, the retriever loads all chunk documents from the same
ChromaDB collection used by the dense retriever, tokenises their content,
and builds a ``BM25Okapi`` index.  This ensures a single source of truth:
the ChromaDB collection is the authoritative store and BM25 operates over
exactly the same corpus.

Tokenisation
-------------
Content is lower-cased and split on whitespace.  Punctuation is stripped
from token boundaries using a lightweight regex so that terms like
``"lockout/tagout"``, ``"MAINT-FLT-03"``, and ``"P/F/N/A"`` are tokenised
consistently without external NLP dependencies.

Score normalisation
--------------------
Raw BM25 scores are non-negative floats with no fixed upper bound.  To make
``sparse_score`` comparable in magnitude to ``dense_score`` (cosine
similarity in ``[0, 1]``), scores are normalised by dividing by the maximum
score in each result set.  A query that matches no documents returns an empty
list.

Index refresh
--------------
Call :meth:`SparseRetriever.rebuild_index` after re-ingestion to reload the
corpus from ChromaDB and rebuild the BM25 index in place.  The retriever
object can continue to serve requests before and after a rebuild without
restart.
"""

import os
import re

import chromadb
from rank_bm25 import BM25Okapi

from generation.schemas import Chunk
from retrieval._parenting import parent_from_child

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K: int = 20
# Regex that matches runs of non-alphanumeric characters at token boundaries.
_NON_ALNUM: re.Pattern[str] = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class SparseRetriever:
    """BM25 keyword retriever backed by the ChromaDB corpus.

    Loads all chunk documents from ChromaDB on construction, builds a BM25
    index, and exposes a :meth:`search` method for keyword retrieval.  The
    index is held in memory; the class is designed to be instantiated once
    at application startup.

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
        """Load the corpus from ChromaDB and build the BM25 index.

        Args:
            chroma_persist_dir: Path to the ChromaDB persistence directory.
                Falls back to ``CHROMA_PERSIST_DIR``, then ``./data/chroma``.
            chroma_collection_name: Name of the ChromaDB collection.
                Falls back to ``CHROMA_COLLECTION_NAME``, then ``"buildcore"``.
        """
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

        # Internal state populated by _build_index
        self._chunk_ids: list[str] = []
        self._chunk_documents: list[str] = []
        self._chunk_metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None

        self._build_index()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Search the BM25 index for chunks matching the query keywords.

        Tokenises the query using the same normalisation applied during index
        construction, scores all documents, and returns the top-k results
        sorted by normalised BM25 score descending.

        Args:
            query: Raw query string.  Expanded variants are typically not
                passed to BM25 — the original query is preferred for sparse
                retrieval to avoid over-counting keyword matches.
            top_k: Maximum number of results to return.  Defaults to the
                ``TOP_K_SPARSE`` environment variable (default ``20``).

        Returns:
            List of parent :class:`~generation.schemas.Chunk` objects with
            ``sparse_score`` populated (normalised to ``[0, 1]``) and sorted by
            ``sparse_score`` descending, at most ``resolved_top_k`` long.  The
            BM25 index scores *child* chunks; each child hit is collapsed to its
            parent, keeping the parent's best-scoring child.  Returns an empty
            list if the BM25 index is empty or no documents score above zero.
        """
        if self._bm25 is None or not self._chunk_ids:
            return []

        resolved_top_k = top_k or int(os.environ.get("TOP_K_SPARSE", _DEFAULT_TOP_K))
        tokens = _tokenise(query)
        scores: list[float] = self._bm25.get_scores(tokens).tolist()

        max_score = max(scores) if scores else 0.0
        if max_score == 0.0:
            return []

        # Collect (index, normalised_score) pairs for non-zero scorers
        scored = [
            (i, score / max_score)
            for i, score in enumerate(scores)
            if score > 0.0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Walk child hits in descending score order, collapsing to parents.
        # The first time a parent is seen carries its best child score, so we
        # keep that and stop once we have resolved_top_k distinct parents.
        best: dict[str, Chunk] = {}
        for idx, normalised_score in scored:
            parent = parent_from_child(
                self._chunk_ids[idx],
                self._chunk_documents[idx],
                self._chunk_metadatas[idx],
                normalised_score,
                "sparse",
            )
            if parent.chunk_id in best:
                continue
            best[parent.chunk_id] = parent
            if len(best) >= resolved_top_k:
                break

        return list(best.values())

    def rebuild_index(self) -> None:
        """Reload all documents from ChromaDB and rebuild the BM25 index.

        Call this after re-ingestion to synchronise the sparse index with
        the updated corpus.  The method is synchronous and blocks until the
        rebuild is complete; it is safe to call while the server is running.
        """
        self._build_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Fetch all documents from ChromaDB and construct the BM25 index.

        Retrieves the full corpus using ChromaDB's ``get`` method (no
        embedding required), tokenises each document's content, and passes
        the tokenised corpus to ``BM25Okapi``.

        If the collection is empty (e.g. before the first ingestion run),
        the index is set to ``None`` and :meth:`search` will return an empty
        list until :meth:`rebuild_index` is called.
        """
        count = self._collection.count()
        if count == 0:
            self._chunk_ids = []
            self._chunk_documents = []
            self._chunk_metadatas = []
            self._bm25 = None
            return

        # ChromaDB's get() without IDs or filters returns all records.
        # Fetch in one call; for very large corpora this could be batched,
        # but the BuildCore corpus fits comfortably in memory.
        all_records = self._collection.get(include=["documents", "metadatas"])

        self._chunk_ids = all_records["ids"]
        self._chunk_documents = all_records["documents"]
        self._chunk_metadatas = all_records["metadatas"]

        tokenised_corpus = [_tokenise(doc) for doc in self._chunk_documents]
        self._bm25 = BM25Okapi(tokenised_corpus)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """Tokenise text for BM25 indexing and querying.

    Lowercases the input and splits on runs of non-alphanumeric characters,
    discarding empty tokens.  This keeps hyphenated identifiers like
    ``"MAINT-FLT-03"`` as two tokens (``"maint"`` and ``"flt"`` and ``"03"``)
    and handles slash-separated values (``"p/f/n/a"`` → ``["p", "f", "n", "a"]``).

    Args:
        text: Raw document content or query string.

    Returns:
        List of lowercase alphanumeric token strings.
    """
    return [t for t in _NON_ALNUM.split(text.lower()) if t]
