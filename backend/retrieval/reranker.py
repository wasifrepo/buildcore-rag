"""Cross-encoder reranker for the BuildCore RAG pipeline.

Reranks the merged hybrid retrieval results using the
``cross-encoder/ms-marco-MiniLM-L-6-v2`` model from HuggingFace
``sentence-transformers``.  A cross-encoder scores each (query, passage)
pair jointly — unlike bi-encoders it sees both inputs together — giving
it higher accuracy than cosine similarity for final-stage ranking at the
cost of being too slow to run at retrieval scale.

Lazy model loading
------------------
The model is **not** loaded at import time.  It is downloaded and
instantiated on the first call to :meth:`CrossEncoderReranker.rerank`.
This keeps FastAPI startup fast: the server is ready to accept requests
before the ~40 MB model weights are pulled from the HuggingFace Hub
(or cache).  The ``CrossEncoder`` instance is cached on the class after
the first load.

Score semantics
---------------
The ms-marco MiniLM model outputs raw logits (unbounded floats, typically
in the range ``[−10, 10]``).  Higher values indicate greater relevance.
Scores are stored as-is in ``rerank_score`` — they are used only for
ranking, not for probability calibration or thresholding, so
normalisation is not needed here.
"""

import os
from typing import ClassVar

from generation.schemas import Chunk

_DEFAULT_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_TOP_K: int = 8


class CrossEncoderReranker:
    """Reranks retrieval results with a HuggingFace cross-encoder model.

    Designed to be instantiated once at application startup.  The underlying
    ``sentence_transformers.CrossEncoder`` is loaded on first use and cached
    as a class variable so that multiple ``CrossEncoderReranker`` instances
    (or repeated calls) share a single model copy.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    """

    # Class-level model cache: shared across all instances to avoid loading
    # the cross-encoder more than once per process.
    _cached_model: ClassVar[object | None] = None
    _cached_model_name: ClassVar[str | None] = None

    def __init__(self, model_name: str | None = None) -> None:
        """Record the model name; actual loading is deferred to first use.

        Args:
            model_name: HuggingFace model identifier.  Defaults to
                ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
        """
        self._model_name = model_name or _DEFAULT_MODEL_NAME

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[Chunk]:
        """Score each (query, chunk) pair and return the top-k ranked chunks.

        Loads the cross-encoder model on first call (subsequent calls reuse
        the cached instance).  Scores all query-chunk pairs in a single
        batched ``predict`` call, attaches the raw logit score as
        ``rerank_score`` on a copy of each chunk, and returns the top-k
        chunks sorted by ``rerank_score`` descending.

        Args:
            query: The original user query string (not the expanded variants).
                The cross-encoder benefits from the unmodified query phrasing.
            chunks: Merged hybrid retrieval results to rerank, typically
                produced by :func:`~retrieval.hybrid_retriever.merge_results`.
            top_k: Number of chunks to return after reranking.  Defaults to
                the ``TOP_K_RERANKED`` environment variable (default ``8``).

        Returns:
            List of at most ``top_k`` :class:`~generation.schemas.Chunk`
            objects sorted by ``rerank_score`` descending.  Returns an empty
            list if ``chunks`` is empty.
        """
        if not chunks:
            return []

        resolved_top_k = top_k or int(
            os.environ.get("TOP_K_RERANKED", _DEFAULT_TOP_K)
        )

        model = self._load_model()

        pairs = [(query, chunk.content) for chunk in chunks]
        raw_scores: list[float] = model.predict(pairs).tolist()

        reranked: list[Chunk] = []
        for chunk, score in zip(chunks, raw_scores):
            reranked.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_type=chunk.document_type,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    dense_score=chunk.dense_score,
                    sparse_score=chunk.sparse_score,
                    rerank_score=score,
                )
            )

        reranked.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return reranked[:resolved_top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Return the cached CrossEncoder, loading it if necessary.

        Uses the class-level ``_cached_model`` and ``_cached_model_name``
        variables so that the model is loaded at most once per process, even
        if multiple ``CrossEncoderReranker`` instances exist.

        Returns:
            A ``sentence_transformers.CrossEncoder`` instance ready for
            ``predict`` calls.
        """
        if (
            CrossEncoderReranker._cached_model is not None
            and CrossEncoderReranker._cached_model_name == self._model_name
        ):
            return CrossEncoderReranker._cached_model

        # Import deferred to avoid loading torch/transformers at module import
        # time, which adds several seconds to server startup.
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        CrossEncoderReranker._cached_model = CrossEncoder(self._model_name)
        CrossEncoderReranker._cached_model_name = self._model_name
        return CrossEncoderReranker._cached_model
