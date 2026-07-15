"""Reranker backend selection.

Chooses the :class:`~retrieval.base.Reranker` implementation from the
``RERANKER_BACKEND`` environment variable, so the pipeline's rerank step can be
served either by the local cross-encoder or by a backend that has already
ranked its own results.

Recognised values (case-insensitive):

* ``cross_encoder`` / ``local`` (default) →
  :class:`~retrieval.reranker.CrossEncoderReranker`
* ``passthrough`` / ``none`` / ``semantic`` →
  :class:`PassthroughReranker`

Why a pass-through rather than removing the step
------------------------------------------------
When Azure AI Search's managed semantic ranker is enabled, results arrive
already reranked (see ``retrieval.azure_ai_search_retriever``).  Running the
MiniLM cross-encoder over them again would be redundant work that discards the
managed ranker's ordering.  Rather than branch the orchestration in
``query.py``, the rerank *step* stays in the pipeline and its *implementation*
becomes a no-op that preserves the incoming order.  The SSE stream, the
``PipelineTrace``, and the UI keep their shape, and the layer sequence stays
true to the documented architecture.

Keeping torch out of the production image
-----------------------------------------
``retrieval.reranker`` imports ``sentence_transformers`` lazily, inside the
method that loads the model, so importing this module never pulls in torch.
Selecting ``passthrough`` means the import never happens at all — which is what
lets the Azure image ship without torch (~3 GB → ~300 MB) and keeps Container
Apps cold starts fast under scale-to-zero.
"""

import os

from generation.schemas import Chunk

_CROSS_ENCODER_ALIASES: frozenset[str] = frozenset(
    {"cross_encoder", "cross-encoder", "local", "minilm"}
)
_PASSTHROUGH_ALIASES: frozenset[str] = frozenset(
    {"passthrough", "none", "semantic", "azure"}
)

_DEFAULT_TOP_K: int = 8


class PassthroughReranker:
    """No-op reranker for backends whose results are already ranked.

    Satisfies the :class:`~retrieval.base.Reranker` protocol structurally.
    Preserves the order it is given and only applies the top-k cut, leaving
    each chunk's existing ``rerank_score`` (populated upstream by the Azure
    semantic ranker) untouched.
    """

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[Chunk]:
        """Return the incoming chunks unchanged, truncated to ``top_k``.

        Args:
            query: Unused; accepted to satisfy the reranker protocol.
            chunks: Chunks in the order the retriever produced them.
            top_k: Maximum chunks to return.  Falls back to ``TOP_K_RERANKED``.

        Returns:
            The first ``top_k`` chunks, in their original order.
        """
        resolved_top_k = top_k or int(
            os.environ.get("TOP_K_RERANKED", _DEFAULT_TOP_K)
        )
        return chunks[:resolved_top_k]


def get_reranker():
    """Instantiate the reranker backend named by ``RERANKER_BACKEND``.

    Returns:
        An object satisfying the :class:`~retrieval.base.Reranker` protocol.
        Defaults to the cross-encoder when the variable is unset.

    Raises:
        ValueError: If ``RERANKER_BACKEND`` is set to an unrecognised value.
    """
    backend = os.environ.get("RERANKER_BACKEND", "cross_encoder").strip().lower()
    if backend in _CROSS_ENCODER_ALIASES:
        # Imported here so that selecting passthrough never imports the
        # cross-encoder module, and torch stays out of the image.
        from retrieval.reranker import CrossEncoderReranker  # noqa: PLC0415

        return CrossEncoderReranker()
    if backend in _PASSTHROUGH_ALIASES:
        return PassthroughReranker()
    raise ValueError(
        f"Unknown RERANKER_BACKEND '{backend}'. "
        f"Expected one of: {sorted(_CROSS_ENCODER_ALIASES | _PASSTHROUGH_ALIASES)}."
    )
