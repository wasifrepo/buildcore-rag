"""Retriever backend selection.

Chooses the :class:`~retrieval.base.Retriever` implementation from the
``RETRIEVER_BACKEND`` environment variable, so the same pipeline code runs
against the local ChromaDB + BM25 stack in development and Azure AI Search in
production without any call-site changes.

Recognised values (case-insensitive):

* ``local`` / ``chroma`` (default) → :class:`~retrieval.local_retriever.LocalRetriever`
* ``azure`` / ``azure_ai_search`` → :class:`~retrieval.azure_ai_search_retriever.AzureAISearchRetriever`

Backend modules are imported lazily, inside :func:`get_retriever`, so that
selecting one backend never imports the other's dependencies.  This is what
allows the Azure production image to omit ``chromadb`` and ``rank-bm25``
entirely: nothing imports ``local_retriever`` unless the local backend is
actually selected.
"""

import os

from retrieval.base import Retriever

_LOCAL_ALIASES: frozenset[str] = frozenset({"local", "chroma"})
_AZURE_ALIASES: frozenset[str] = frozenset({"azure", "azure_ai_search", "azure-ai-search"})


def get_retriever() -> Retriever:
    """Instantiate the retriever backend named by ``RETRIEVER_BACKEND``.

    Returns:
        A ready-to-use :class:`~retrieval.base.Retriever`.  Defaults to the
        local backend when the variable is unset.

    Raises:
        ValueError: If ``RETRIEVER_BACKEND`` is set to an unrecognised value.
    """
    backend = os.environ.get("RETRIEVER_BACKEND", "local").strip().lower()
    if backend in _LOCAL_ALIASES:
        from retrieval.local_retriever import LocalRetriever  # noqa: PLC0415

        return LocalRetriever()
    if backend in _AZURE_ALIASES:
        from retrieval.azure_ai_search_retriever import (  # noqa: PLC0415
            AzureAISearchRetriever,
        )

        return AzureAISearchRetriever()
    raise ValueError(
        f"Unknown RETRIEVER_BACKEND '{backend}'. "
        f"Expected one of: {sorted(_LOCAL_ALIASES | _AZURE_ALIASES)}."
    )
