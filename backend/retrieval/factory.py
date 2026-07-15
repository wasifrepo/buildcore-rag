"""Retriever backend selection.

Chooses the :class:`~retrieval.base.Retriever` implementation from the
``RETRIEVER_BACKEND`` environment variable, so the same pipeline code runs
against the local ChromaDB + BM25 stack in development and Azure AI Search in
production without any call-site changes.

Recognised values (case-insensitive):

* ``local`` / ``chroma`` (default) → :class:`~retrieval.local_retriever.LocalRetriever`
* ``azure`` / ``azure_ai_search`` → :class:`~retrieval.azure_ai_search_retriever.AzureAISearchRetriever`
"""

import os

from retrieval.azure_ai_search_retriever import AzureAISearchRetriever
from retrieval.base import Retriever
from retrieval.local_retriever import LocalRetriever

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
        return LocalRetriever()
    if backend in _AZURE_ALIASES:
        return AzureAISearchRetriever()
    raise ValueError(
        f"Unknown RETRIEVER_BACKEND '{backend}'. "
        f"Expected one of: {sorted(_LOCAL_ALIASES | _AZURE_ALIASES)}."
    )
