"""Naive RAG baseline for comparison against the full BuildCore pipeline.

The baseline intentionally omits every advanced retrieval feature — no query
expansion, no sparse retrieval, no hybrid fusion, no reranking (cross-encoder
or semantic), and no retrieval critic.  Its job is to represent what a simple
embed-retrieve-generate system produces so that the delta between the baseline
and the full pipeline is visible in the evaluation report.

Pipeline
--------
1. Embed the raw question using the configured embedding model.
2. Pure vector search for the top-5 nearest child chunks (cosine similarity).
3. Resolve each hit to its parent text and concatenate the distinct parents
   into a single context block.
4. Send the question + context to the generation model with a plain prompt.
5. Return the response as a raw string (no structured output, no citations).

Holding variables constant
--------------------------
The baseline exists to isolate *one* variable: the pipeline.  Everything else
is deliberately held identical to the full system.

* **Same retrieval substrate.**  The baseline follows ``RETRIEVER_BACKEND``, so
  when the system runs on Azure AI Search the baseline queries the same index.
  A baseline pinned to a different store would fold "Azure vs ChromaDB" and
  "different corpus snapshot" into a number that is supposed to measure query
  expansion, hybrid fusion, reranking, and the critic.
* **Same chunks.**  Hits resolve to parent text rather than the raw 2-3
  sentence child snippets, so the delta is not merely an artefact of the
  baseline receiving smaller context.
* **Same models.**  Embedding, generation, and reasoning effort all resolve
  through ``common.llm_client``.

What is deliberately *not* held constant is exactly the list in the first
paragraph — those differences are the thing being measured.

Naivety of the vector query
---------------------------
On Azure the query passes ``search_text=None`` and omits ``query_type`` so that
neither BM25 nor the managed semantic ranker contributes: it is a bare
nearest-neighbour lookup, matching what the ChromaDB path does.  This is what
makes it a fair stand-in for "what most people build first".
"""

import os

from common.llm_client import (
    embed_texts,
    get_embedding_model,
    get_generation_model,
    get_llm_client,
    reasoning_extra_body,
)

_BASELINE_TOP_K: int = 5

_AZURE_ALIASES: frozenset[str] = frozenset(
    {"azure", "azure_ai_search", "azure-ai-search"}
)

_SYSTEM_PROMPT = """\
You are a helpful assistant for BuildCore Operations, a construction and
facilities management company.

Answer the user's question using only the context provided below.
If the context does not contain the information needed to answer, say so
clearly — do not make up information.
"""


def run_baseline(question: str) -> str:
    """Run the naive RAG baseline for a single question.

    Embeds the question, retrieves the top-5 nearest child chunks from the
    active retrieval backend by cosine similarity alone, resolves them to their
    parent text, and calls the generation model with a plain prompt.

    Args:
        question: The raw question string to answer.

    Returns:
        Plain text answer string.  May be an explicit "I don't know" if the
        retrieved context is not relevant.
    """
    query_embedding = embed_texts([question], model=get_embedding_model())[0]

    if _is_azure_backend():
        contexts = _search_azure(query_embedding, _BASELINE_TOP_K)
    else:
        contexts = _search_chroma(query_embedding, _BASELINE_TOP_K)

    context = "\n\n---\n\n".join(contexts)
    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    completion = get_llm_client().chat.completions.create(
        model=get_generation_model(),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        extra_body=reasoning_extra_body("generation"),
    )
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_azure_backend() -> bool:
    """Report whether the Azure AI Search backend is selected.

    Returns:
        ``True`` when ``RETRIEVER_BACKEND`` names the Azure backend.  Any other
        value (including unset) selects the local ChromaDB path, matching
        ``retrieval.factory``'s default.
    """
    return os.environ.get("RETRIEVER_BACKEND", "local").strip().lower() in _AZURE_ALIASES


def _collapse_to_parent_texts(
    hits: list[tuple[str, str | None, str | None]],
) -> list[str]:
    """De-duplicate child hits by parent, preserving retrieval order.

    Args:
        hits: Ordered ``(child_text, parent_id, parent_content)`` triples.
            ``parent_id`` and ``parent_content`` may be ``None`` for an index
            predating the parent-child migration.

    Returns:
        Distinct parent texts in retrieval order, falling back to the child's
        own text when no parent linkage is present.
    """
    seen: set[str] = set()
    contexts: list[str] = []
    for child_text, parent_id, parent_content in hits:
        key = parent_id or child_text
        if key in seen:
            continue
        seen.add(key)
        contexts.append(parent_content or child_text)
    return contexts


def _search_azure(embedding: list[float], top_k: int) -> list[str]:
    """Pure vector search against Azure AI Search, collapsed to parent texts.

    Passes ``search_text=None`` and no ``query_type``, so neither BM25 nor the
    semantic ranker participates — only HNSW nearest-neighbour search.

    Args:
        embedding: The question's embedding vector.
        top_k: Number of child chunks to retrieve before parent de-duplication.

    Returns:
        Distinct parent texts in retrieval order.
    """
    # Imported here so the module does not require azure-search-documents when
    # running the local backend.
    from azure.search.documents.models import VectorizedQuery  # noqa: PLC0415

    from ingestion.azure_index import (  # noqa: PLC0415
        get_parent_id_field,
        get_vector_field,
        search_with_retry,
    )

    parent_id_field = get_parent_id_field()
    results = search_with_retry(
        search_text=None,
        vector_queries=[
            VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=top_k,
                fields=get_vector_field(),
            )
        ],
        top=top_k,
        select=["content", "parent_content", parent_id_field],
    )
    hits = [
        (
            row.get("content", ""),
            row.get(parent_id_field),
            row.get("parent_content"),
        )
        for row in results
    ]
    return _collapse_to_parent_texts(hits)


def _search_chroma(embedding: list[float], top_k: int) -> list[str]:
    """Pure vector search against the local ChromaDB collection.

    Args:
        embedding: The question's embedding vector.
        top_k: Number of child chunks to retrieve before parent de-duplication.

    Returns:
        Distinct parent texts in retrieval order.
    """
    # Imported inside the function, not at module scope. api/main.py reaches
    # this module transitively (routes.evaluate -> evaluator -> baseline), so a
    # module-level import would make chromadb a hard dependency of simply
    # starting the API — and the Azure production image deliberately omits it.
    # The container would then die on startup with ModuleNotFoundError before
    # uvicorn ever binds a port.
    import chromadb  # noqa: PLC0415

    chroma_client = chromadb.PersistentClient(
        path=os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
    )
    collection = chroma_client.get_or_create_collection(
        name=os.environ.get("CHROMA_COLLECTION_NAME", "buildcore"),
        metadata={"hnsw:space": "cosine"},
    )
    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    hits = [
        (document, meta.get("parent_id"), meta.get("parent_content"))
        for document, meta in zip(result["documents"][0], result["metadatas"][0])
    ]
    return _collapse_to_parent_texts(hits)
