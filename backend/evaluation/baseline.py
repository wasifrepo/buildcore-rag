"""Naive RAG baseline for comparison against the full BuildCore pipeline.

The baseline intentionally omits every advanced retrieval feature — no query
expansion, no sparse retrieval, no hybrid fusion, no cross-encoder reranking,
and no retrieval critic.  Its job is to represent what a simple
embed-retrieve-generate system produces so that the delta between the
baseline and the full pipeline is visible in the evaluation report.

Pipeline
--------
1. Embed the raw question using ``text-embedding-3-small``.
2. Query ChromaDB for the top-5 nearest child chunks (cosine similarity).
3. Resolve each hit to its parent text and concatenate the distinct parents
   into a single context block.
4. Send the question + context to GPT-4o with a plain prompt.
5. Return the response as a raw string (no structured output, no citations).

Resolving hits to parent text (rather than using the raw 2-3 sentence child
snippets) keeps the comparison fair: the delta against the full pipeline then
reflects the *pipeline's* sophistication — query expansion, hybrid retrieval,
reranking, and the critic — not merely the fact that it returns larger chunks.

The same ChromaDB collection and OpenAI models are used as the full pipeline,
controlled by the ``CHROMA_PERSIST_DIR``, ``CHROMA_COLLECTION_NAME``,
``EMBEDDING_MODEL``, and ``GENERATION_MODEL`` environment variables.
"""

import os

import chromadb
from common.llm_client import (
    embed_texts,
    get_embedding_model,
    get_generation_model,
    get_llm_client,
)

_BASELINE_TOP_K: int = 5

_SYSTEM_PROMPT = """\
You are a helpful assistant for BuildCore Operations, a construction and
facilities management company.

Answer the user's question using only the context provided below.
If the context does not contain the information needed to answer, say so
clearly — do not make up information.
"""


def run_baseline(question: str) -> str:
    """Run the naive RAG baseline for a single question.

    Embeds the question, retrieves the top-5 chunks from ChromaDB by cosine
    similarity, concatenates their text as context, and calls GPT-4o with a
    plain prompt.  Returns the model's response as a raw string.

    Args:
        question: The raw question string to answer.

    Returns:
        Plain text answer string from GPT-4o.  May be an explicit "I don't
        know" if the retrieved context is not relevant.
    """
    openai_client = get_llm_client()
    embed_model = get_embedding_model()
    gen_model = get_generation_model()

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "buildcore")

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed the question
    query_embedding = embed_texts([question], model=embed_model)[0]

    # Retrieve top-5 child chunks
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=_BASELINE_TOP_K,
        include=["documents", "metadatas"],
    )
    documents: list[str] = result["documents"][0]
    metadatas: list[dict] = result["metadatas"][0]

    # Resolve each child hit to its parent text, de-duplicating parents while
    # preserving retrieval order.  Falls back to the child text if an index
    # predates the parent-child migration and carries no parent_content.
    seen_parents: set[str] = set()
    contexts: list[str] = []
    for document, meta in zip(documents, metadatas):
        parent_id = meta.get("parent_id")
        parent_text = meta.get("parent_content") or document
        key = parent_id or document
        if key in seen_parents:
            continue
        seen_parents.add(key)
        contexts.append(parent_text)

    # Concatenate parent text into a single context block
    context = "\n\n---\n\n".join(contexts)

    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    completion = openai_client.chat.completions.create(
        model=gen_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return completion.choices[0].message.content or ""
