"""Naive RAG baseline for comparison against the full BuildCore pipeline.

The baseline intentionally omits every advanced retrieval feature — no query
expansion, no sparse retrieval, no hybrid fusion, no cross-encoder reranking,
and no retrieval critic.  Its job is to represent what a simple
embed-retrieve-generate system produces so that the delta between the
baseline and the full pipeline is visible in the evaluation report.

Pipeline
--------
1. Embed the raw question using ``text-embedding-3-small``.
2. Query ChromaDB for the top-5 nearest chunks (cosine similarity).
3. Concatenate the raw chunk text into a single context block.
4. Send the question + context to GPT-4o with a plain prompt.
5. Return the response as a raw string (no structured output, no citations).

The same ChromaDB collection and OpenAI models are used as the full pipeline,
controlled by the ``CHROMA_PERSIST_DIR``, ``CHROMA_COLLECTION_NAME``,
``EMBEDDING_MODEL``, and ``GENERATION_MODEL`` environment variables.
"""

import os

import chromadb
from openai import OpenAI

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
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embed_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    gen_model = os.environ.get("GENERATION_MODEL", "gpt-4o")

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "buildcore")

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed the question
    embed_response = openai_client.embeddings.create(
        model=embed_model,
        input=[question],
    )
    query_embedding = embed_response.data[0].embedding

    # Retrieve top-5 chunks
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=_BASELINE_TOP_K,
        include=["documents"],
    )
    chunks: list[str] = result["documents"][0]

    # Concatenate chunk text into a single context block
    context = "\n\n---\n\n".join(chunks)

    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    completion = openai_client.chat.completions.create(
        model=gen_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return completion.choices[0].message.content or ""
