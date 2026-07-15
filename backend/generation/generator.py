"""Final answer generator for the BuildCore RAG pipeline.

Takes the original user query, the reranked list of
:class:`~generation.schemas.Chunk` objects, and the
:class:`~generation.schemas.CriticVerdict` from the retrieval critic, and
returns a :class:`~generation.schemas.GeneratedAnswer` via GPT-4o structured
output.

Refusal logic
-------------
Two conditions produce a refusal (``refused=True``, empty ``citations``,
``confidence=0.0``):

1. **Out-of-scope query** — the caller signals this by passing
   ``query_type="out_of_scope"`` (the ``QueryType`` enum value from
   :class:`~generation.schemas.QueryAnalysis`).  The generator never
   attempts to answer from external knowledge.

2. **Insufficient context** — the :class:`~generation.schemas.CriticVerdict`
   carries ``sufficient=False`` after the second retrieval pass (the
   pipeline only calls the generator after the critic has had a chance to
   trigger a second pass; if it is still insufficient after that, the
   generator refuses).

Citation format
---------------
Each :class:`~generation.schemas.Citation` references:

- ``chunk_id`` — the SHA-256-derived identifier of the source chunk
- ``document_id`` — human-readable document stem (e.g. ``MAINT-FLT-03``).
  Snapped to the canonical id of a chunk actually shown to the model if the
  model's own value is a shortened prefix (see ``_snap_citation_document_id``).
- ``document_name`` — a display-friendly name derived from ``document_id``
  (underscores replaced with spaces, title-cased)
- ``excerpt`` — a short verbatim sentence or phrase from the chunk that
  directly supports the cited claim

The system prompt instructs the model to cite by ``document_id`` in the
answer text and to match each inline citation to a ``Citation`` object.

Token budget
------------
The reranker has already filtered to at most ``TOP_K_RERANKED`` chunks
(default 8).  Full chunk content is included in the user message — the
critic's truncation was only for its own evaluation, not for generation.
GPT-4o's 128k context window comfortably accommodates the BuildCore corpus
chunk sizes at this volume.
"""

import os

from common.llm_client import (
    get_generation_model,
    get_llm_client,
    reasoning_extra_body,
)

from generation.schemas import (
    Chunk,
    Citation,
    CriticVerdict,
    GeneratedAnswer,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the answer generator for BuildCore RAG — an enterprise question-
answering system for BuildCore Operations, a mid-size construction and
facilities management company.

You will receive a user query and a set of retrieved document chunks from
the BuildCore corpus.  Your job is to produce a precise, well-grounded
answer based exclusively on the provided chunks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT GROUNDING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. USE ONLY THE PROVIDED CHUNKS. Do not use any external knowledge,
   general construction industry knowledge, or information inferred beyond
   what is explicitly written in the chunks.  If the chunks do not contain
   enough information to answer the question, say so — do not fabricate.

2. CITE EVERY CLAIM. Every factual statement in your answer must be
   supported by a specific chunk.  Reference the source inline using the
   document_id (e.g. "According to SC-2024-038, …" or "The MAINT-FLT-03
   manual states …").  Vague or uncited claims are not acceptable.

3. ONE CITATION OBJECT PER CITED CHUNK. For each document_id you reference
   inline, include a corresponding Citation object in the citations list.
   The excerpt field must be a short verbatim sentence or phrase (≤ 2
   sentences) from the chunk text that directly supports the claim you made.

4. DO NOT PARAPHRASE AS FACT. If a document says "should" or "where
   applicable", preserve that nuance — do not present conditional
   requirements as absolutes.

5. ANSWER FORMAT. Write in clear, professional prose.  For procedural
   queries, use a numbered list for steps.  For factual queries, state the
   fact directly.  For cross-document queries, explicitly compare or
   synthesise across the cited sources.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFUSAL CONDITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return a refusal (refused=True, empty citations list, confidence=0.0) when:

  • The user message explicitly states the query is OUT OF SCOPE — the
    requested information is not part of the BuildCore corpus.  Explain
    clearly what the system covers.
  • The user message explicitly states the context is INSUFFICIENT — the
    retrieved chunks do not contain what is needed to answer the query.
    Do not attempt to answer from memory; acknowledge the gap.

When refusing, set:
  • refused = true
  • answer = a polite, one-sentence acknowledgement
  • refusal_reason = a clear explanation of why the system cannot answer
  • citations = [] (empty list)
  • confidence = 0.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE SCORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Set confidence (0.0–1.0) to reflect how completely the retrieved chunks
support your answer:

  0.9–1.0  All key facts are present verbatim in the chunks.
  0.7–0.9  Most facts are present; minor details may be inferred from context.
  0.5–0.7  The chunks are partially relevant; the answer addresses the query
           but may be incomplete.
  < 0.5    The chunks are only tangentially related; treat this as a soft
           refusal and note limitations explicitly in the answer.
"""

# Prefix added to the user message when a refusal is pre-determined so the
# model does not attempt to construct an answer.
_OUT_OF_SCOPE_NOTICE = (
    "[SYSTEM NOTICE] This query has been classified as OUT OF SCOPE for the "
    "BuildCore corpus.  Return a refusal with refused=True.  Do not attempt "
    "to answer from external knowledge.\n\n"
)

_INSUFFICIENT_CONTEXT_NOTICE = (
    "[SYSTEM NOTICE] The retrieval critic has flagged the retrieved context "
    "as INSUFFICIENT to answer this query confidently (critic reasoning: "
    "{reasoning}).  Return a refusal with refused=True.  Do not attempt to "
    "answer from the partial evidence below.\n\n"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_answer(
    query: str,
    chunks: list[Chunk],
    critic_verdict: CriticVerdict,
    query_type: str = "factual",
) -> GeneratedAnswer:
    """Generate a grounded, cited answer from the reranked retrieval results.

    Builds a structured user message containing the query, any pre-determined
    refusal notice, and the full content of each retrieved chunk, then calls
    GPT-4o with structured output mode to produce a
    :class:`~generation.schemas.GeneratedAnswer` directly.

    Args:
        query: The original raw user query string.
        chunks: Reranked list of :class:`~generation.schemas.Chunk` objects
            from the cross-encoder reranker.  Full content is included in the
            prompt — no truncation at this stage.
        critic_verdict: The :class:`~generation.schemas.CriticVerdict` from
            the retrieval critic.  If ``sufficient=False``, a refusal notice
            is prepended to the user message so the model does not hallucinate
            an answer from weak evidence.
        query_type: The ``QueryType`` string value from
            :class:`~generation.schemas.QueryAnalysis` (e.g. ``"factual"``,
            ``"out_of_scope"``).  An ``"out_of_scope"`` value triggers an
            unconditional refusal notice before the model is called.

    Returns:
        A fully populated :class:`~generation.schemas.GeneratedAnswer` with
        ``answer``, ``citations``, ``confidence``, ``refused``, and
        optionally ``refusal_reason`` fields.

    Raises:
        openai.OpenAIError: On any API-level failure.  The caller is
            responsible for retry logic.
    """
    client = get_llm_client()
    model = get_generation_model()

    user_message = _build_user_message(query, chunks, critic_verdict, query_type)

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=GeneratedAnswer,
        extra_body=reasoning_extra_body("generation"),
    )

    answer: GeneratedAnswer = completion.choices[0].message.parsed

    # Snap each citation's document_id to a real chunk's document_id before
    # deriving the display name. The system prompt's own example ("According
    # to SC-2024-038, ...") uses a shortened form that doesn't exactly match
    # the canonical document_id (e.g. "SC-2024-038-apex-plumbing"), so the
    # model sometimes cites correctly but with the wrong exact string —
    # which downstream exact-match scoring (see evaluation/metrics.py) would
    # otherwise count as an uncited claim.
    valid_document_ids = [c.document_id for c in chunks]
    answer.citations = [
        _normalise_citation(_snap_citation_document_id(c, valid_document_ids))
        for c in answer.citations
    ]

    return answer


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_user_message(
    query: str,
    chunks: list[Chunk],
    critic_verdict: CriticVerdict,
    query_type: str,
) -> str:
    """Compose the full user message sent to the generation model.

    Prepends a refusal notice when the query is out of scope or the critic
    has flagged insufficient context.  Appends every chunk in full so the
    model has complete grounding material.

    Args:
        query: The original user query string.
        chunks: Reranked retrieval results to include verbatim.
        critic_verdict: Critic's sufficiency assessment.
        query_type: QueryType string value; ``"out_of_scope"`` triggers an
            unconditional refusal notice.

    Returns:
        Multi-line string forming the complete user turn of the chat prompt.
    """
    lines: list[str] = []

    # --- Refusal notice (prepended before the query) ---
    if query_type == "out_of_scope":
        lines.append(_OUT_OF_SCOPE_NOTICE)
    elif not critic_verdict.sufficient:
        notice = _INSUFFICIENT_CONTEXT_NOTICE.format(
            reasoning=critic_verdict.reasoning
        )
        lines.append(notice)

    # --- Query ---
    lines += [
        f"USER QUERY: {query}",
        "",
    ]

    # --- Retrieved chunks (full content, no truncation) ---
    if chunks:
        lines += [
            f"RETRIEVED CONTEXT ({len(chunks)} chunks):",
            "━" * 56,
            "",
        ]
        for rank, chunk in enumerate(chunks, start=1):
            rerank_info = (
                f"  rerank_score={chunk.rerank_score:.4f}"
                if chunk.rerank_score is not None
                else ""
            )
            lines += [
                f"[{rank}] chunk_id={chunk.chunk_id}",
                f"    document_id={chunk.document_id}  "
                f"type={chunk.document_type}{rerank_info}",
                "",
                chunk.content,
                "",
                "─" * 40,
                "",
            ]
    else:
        lines += [
            "RETRIEVED CONTEXT: (none — no chunks were retrieved)",
            "",
        ]

    lines.append(
        "Answer the user query using only the retrieved context above. "
        "Cite every claim by document_id."
    )

    return "\n".join(lines)


def _snap_citation_document_id(
    citation: Citation, valid_document_ids: list[str]
) -> Citation:
    """Correct a citation's document_id to match a chunk actually shown to the model.

    If ``citation.document_id`` is not an exact match to any of
    ``valid_document_ids``, looks for a valid id that the citation's value is
    a prefix of (or vice versa) — e.g. the model wrote ``"SC-2024-038"`` but
    the canonical id is ``"SC-2024-038-apex-plumbing"`` — and substitutes the
    canonical id. Leaves the citation unchanged if no match is found.

    Args:
        citation: A Citation as produced by the structured output parser.
        valid_document_ids: ``document_id`` values of the chunks that were
            actually included in the generation prompt.

    Returns:
        A Citation with a corrected ``document_id``, or the original citation
        if it already matches exactly or no plausible match exists.
    """
    if citation.document_id in valid_document_ids:
        return citation

    for real_id in valid_document_ids:
        if real_id.startswith(citation.document_id) or citation.document_id.startswith(real_id):
            return Citation(
                chunk_id=citation.chunk_id,
                document_id=real_id,
                document_name=citation.document_name,
                excerpt=citation.excerpt,
            )

    return citation


def _normalise_citation(citation: Citation) -> Citation:
    """Ensure every Citation has a populated ``document_name`` field.

    Derives a display-friendly name from ``document_id`` by replacing
    hyphens and underscores with spaces and applying title casing.  This
    is applied as a post-processing step so the model's ``document_name``
    output (which may be empty or inconsistent) is always overridden with
    a deterministic value.

    Args:
        citation: A :class:`~generation.schemas.Citation` as produced by
            the structured output parser.

    Returns:
        A new :class:`~generation.schemas.Citation` with ``document_name``
        set to the normalised display name.
    """
    display_name = (
        citation.document_id
        .replace("-", " ")
        .replace("_", " ")
        .title()
    )
    return Citation(
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        document_name=display_name,
        excerpt=citation.excerpt,
    )
