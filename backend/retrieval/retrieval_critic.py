"""Retrieval critic — LLM-as-judge step in the BuildCore RAG pipeline.

After the reranker produces a final set of chunks, the retrieval critic
evaluates whether those chunks contain enough grounded information to answer
the original user query confidently.  If not, it produces a
:class:`~generation.schemas.CriticVerdict` with ``sufficient=False`` and a
``refined_query`` that the pipeline can use for a second retrieval pass.

This step acts as a safety layer between retrieval and generation: it would
rather surface an honest "insufficient context" signal than allow the
generator to hallucinate a plausible-sounding answer from weak evidence.

Design principles for the system prompt
-----------------------------------------
* **Strict by default** — the critic is instructed to flag insufficiency
  whenever key facts needed to answer the query are absent, even if the
  chunks are tangentially related.
* **Corpus-aware** — the prompt tells the model what documents exist so it
  can distinguish "the answer isn't in these chunks but might be in another
  chunk" (→ refined_query) from "the answer isn't in BuildCore's corpus at
  all" (→ sufficient=False, no refined_query).
* **Refined query quality** — when the critic requests a second pass, the
  refined query must be concrete and retrieval-friendly, not a rephrasing of
  the same vague question.

The critic uses ``gpt-4o-mini`` (the ``ANALYSIS_MODEL`` env var) with
structured output so the response is parsed directly into a
:class:`~generation.schemas.CriticVerdict` with no post-processing.
"""

import os

from openai import OpenAI

from generation.schemas import Chunk, CriticVerdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of characters from each chunk included in the user message.
# This caps token usage while still giving the critic enough text to judge.
_MAX_CHUNK_CHARS: int = 2000

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the retrieval critic for BuildCore RAG — an enterprise question-
answering system over construction and facilities management documents.

Your role is to read a user query and the chunks that were retrieved for it,
then decide whether those chunks contain sufficient grounded information to
answer the query accurately and completely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUILDCORE CORPUS — what documents exist
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Safety SOPs: chemical spill response, working at height, hot-works permit,
   confined space entry, electrical isolation (lockout / tagout).
2. Subcontractor contracts: Apex Electrical (SC-2024-038) and Harrington
   Scaffolding (SC-2024-041) — scope, price, and general conditions.
3. Incident emails: INC-2024-007 forklift near-miss at Commerce Drive
   (5 Feb 2024), INC-2024-002 laceration LTI — Sam Osei, Zone 3 Warehouse
   (14 Feb 2024), INC-2024-009 epoxy chemical spill, Zone 3 (22 Feb 2024).
4. Maintenance manuals: Toyota 8FGF25 forklift (MAINT-FLT-03) and
   Denyo DCA-45SPK3 generator (MAINT-GEN-01).
5. Compliance checklists: SSIC-001 daily site safety inspection and
   SC-PMCL-001 subcontractor pre-mobilisation checklist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION CRITERIA — be strict
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mark sufficient=True ONLY when:
  • The retrieved chunks contain enough information to construct a complete
    and accurate answer, even if the exact phrasing differs from what the
    user expects.
  • The answer can be grounded in the chunk text without inference or
    extrapolation beyond what is written.
  • Maintenance manuals contain operational and emergency procedures that
    are equally valid sources as SOPs. Do not mark insufficient just because
    the source is a manual rather than an SOP.

Mark sufficient=False when:
  • Key facts needed to answer the query are absent from the chunks.
  • The chunks are only tangentially related to the query.
  • The chunks answer a different question than the one asked.
  • The query requires cross-document synthesis but the chunks cover only
    one side.

When sufficient=False:
  • If the missing information is likely present elsewhere in the BuildCore
    corpus (based on the document inventory above), set refined_query to a
    concrete, retrieval-friendly query that targets the missing information.
    Example: instead of "tell me about the contract", write
    "Apex Electrical SC-2024-038 payment terms and variation clause".
  • If the query is genuinely out of scope for the BuildCore corpus, set
    refined_query to null and explain in reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FIELDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sufficient   — bool. True only if the chunks are genuinely sufficient.
confidence   — float [0, 1]. Your confidence in this verdict. Use < 0.6 when
               the chunks are partially relevant but not conclusive.
reasoning    — One to three sentences explaining your verdict. Reference
               specific missing information when sufficient=False.
refined_query — Non-null only when sufficient=False AND the missing info is
               plausibly in the corpus. Must be a concrete retrieval query.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_retrieval(query: str, chunks: list[Chunk]) -> CriticVerdict:
    """Evaluate whether the retrieved chunks are sufficient to answer the query.

    Formats the query and a truncated preview of each chunk into a structured
    user message, then calls the OpenAI Chat Completions API with structured
    output mode to obtain a :class:`~generation.schemas.CriticVerdict`.

    Args:
        query: The original raw user query string.
        chunks: Reranked list of :class:`~generation.schemas.Chunk` objects
            produced by the cross-encoder reranker.  The critic reads the
            actual chunk text, so passing an empty list will reliably produce
            ``sufficient=False``.

    Returns:
        A :class:`~generation.schemas.CriticVerdict` with ``sufficient``,
        ``confidence``, ``reasoning``, and optionally ``refined_query``
        populated by the LLM.

    Raises:
        openai.OpenAIError: On any API-level failure.  The caller is
            responsible for retry logic.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("ANALYSIS_MODEL", "gpt-4o-mini")

    user_message = _format_user_message(query, chunks)

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=CriticVerdict,
    )

    return completion.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_user_message(query: str, chunks: list[Chunk]) -> str:
    """Build the user message that presents the query and retrieved chunks.

    Each chunk is formatted with its rank, document ID, document type, and
    a truncated excerpt of its content.  Truncation is applied per chunk at
    ``_MAX_CHUNK_CHARS`` characters to keep the message within a reasonable
    token budget while preserving enough context for the critic to judge.

    Args:
        query: The original user query string.
        chunks: Retrieved chunks to present to the critic.

    Returns:
        Formatted multi-line string ready to send as the user turn.
    """
    lines: list[str] = [
        f"USER QUERY: {query}",
        "",
        f"RETRIEVED CHUNKS ({len(chunks)} total):",
        "━" * 56,
    ]

    if not chunks:
        lines.append("(no chunks were retrieved)")
    else:
        for rank, chunk in enumerate(chunks, start=1):
            excerpt = chunk.content[:_MAX_CHUNK_CHARS]
            if len(chunk.content) > _MAX_CHUNK_CHARS:
                excerpt += "… [truncated]"
            lines += [
                f"[{rank}] document_id={chunk.document_id}  "
                f"type={chunk.document_type}  "
                f"rerank_score={chunk.rerank_score:.4f}"
                if chunk.rerank_score is not None
                else f"[{rank}] document_id={chunk.document_id}  "
                     f"type={chunk.document_type}",
                excerpt,
                "─" * 40,
            ]

    lines += [
        "",
        "Evaluate whether these chunks are sufficient to answer the query.",
        "Be strict: only mark sufficient=True if the answer is grounded in "
        "the chunk text above.",
    ]
    return "\n".join(lines)
