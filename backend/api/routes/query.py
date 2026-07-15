"""FastAPI SSE streaming endpoint for the BuildCore RAG query pipeline.

Receives a :class:`QueryRequest` (a single ``question`` string), orchestrates
the full six-step retrieval-generation pipeline, and streams each completed
step to the client as a Server-Sent Event.

SSE event format
----------------
Every event is emitted as::

    data: {"step": "<event_name>", "payload": { ... }}\n\n

The ``payload`` mirrors the Pydantic model for that step.  The client should
parse the outer ``step`` field to route payloads to the correct UI component.

Pipeline order and SSE event names
------------------------------------
1. ``query_analyzed``       — :class:`~generation.schemas.QueryAnalysis`
2. ``queries_expanded``     — :class:`~generation.schemas.ExpandedQueries`
3. ``chunks_retrieved``     — list of chunk summaries (id, document_id, type,
                             dense_score, sparse_score) after RRF merge
4. ``chunks_reranked``      — list of chunk summaries after cross-encoder
                             reranking (adds rerank_score)
5. ``critic_verdict``       — :class:`~generation.schemas.CriticVerdict`
                             from the first retrieval pass
6. ``second_pass_triggered``— (conditional) emitted when the critic returns
                             ``sufficient=False``; payload contains the
                             ``refined_query`` used for the second pass
7. ``answer_generated``     — :class:`~generation.schemas.GeneratedAnswer`

If ``second_pass_triggered`` fires, a full dense + sparse + hybrid + rerank
cycle is performed with the critic's ``refined_query``.  The critic is
re-run on the second-pass chunks to produce the verdict passed to the
generator; the ``critic_verdict`` SSE event always shows the first-pass
verdict so the frontend can display "second pass triggered" context.

Singletons
----------
The :class:`~retrieval.base.Retriever` backend (chosen by ``RETRIEVER_BACKEND``
via :func:`~retrieval.factory.get_retriever`) and the
:class:`~retrieval.reranker.CrossEncoderReranker` are instantiated once at
module load time and reused across requests.  The cross-encoder model is
loaded lazily on first use (see ``reranker.py``), so the first request after
server startup will be slower than subsequent ones.

Trace persistence
-----------------
After every successful pipeline run a :class:`~generation.schemas.PipelineTrace`
JSON file is written to the ``TRACES_DIR`` directory (env var, default
``./traces``).  The filename is ``{trace_id}.json`` where ``trace_id`` is a
UUID4.
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generation.schemas import (
    Chunk,
    CriticVerdict,
    ExpandedQueries,
    PipelineTrace,
    QueryAnalysis,
)
from generation.generator import generate_answer
from retrieval.factory import get_retriever
from retrieval.query_analyzer import analyze_query
from retrieval.query_expander import expand_query
from retrieval.reranker import CrossEncoderReranker
from retrieval.retrieval_critic import assess_retrieval

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

# Instantiated once so ChromaDB connections and the BM25 index are built only
# at startup.  The retriever backend (local ChromaDB+BM25 or Azure AI Search)
# is chosen by RETRIEVER_BACKEND.  The cross-encoder model inside
# CrossEncoderReranker is loaded lazily on first use.
_retriever = get_retriever()
_reranker = CrossEncoderReranker()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Multiplier applied to TOP_K_RERANKED for queries that need evidence from
# more than one document, so a fixed 8-chunk cap doesn't get filled entirely
# by one source document and starve out a second one the query also needs.
_MULTI_HOP_RERANK_MULTIPLIER: int = 2


def _resolve_rerank_top_k(query_analysis: QueryAnalysis) -> int | None:
    """Widen the post-rerank chunk budget for multi-document queries.

    Args:
        query_analysis: The QueryAnalysis for the current query.

    Returns:
        An explicit top_k value (TOP_K_RERANKED doubled) when the query
        requires cross-document reasoning, or None to let the reranker fall
        back to its TOP_K_RERANKED env-var default for all other queries.
    """
    needs_wide_context = (
        query_analysis.requires_multi_hop
        or query_analysis.query_type.value == "cross_document"
    )
    if not needs_wide_context:
        return None
    base = int(os.environ.get("TOP_K_RERANKED", 8))
    return base * _MULTI_HOP_RERANK_MULTIPLIER


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming query payload for the /query/stream endpoint."""

    question: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/stream")
async def stream_query(request: QueryRequest) -> StreamingResponse:
    """Run the full RAG pipeline and stream each step as a Server-Sent Event.

    Orchestrates query analysis → expansion → hybrid retrieval → reranking
    → retrieval critic → (optional second pass) → generation.  Each step
    is emitted to the client as it completes so the frontend can display
    live pipeline progress.

    Args:
        request: A :class:`QueryRequest` containing the user's ``question``.

    Returns:
        A :class:`fastapi.responses.StreamingResponse` with
        ``media_type="text/event-stream"``.  Each SSE event is a JSON object
        with ``step`` and ``payload`` keys.
    """

    async def event_generator():
        """Async generator that yields SSE-formatted strings for each step."""
        start_time = time.monotonic()
        trace_id = str(uuid.uuid4())

        # Accumulate pipeline state for the final trace
        query_analysis: QueryAnalysis | None = None
        expanded_queries: ExpandedQueries | None = None
        first_merged: list[Chunk] = []
        first_reranked: list[Chunk] = []
        first_verdict: CriticVerdict | None = None
        second_pass = False
        final_chunks: list[Chunk] = []
        final_verdict: CriticVerdict | None = None

        try:
            # ── Step 1: Query analysis ─────────────────────────────────────
            query_analysis = await asyncio.to_thread(
                analyze_query, request.question
            )
            yield _sse("query_analyzed", query_analysis.model_dump())

            # ── Step 2: Query expansion ────────────────────────────────────
            expanded_queries = await asyncio.to_thread(
                expand_query, request.question
            )
            yield _sse("queries_expanded", expanded_queries.model_dump())

            # ── Step 3: Hybrid retrieval (first pass) ──────────────────────
            all_queries = [expanded_queries.original] + expanded_queries.variants
            first_merged = await asyncio.to_thread(
                _retriever.retrieve,
                all_queries,
                request.question,
                query_analysis,
            )
            yield _sse("chunks_retrieved", _chunk_summaries(first_merged))

            # ── Step 4: Reranking (first pass) ────────────────────────────
            rerank_top_k = _resolve_rerank_top_k(query_analysis)
            first_reranked = await asyncio.to_thread(
                _reranker.rerank, request.question, first_merged, rerank_top_k
            )
            yield _sse("chunks_reranked", _chunk_summaries(first_reranked))

            # ── Step 5: Retrieval critic (first pass) ─────────────────────
            first_verdict = await asyncio.to_thread(
                assess_retrieval, request.question, first_reranked
            )
            # Always emit the first-pass verdict so the frontend has context
            yield _sse("critic_verdict", first_verdict.model_dump())

            final_chunks = first_reranked
            final_verdict = first_verdict

            # ── Step 6: Second pass (if critic flagged insufficient) ───────
            if not first_verdict.sufficient and first_verdict.refined_query:
                second_pass = True
                refined_query: str = first_verdict.refined_query
                yield _sse("second_pass_triggered", {"refined_query": refined_query})

                merged2 = await asyncio.to_thread(
                    _retriever.retrieve,
                    [refined_query],
                    refined_query,
                    query_analysis,
                )
                reranked2 = await asyncio.to_thread(
                    _reranker.rerank, refined_query, merged2, rerank_top_k
                )
                # Re-assess the critic on second-pass results so the generator
                # receives an accurate sufficiency signal.
                second_verdict = await asyncio.to_thread(
                    assess_retrieval, refined_query, reranked2
                )
                final_chunks = reranked2
                final_verdict = second_verdict

            # ── Step 7: Generation ─────────────────────────────────────────
            answer = await asyncio.to_thread(
                generate_answer,
                request.question,
                final_chunks,
                final_verdict,
                query_analysis.query_type.value,
            )
            yield _sse("answer_generated", answer.model_dump())

            # ── Trace persistence ──────────────────────────────────────────
            elapsed_ms = (time.monotonic() - start_time) * 1000
            trace = PipelineTrace(
                trace_id=trace_id,
                question=request.question,
                query_analysis=query_analysis,
                expanded_queries=expanded_queries,
                chunks_retrieved=first_merged,
                chunks_reranked=first_reranked,
                critic_verdict=first_verdict,
                second_pass_triggered=second_pass,
                final_answer=answer,
                retrieval_passes=2 if second_pass else 1,
                total_latency_ms=elapsed_ms,
            )
            await asyncio.to_thread(_write_trace, trace)

        except Exception as exc:  # noqa: BLE001
            yield _sse(
                "error",
                {"message": str(exc), "type": type(exc).__name__},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sse(step: str, payload: dict) -> str:
    """Format a single Server-Sent Event string.

    Args:
        step: Event name string (e.g. ``"query_analyzed"``).
        payload: JSON-serialisable dict to include as the event payload.

    Returns:
        A complete SSE message string ending with the required double newline.
    """
    data = json.dumps({"step": step, "payload": payload}, default=str)
    return f"data: {data}\n\n"


def _chunk_summaries(chunks: list[Chunk]) -> list[dict]:
    """Produce lightweight chunk summary dicts for SSE streaming.

    Full chunk content is omitted from SSE events (it can be retrieved from
    the trace) to keep the event payload small.  The frontend uses the
    chunk summaries to display a live retrieval result list with scores.

    Args:
        chunks: List of :class:`~generation.schemas.Chunk` objects to
            summarise.

    Returns:
        List of dicts, one per chunk, containing ``chunk_id``,
        ``document_id``, ``document_type``, and whichever score fields are
        non-null.
    """
    summaries = []
    for chunk in chunks:
        entry: dict = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_type": chunk.document_type,
        }
        if chunk.dense_score is not None:
            entry["dense_score"] = round(chunk.dense_score, 6)
        if chunk.sparse_score is not None:
            entry["sparse_score"] = round(chunk.sparse_score, 6)
        if chunk.rerank_score is not None:
            entry["rerank_score"] = round(chunk.rerank_score, 4)
        summaries.append(entry)
    return summaries


def _write_trace(trace: PipelineTrace) -> None:
    """Serialise and write a pipeline trace to disk.

    Creates the traces directory if it does not exist.  The filename is
    ``{trace_id}.json``.  Called via ``asyncio.to_thread`` from the async
    event generator so file I/O does not block the event loop.

    Args:
        trace: The completed :class:`~generation.schemas.PipelineTrace` to
            persist.
    """
    traces_dir = Path(os.environ.get("TRACES_DIR", "./traces"))
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{trace.trace_id}.json"
    trace_path.write_text(
        trace.model_dump_json(indent=2),
        encoding="utf-8",
    )
