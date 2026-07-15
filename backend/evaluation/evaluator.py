"""Evaluation orchestrator for the BuildCore RAG system.

Runs the full 50-item test suite against both the production pipeline and
the naive baseline, scores every result, and returns a structured
:class:`EvaluationReport`.

Pipeline used during evaluation
---------------------------------
The evaluation pipeline mirrors the production pipeline in ``query.py``
exactly, including the second retrieval pass: if the first-pass retrieval
critic returns ``sufficient=False`` and supplies a ``refined_query``, a full
second dense + sparse + hybrid + rerank + critic cycle runs before
generation, identical to the production behaviour.  This costs extra LLM
calls only for the subset of items whose first-pass retrieval the critic
actually flags as insufficient — evaluating against a weaker pipeline than
production actually runs would understate the system's real quality.

Module-level singletons
------------------------
The :class:`~retrieval.base.Retriever` backend (chosen by ``RETRIEVER_BACKEND``
via :func:`~retrieval.factory.get_retriever`) and the
:class:`~retrieval.reranker.CrossEncoderReranker` are instantiated once at
module load, matching the production route so evaluation scores the shipping
system.  The cross-encoder model is loaded lazily on first use, so the first
evaluated item is slower than subsequent ones.

Exported symbols
-----------------
- :class:`ScoreSet` — aggregate scores for one system (system or baseline)
- :class:`ItemResult` — per-question evaluation result
- :class:`EvaluationReport` — full report returned by :func:`run_evaluation_suite`
- :func:`load_test_suite` — loads items from ``test_suite.json``
- :func:`evaluate_single_item` — evaluates one test item (called by the SSE route)
- :func:`run_evaluation_suite` — runs all items and returns the complete report
"""

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evaluation.baseline import run_baseline
from evaluation.metrics import (
    FaithfulnessScore,
    score_citation_presence,
    score_faithfulness,
    score_refusal_accuracy,
)
from generation.generator import generate_answer
from generation.schemas import GeneratedAnswer
from retrieval.factory import get_retriever
from retrieval.query_analyzer import analyze_query
from retrieval.query_expander import expand_query
from retrieval.reranker import CrossEncoderReranker
from retrieval.retrieval_critic import assess_retrieval

# ---------------------------------------------------------------------------
# Module-level singletons (instantiated once at import time)
# ---------------------------------------------------------------------------

# Same retriever backend as the production route, so evaluation scores the
# system that actually ships (local ChromaDB+BM25 or Azure AI Search).
_retriever = get_retriever()
_reranker = CrossEncoderReranker()

# Path to test_suite.json, resolved relative to this file
_TEST_SUITE_PATH: Path = Path(__file__).parent / "test_suite.json"

# Multiplier applied to TOP_K_RERANKED for queries that need evidence from
# more than one document, so a fixed 8-chunk cap doesn't get filled entirely
# by one source document and starve out a second one the query also needs.
_MULTI_HOP_RERANK_MULTIPLIER: int = 2


def _resolve_rerank_top_k(query_analysis) -> int | None:
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
# Pydantic report models
# ---------------------------------------------------------------------------


class ScoreSet(BaseModel):
    """Aggregate scores for one system (full pipeline or baseline)."""

    avg_faithfulness: float = Field(description="Mean faithfulness score across all items")
    avg_citation_presence: float = Field(description="Mean citation presence score across all items")
    avg_refusal_accuracy: float = Field(description="Mean refusal accuracy score across all items")
    overall: float = Field(description="Mean of the three average scores")


class ItemResult(BaseModel):
    """Per-question evaluation result for both the full pipeline and baseline."""

    id: str
    question: str
    difficulty: str
    source_documents: list[str]

    # Full pipeline
    system_answer: str
    system_faithfulness: float
    system_faithfulness_reasoning: str
    system_citation_presence: float
    system_refusal_accuracy: float
    system_overall: float
    system_refused: bool

    # Naive baseline
    baseline_answer: str
    baseline_faithfulness: float
    baseline_faithfulness_reasoning: str
    baseline_citation_presence: float
    baseline_refusal_accuracy: float
    baseline_overall: float

    # Verdict: system overall >= 0.7
    passed: bool


class EvaluationReport(BaseModel):
    """Full evaluation report over all test suite items."""

    total_questions: int
    system_scores: ScoreSet
    baseline_scores: ScoreSet
    per_item_results: list[ItemResult]
    delta: float = Field(
        description="system_scores.overall minus baseline_scores.overall"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_test_suite() -> list[dict[str, Any]]:
    """Load and return all items from ``test_suite.json``.

    Args: None.

    Returns:
        List of test item dicts, each with keys: ``id``, ``difficulty``,
        ``question``, ``expected_answer``, ``source_documents``,
        ``evaluation_notes``.
    """
    with _TEST_SUITE_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["items"]


def _run_full_pipeline(question: str, query_analysis=None) -> GeneratedAnswer:
    """Run the production pipeline, including the second pass, and return a GeneratedAnswer.

    Mirrors the logic in ``api/routes/query.py`` exactly (query analysis,
    expansion, hybrid retrieval, reranking, critic, optional second pass,
    generation), only omitting SSE event emission and trace persistence.

    Args:
        question: Raw user question string.
        query_analysis: Pre-computed QueryAnalysis (optional; if None it is
            computed here).

    Returns:
        A :class:`~generation.schemas.GeneratedAnswer`.
    """
    if query_analysis is None:
        query_analysis = analyze_query(question)

    expanded = expand_query(question)
    all_queries = [expanded.original] + expanded.variants

    rerank_top_k = _resolve_rerank_top_k(query_analysis)

    merged = _retriever.retrieve(all_queries, question, query_analysis)
    reranked = _reranker.rerank(question, merged, rerank_top_k)
    verdict = assess_retrieval(question, reranked)

    final_chunks = reranked
    final_verdict = verdict

    if not verdict.sufficient and verdict.refined_query:
        refined_query = verdict.refined_query
        merged2 = _retriever.retrieve([refined_query], refined_query, query_analysis)
        reranked2 = _reranker.rerank(refined_query, merged2, rerank_top_k)
        second_verdict = assess_retrieval(refined_query, reranked2)
        final_chunks = reranked2
        final_verdict = second_verdict

    return generate_answer(
        question,
        final_chunks,
        final_verdict,
        query_analysis.query_type.value,
    )


def _score_system_answer(
    system_answer: GeneratedAnswer,
    baseline_raw: str,
    item: dict[str, Any],
) -> tuple[FaithfulnessScore, FaithfulnessScore]:
    """Score both the system and baseline answers for faithfulness.

    Returns two FaithfulnessScore objects: one for the system answer and
    one for the baseline answer.  The baseline answer is a plain string, so
    it is scored against the same expected answer and rubric.

    Args:
        system_answer: GeneratedAnswer from the full pipeline.
        baseline_raw: Plain text answer from the baseline.
        item: Test suite item dict.

    Returns:
        Tuple of (system_faithfulness, baseline_faithfulness).
    """
    expected = item["expected_answer"]
    notes = item["evaluation_notes"]

    sys_faith = score_faithfulness(system_answer.answer, expected, notes)
    base_faith = score_faithfulness(baseline_raw, expected, notes)
    return sys_faith, base_faith


def _compute_overall(faithfulness: float, citation: float, refusal: float) -> float:
    """Compute the overall score as the mean of the three metrics.

    Args:
        faithfulness: Faithfulness score in [0, 1].
        citation: Citation presence score in [0, 1].
        refusal: Refusal accuracy score in 0 or 1.

    Returns:
        Mean of the three values.
    """
    return (faithfulness + citation + refusal) / 3.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_single_item(item: dict[str, Any]) -> ItemResult:
    """Run and score one test suite item against both pipelines.

    Calls the full production pipeline (one pass) and the naive baseline,
    then scores both using all three metrics.  This function is synchronous
    and blocking — call it via ``asyncio.to_thread`` from async code.

    Args:
        item: A test suite item dict as returned by :func:`load_test_suite`.

    Returns:
        A fully populated :class:`ItemResult`.
    """
    question: str = item["question"]
    difficulty: str = item["difficulty"]
    source_docs: list[str] = item["source_documents"]

    # --- Full pipeline ---
    system_answer = _run_full_pipeline(question)

    # --- Baseline ---
    baseline_raw = run_baseline(question)

    # Wrap baseline string in a minimal GeneratedAnswer for scoring.
    # The baseline never produces citations, so citation_presence will
    # reflect that gap correctly.
    baseline_answer_obj = GeneratedAnswer(
        answer=baseline_raw,
        citations=[],
        confidence=0.0,
        refused=False,
    )

    # --- Score faithfulness (LLM judge) ---
    sys_faith, base_faith = _score_system_answer(system_answer, baseline_raw, item)

    # --- Rule-based scores ---
    sys_citation = score_citation_presence(system_answer, source_docs)
    sys_refusal = score_refusal_accuracy(system_answer, difficulty)

    base_citation = score_citation_presence(baseline_answer_obj, source_docs)
    base_refusal = score_refusal_accuracy(baseline_answer_obj, difficulty)

    # --- Overall ---
    sys_overall = _compute_overall(sys_faith.score, sys_citation, sys_refusal)
    base_overall = _compute_overall(base_faith.score, base_citation, base_refusal)

    return ItemResult(
        id=item["id"],
        question=question,
        difficulty=difficulty,
        source_documents=source_docs,
        # System
        system_answer=system_answer.answer,
        system_faithfulness=sys_faith.score,
        system_faithfulness_reasoning=sys_faith.reasoning,
        system_citation_presence=sys_citation,
        system_refusal_accuracy=sys_refusal,
        system_overall=sys_overall,
        system_refused=system_answer.refused,
        # Baseline
        baseline_answer=baseline_raw,
        baseline_faithfulness=base_faith.score,
        baseline_faithfulness_reasoning=base_faith.reasoning,
        baseline_citation_presence=base_citation,
        baseline_refusal_accuracy=base_refusal,
        baseline_overall=base_overall,
        # Verdict
        passed=sys_overall >= 0.7,
    )


def run_evaluation_suite() -> EvaluationReport:
    """Run all test suite items and return the complete EvaluationReport.

    Iterates through all items in ``test_suite.json``, evaluates each with
    :func:`evaluate_single_item`, and aggregates results into an
    :class:`EvaluationReport`.

    Returns:
        A fully populated :class:`EvaluationReport` including per-item
        results, aggregate scores for both systems, and the delta.
    """
    items = load_test_suite()
    results: list[ItemResult] = []
    for item in items:
        results.append(evaluate_single_item(item))

    return _compile_report(results)


def _compile_report(results: list[ItemResult]) -> EvaluationReport:
    """Build an EvaluationReport from a list of ItemResults.

    Args:
        results: List of completed :class:`ItemResult` objects.

    Returns:
        A compiled :class:`EvaluationReport`.
    """
    n = len(results)

    def avg(values: list[float]) -> float:
        return sum(values) / n if n else 0.0

    sys_faith = avg([r.system_faithfulness for r in results])
    sys_cite = avg([r.system_citation_presence for r in results])
    sys_refusal = avg([r.system_refusal_accuracy for r in results])
    sys_overall = _compute_overall(sys_faith, sys_cite, sys_refusal)

    base_faith = avg([r.baseline_faithfulness for r in results])
    base_cite = avg([r.baseline_citation_presence for r in results])
    base_refusal = avg([r.baseline_refusal_accuracy for r in results])
    base_overall = _compute_overall(base_faith, base_cite, base_refusal)

    return EvaluationReport(
        total_questions=n,
        system_scores=ScoreSet(
            avg_faithfulness=sys_faith,
            avg_citation_presence=sys_cite,
            avg_refusal_accuracy=sys_refusal,
            overall=sys_overall,
        ),
        baseline_scores=ScoreSet(
            avg_faithfulness=base_faith,
            avg_citation_presence=base_cite,
            avg_refusal_accuracy=base_refusal,
            overall=base_overall,
        ),
        per_item_results=results,
        delta=round(sys_overall - base_overall, 4),
    )
