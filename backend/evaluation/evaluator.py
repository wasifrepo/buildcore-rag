"""Evaluation orchestrator for the BuildCore RAG system.

Runs the full 50-item test suite against both the production pipeline and
the naive baseline, scores every result, and returns a structured
:class:`EvaluationReport`.

Pipeline used during evaluation
---------------------------------
The evaluation pipeline mirrors the production pipeline in ``query.py`` with
one deliberate difference: **no second retrieval pass**.  If the retrieval
critic returns ``sufficient=False``, the evaluator proceeds directly to
generation with the first-pass chunks.  This keeps evaluation fast — 50
cases × 2 pipelines would take too long if every critic failure triggered
a full second pass.

Module-level singletons
------------------------
:class:`~retrieval.dense_retriever.DenseRetriever`,
:class:`~retrieval.sparse_retriever.SparseRetriever`, and
:class:`~retrieval.reranker.CrossEncoderReranker` are instantiated once at
module load.  The cross-encoder model is loaded lazily on first use, so the
first evaluated item is slower than subsequent ones.

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
import re
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
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import merge_results
from retrieval.query_analyzer import analyze_query
from retrieval.query_expander import expand_query
from retrieval.reranker import CrossEncoderReranker
from retrieval.retrieval_critic import assess_retrieval
from retrieval.sparse_retriever import SparseRetriever

# ---------------------------------------------------------------------------
# Module-level singletons (instantiated once at import time)
# ---------------------------------------------------------------------------

_dense_retriever = DenseRetriever()
_sparse_retriever = SparseRetriever()
_reranker = CrossEncoderReranker()

# Pattern for document_type filter hint in retrieval_strategy strings
_DOC_TYPE_FILTER_RE: re.Pattern[str] = re.compile(r"document_type=([a-z_]+)")

# Path to test_suite.json, resolved relative to this file
_TEST_SUITE_PATH: Path = Path(__file__).parent / "test_suite.json"


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
    """Run the production pipeline (one pass) and return a GeneratedAnswer.

    Mirrors the logic in ``api/routes/query.py`` but without the second
    retrieval pass and without writing a trace.

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

    doc_type_filter: str | None = None
    m = _DOC_TYPE_FILTER_RE.search(query_analysis.retrieval_strategy)
    if m:
        doc_type_filter = m.group(1)

    dense_chunks = _dense_retriever.search(all_queries, None, doc_type_filter)
    sparse_chunks = _sparse_retriever.search(question)
    merged = merge_results(dense_chunks, sparse_chunks, query_analysis)
    reranked = _reranker.rerank(question, merged)
    verdict = assess_retrieval(question, reranked)
    # One pass only — no second retrieval regardless of critic verdict
    return generate_answer(
        question,
        reranked,
        verdict,
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
