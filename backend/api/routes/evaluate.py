"""FastAPI SSE streaming endpoint for the BuildCore RAG evaluation suite.

``POST /evaluate/run``
    Runs all 50 test suite items against both the full BuildCore RAG pipeline
    and the naive baseline.  Streams a ``case_complete`` SSE event after each
    item completes, then a final ``evaluation_complete`` event with the
    aggregate report summary.  Saves the full report to
    ``{TRACES_DIR}/evaluation_report.json``.

``GET /evaluate/latest``
    Returns the most recently saved :class:`~evaluation.evaluator.EvaluationReport`
    from disk.  Returns ``404`` if no report has been saved yet.

SSE event format
-----------------
Every event follows the same envelope as the query pipeline::

    data: {"step": "<event_name>", "payload": { ... }}\n\n

``case_complete`` payload keys:
    - ``id`` — test case ID (e.g. ``"factual_01"``)
    - ``question`` — the question string
    - ``difficulty`` — difficulty tier
    - ``system_score`` — system overall score (0–1)
    - ``baseline_score`` — baseline overall score (0–1)
    - ``system_faithfulness`` — system faithfulness score
    - ``baseline_faithfulness`` — baseline faithfulness score
    - ``system_citation_presence`` — citation presence score
    - ``system_refusal_accuracy`` — refusal accuracy score
    - ``passed`` — bool (system_overall >= 0.7)

``evaluation_complete`` payload keys:
    - ``total_questions``
    - ``system_scores`` — avg faithfulness, citation, refusal, overall
    - ``baseline_scores`` — same structure
    - ``delta`` — system overall minus baseline overall
    - ``pass_rate`` — fraction of items where passed=True
"""

import asyncio
import json
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from evaluation.evaluator import (
    EvaluationReport,
    ItemResult,
    _compile_report,
    evaluate_single_item,
    load_test_suite,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /evaluate/run — stream evaluation results case-by-case
# ---------------------------------------------------------------------------


@router.post("/run")
async def run_evaluation() -> StreamingResponse:
    """Run the full evaluation suite and stream results as SSE events.

    Iterates through all items in ``test_suite.json``, evaluating each
    against both the full BuildCore RAG pipeline and the naive baseline.
    Emits a ``case_complete`` event after each item, then an
    ``evaluation_complete`` event with aggregate statistics after all items
    finish.  Saves the complete :class:`~evaluation.evaluator.EvaluationReport`
    to ``{TRACES_DIR}/evaluation_report.json``.

    Returns:
        A :class:`fastapi.responses.StreamingResponse` with
        ``media_type="text/event-stream"``.
    """

    async def event_generator():
        items = await asyncio.to_thread(load_test_suite)
        results: list[ItemResult] = []

        for item in items:
            try:
                result: ItemResult = await asyncio.to_thread(
                    evaluate_single_item, item
                )
                results.append(result)
                yield _sse("case_complete", _case_payload(result))
            except Exception as exc:  # noqa: BLE001
                yield _sse(
                    "case_error",
                    {
                        "id": item.get("id", "unknown"),
                        "error": str(exc),
                        "type": type(exc).__name__,
                    },
                )

        if results:
            report = await asyncio.to_thread(_compile_report, results)
            await asyncio.to_thread(_save_report, report)
            yield _sse("evaluation_complete", _summary_payload(report))

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
# GET /evaluate/latest — return the most recently saved report
# ---------------------------------------------------------------------------


@router.get("/latest")
async def get_latest_report() -> JSONResponse:
    """Return the most recently saved evaluation report from disk.

    Reads ``{TRACES_DIR}/evaluation_report.json`` if it exists and returns
    its contents as a JSON response.

    Returns:
        The saved :class:`~evaluation.evaluator.EvaluationReport` as JSON,
        or a ``404`` response if no report has been saved yet.
    """
    report_path = _report_path()
    if not report_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "No evaluation report found. Run /evaluate/run first."},
        )

    raw = await asyncio.to_thread(report_path.read_text, "utf-8")
    return JSONResponse(content=json.loads(raw))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sse(step: str, payload: dict) -> str:
    """Format a Server-Sent Event string.

    Args:
        step: Event name.
        payload: JSON-serialisable payload dict.

    Returns:
        Complete SSE message string ending with ``\\n\\n``.
    """
    data = json.dumps({"step": step, "payload": payload}, default=str)
    return f"data: {data}\n\n"


def _case_payload(result: ItemResult) -> dict:
    """Build the ``case_complete`` SSE payload from an ItemResult.

    Args:
        result: Completed per-item evaluation result.

    Returns:
        Dict with the fields documented in the module docstring.
    """
    return {
        "id": result.id,
        "question": result.question,
        "difficulty": result.difficulty,
        "system_score": round(result.system_overall, 4),
        "baseline_score": round(result.baseline_overall, 4),
        "system_faithfulness": round(result.system_faithfulness, 4),
        "baseline_faithfulness": round(result.baseline_faithfulness, 4),
        "system_citation_presence": round(result.system_citation_presence, 4),
        "system_refusal_accuracy": round(result.system_refusal_accuracy, 4),
        "system_refused": result.system_refused,
        "passed": result.passed,
    }


def _summary_payload(report: EvaluationReport) -> dict:
    """Build the ``evaluation_complete`` SSE payload from an EvaluationReport.

    Omits ``per_item_results`` to keep the final event small — the full
    report is available via ``GET /evaluate/latest``.

    Args:
        report: Completed evaluation report.

    Returns:
        Dict with aggregate statistics.
    """
    passed_count = sum(1 for r in report.per_item_results if r.passed)
    return {
        "total_questions": report.total_questions,
        "pass_rate": round(passed_count / report.total_questions, 4)
        if report.total_questions
        else 0.0,
        "system_scores": report.system_scores.model_dump(),
        "baseline_scores": report.baseline_scores.model_dump(),
        "delta": report.delta,
    }


def _report_path() -> Path:
    """Return the filesystem path for the saved evaluation report.

    Returns:
        ``{TRACES_DIR}/evaluation_report.json``
    """
    traces_dir = Path(os.environ.get("TRACES_DIR", "./traces"))
    traces_dir.mkdir(parents=True, exist_ok=True)
    return traces_dir / "evaluation_report.json"


def _save_report(report: EvaluationReport) -> None:
    """Serialise and save the evaluation report to disk.

    Args:
        report: The completed :class:`~evaluation.evaluator.EvaluationReport`.
    """
    path = _report_path()
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
