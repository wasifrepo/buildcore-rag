"""FastAPI routes for browsing stored pipeline traces.

Every successful query run writes a :class:`~generation.schemas.PipelineTrace`
JSON file to ``TRACES_DIR`` (env var, default ``./traces``).  These endpoints
expose that directory as a simple read-only API.

``GET /traces/``
    Returns a list of trace summaries sorted newest first by file modification
    time.  Each summary contains the fields most useful for a trace history
    view: ID, question, latency, retrieval pass count, and answer confidence.

``GET /traces/{trace_id}``
    Returns the full :class:`~generation.schemas.PipelineTrace` JSON for a
    single trace.  Returns ``404`` if the trace file does not exist.

The evaluation report (``evaluation_report.json``) written to the same
directory by the evaluation route is excluded from the trace listing.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

# Filename written by the evaluation route — excluded from trace listings
_EVAL_REPORT_FILENAME = "evaluation_report.json"


# ---------------------------------------------------------------------------
# Response model for the summary list
# ---------------------------------------------------------------------------


class TraceSummary(BaseModel):
    """Lightweight summary of a single pipeline trace for list views."""

    trace_id: str
    question: str
    total_latency_ms: float
    retrieval_passes: int
    second_pass_triggered: bool
    answer_confidence: float
    answer_refused: bool
    timestamp: str  # ISO-8601 UTC, derived from file mtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _traces_dir() -> Path:
    """Return the resolved traces directory path.

    Returns:
        :class:`pathlib.Path` for the directory specified by ``TRACES_DIR``
        (default ``./traces``).  The directory is not created here — if it
        does not exist, the listing endpoints return empty results.
    """
    return Path(os.environ.get("TRACES_DIR", "./traces"))


def _trace_files() -> list[Path]:
    """Return all trace JSON files sorted by modification time, newest first.

    Excludes ``evaluation_report.json`` from the listing.

    Returns:
        List of :class:`pathlib.Path` objects for trace files.
    """
    traces_dir = _traces_dir()
    if not traces_dir.is_dir():
        return []

    files = [
        f
        for f in traces_dir.glob("*.json")
        if f.name != _EVAL_REPORT_FILENAME
    ]
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def _mtime_to_iso(path: Path) -> str:
    """Convert a file's modification time to an ISO-8601 UTC string.

    Args:
        path: Path to the file whose mtime is used.

    Returns:
        UTC timestamp string in ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` format.
    """
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _load_trace(path: Path) -> dict:
    """Read and JSON-parse a trace file.

    Args:
        path: Path to the ``.json`` trace file.

    Returns:
        Parsed dict representation of the
        :class:`~generation.schemas.PipelineTrace`.
    """
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/")
async def list_traces() -> JSONResponse:
    """Return a list of all stored pipeline traces, newest first.

    Reads every ``.json`` file in ``TRACES_DIR`` (excluding the evaluation
    report), extracts a lightweight summary from each, and returns the list
    sorted by file modification time descending.

    Returns:
        JSON array of :class:`TraceSummary` objects.  Returns an empty array
        if no traces have been written yet.
    """
    summaries: list[dict] = []

    for file_path in _trace_files():
        try:
            trace = _load_trace(file_path)
            summary = TraceSummary(
                trace_id=trace["trace_id"],
                question=trace["question"],
                total_latency_ms=trace.get("total_latency_ms", 0.0),
                retrieval_passes=trace.get("retrieval_passes", 1),
                second_pass_triggered=trace.get("second_pass_triggered", False),
                answer_confidence=trace.get("final_answer", {}).get("confidence", 0.0),
                answer_refused=trace.get("final_answer", {}).get("refused", False),
                timestamp=_mtime_to_iso(file_path),
            )
            summaries.append(summary.model_dump())
        except Exception:  # noqa: BLE001
            # Skip malformed or partially-written trace files
            continue

    return JSONResponse(content={"traces": summaries})


@router.get("/{trace_id}")
async def get_trace(trace_id: str) -> JSONResponse:
    """Return the full pipeline trace JSON for the specified trace ID.

    Constructs the expected filename ``{trace_id}.json`` and reads it from
    ``TRACES_DIR``.

    Args:
        trace_id: The UUID trace identifier (path parameter).

    Returns:
        The full :class:`~generation.schemas.PipelineTrace` JSON object,
        or a ``404`` response if no trace with that ID exists.
    """
    trace_path = _traces_dir() / f"{trace_id}.json"

    if not trace_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Trace '{trace_id}' not found."},
        )

    try:
        trace = _load_trace(trace_path)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to read trace: {exc}"},
        )

    return JSONResponse(content={"trace_id": trace_id, "trace": trace})
