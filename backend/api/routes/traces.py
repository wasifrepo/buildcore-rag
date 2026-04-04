from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_traces():
    """Returns a list of all stored query traces, newest first."""
    return {"traces": []}


@router.get("/{trace_id}")
async def get_trace(trace_id: str):
    """Returns the full JSON reasoning trace for a specific query."""
    return {"trace_id": trace_id, "trace": {}}
