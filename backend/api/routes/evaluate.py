from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/run")
async def run_evaluation():
    """
    Runs the full evaluation suite against both the BuildCore RAG system
    and the naive baseline. Streams results row by row via SSE.
    """
    async def event_generator():
        yield "data: {\"status\": \"started\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
