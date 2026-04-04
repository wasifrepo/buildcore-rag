from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("/stream")
async def stream_query(request: QueryRequest):
    """
    Runs the full retrieval pipeline and streams each step via SSE.
    Steps emitted: query_analyzed, queries_expanded, chunks_retrieved,
    chunks_reranked, critic_verdict, answer_generated.
    """
    async def event_generator():
        # Pipeline steps will be yielded here as they complete
        # Each step: data: {"step": "...", "payload": {...}}\n\n
        yield "data: {\"step\": \"connected\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
