from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query, evaluate, traces

app = FastAPI(
    title="BuildCore RAG API",
    description="Enterprise RAG system with multi-layer retrieval and evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["evaluate"])
app.include_router(traces.router, prefix="/traces", tags=["traces"])


@app.get("/health")
async def health():
    return {"status": "ok"}
