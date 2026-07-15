"""FastAPI application entry point for the BuildCore RAG API.

Loads environment configuration, applies CORS, and mounts the query,
evaluation, and trace routers.

CORS origins
------------
The browser enforces same-origin policy on every call the frontend makes, so
the deployed frontend's origin must be allowed explicitly.  ``CORS_ORIGINS`` is
a comma-separated list; it defaults to the local Vite dev server so that a
fresh clone works with no configuration.  In Azure this is set to the frontend
Container App's FQDN (the frontend calls the backend's public URL directly
rather than being proxied behind it).
"""

from dotenv import load_dotenv

load_dotenv()

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query, evaluate, traces

logger = logging.getLogger(__name__)

_DEFAULT_CORS_ORIGINS: str = "http://localhost:5173"

app = FastAPI(
    title="BuildCore RAG API",
    description="Enterprise RAG system with multi-layer retrieval and evaluation",
    version="1.0.0",
)


def _resolve_cors_origins() -> list[str]:
    """Parse the allowed CORS origins from the environment.

    Returns:
        Origins from ``CORS_ORIGINS`` (comma-separated), falling back to the
        local Vite dev server. Blank entries are ignored so that a trailing
        comma is harmless.
    """
    raw = os.environ.get("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS)
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return origins or [_DEFAULT_CORS_ORIGINS]


_origins = _resolve_cors_origins()
logger.info("CORS allowed origins: %s", _origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["evaluate"])
app.include_router(traces.router, prefix="/traces", tags=["traces"])


@app.get("/health")
async def health():
    """Liveness/readiness probe for Container Apps.

    Deliberately does no downstream work: it must answer while the retriever
    is still warming up, or the platform will restart the container before it
    can serve its first request.

    Returns:
        A JSON object reporting service health and the active backends, which
        makes a misconfigured deployment obvious from one curl.
    """
    return {
        "status": "ok",
        "retriever_backend": os.environ.get("RETRIEVER_BACKEND", "local"),
        "reranker_backend": os.environ.get("RERANKER_BACKEND", "cross_encoder"),
        "llm_backend": os.environ.get("LLM_BACKEND", "openai"),
    }
