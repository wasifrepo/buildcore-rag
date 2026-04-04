# BuildCore RAG — Claude Code Context

## What this project is
A portfolio project demonstrating enterprise-grade RAG with a multi-layer retrieval architecture
and built-in evaluation system. Demo scenario: BuildCore Operations, a fictional mid-size
construction and facilities management company.

This is intentionally built WITHOUT LangChain. Every retrieval layer is a standalone module
so the architecture is visible, explainable, and defensible.

## Stack
- Backend: Python 3.11, FastAPI, uvicorn
- Frontend: React 18, Vite, react-router-dom
- Vector store: ChromaDB (persisted locally)
- Embeddings: OpenAI text-embedding-3-small
- Query analysis + retrieval critic: GPT-4o-mini (structured outputs via Pydantic)
- Final generation: GPT-4o
- Sparse retrieval: rank-bm25
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (HuggingFace, runs locally)
- Ingestion: unstructured library
- Evaluation: custom harness + RAGAS metrics
- Infra: Docker Compose (backend + frontend as separate services)

## Retrieval pipeline — layer order
1. Query classification and intent analysis (query_analyzer.py)
2. Multi-query expansion — 3 variants generated (query_expander.py)
3. Hybrid retrieval — dense (ChromaDB) + sparse (BM25) merged (hybrid_retriever.py)
4. Cross-encoder reranking (reranker.py)
5. Retrieval critic — LLM-as-judge, triggers second pass if confidence is low (retrieval_critic.py)
6. Generation with inline citation tracking (generator.py)

## Document types in corpus
- safety_sops/ — section-aware chunking (respects headers and numbered sections)
- contracts/ — clause and table-aware chunking
- incident_emails/ — thread-aware chunking (preserves sender, date, subject context)
- maintenance_manuals/ — step-aware chunking (numbered procedures stay intact)
- compliance_checklists/ — row-aware chunking (tabular structure preserved)

## Code conventions
- Every function has a docstring
- Pydantic for ALL structured data — see generation/schemas.py for all models
- No placeholder comments like "add logic here" — write the full implementation
- Each module has one responsibility
- Every pipeline run writes a full JSON trace to backend/traces/
- Structured outputs from LLMs always use response_format with Pydantic models

## Key Pydantic models (generation/schemas.py)
- QueryAnalysis — output of query classifier
- ExpandedQueries — output of query expander
- Chunk — retrieved chunk with dense/sparse/rerank scores
- CriticVerdict — retrieval critic output, includes refined_query if second pass needed
- GeneratedAnswer — final answer with citations list
- PipelineTrace — full trace of a single pipeline run

## API routes
- POST /query/stream — runs full pipeline, streams steps via SSE
- POST /evaluate/run — runs test suite, streams results via SSE
- GET /traces/ — list all traces
- GET /traces/{trace_id} — get single trace

## Frontend structure
- QueryPage — query input, live SSE pipeline steps, answer with citations
- EvalPage — trigger eval run, results table, system vs baseline comparison card
- TracesPage — searchable trace history, expandable JSON per trace
- useSSE.js — hook for consuming SSE streams
- api.js — all fetch calls, SSE stream handler

## Environment variables
See .env.example. Copy to .env before running. Never commit .env.

## Running locally (without Docker)
```
# Backend
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

## Running with Docker
```
docker-compose up --build
```

## What NOT to do
- Do not use LangChain or LlamaIndex
- Do not use fixed-size naive chunking anywhere
- Do not hardcode API keys
- Do not skip docstrings
- Do not write partial implementations — complete every function fully
