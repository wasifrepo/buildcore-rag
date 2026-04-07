![BuildCore Intelligence UI](docs/screenshot.png)

# BuildCore Intelligence

An enterprise-grade retrieval-augmented generation system for construction
and facilities management operations. Built to demonstrate what production
RAG looks like beyond the tutorial — multi-layer retrieval architecture,
document-type-aware ingestion, a self-correcting retrieval critic, and a
built-in evaluation framework with measurable before/after results.

---

## The Problem with Naive RAG

The standard RAG implementation — embed a query, retrieve top-k by cosine
similarity, generate an answer — works in demos but breaks in production
for three specific reasons.

**Retrieval is a single undifferentiated step.** A query asking for a
specific contract clause and a query asking how to inspect a harness both
get treated identically. There is no routing, no strategy differentiation,
no awareness that these are fundamentally different retrieval tasks.

**Fixed-size chunking destroys document structure.** Splitting a numbered
safety procedure mid-step, separating a financial table from its headers,
or merging three unrelated email messages into one chunk produces broken
fragments. The retrieval system then operates on semantically corrupted
inputs.

**There is no quality gate before generation.** The model receives whatever
cosine similarity returned and generates an answer — regardless of whether
the evidence is strong or weak. Hallucination is invisible because the
system has no mechanism to say "I don't have enough information to answer
this reliably."

BuildCore Intelligence addresses all three.

---

## Architecture

The pipeline has six layers. Each exists for a specific, defensible reason.

### Layer 1 — Query Analysis

Before touching the vector store, the incoming query is classified using
GPT-4o-mini with structured output. Queries are routed into one of five
types: `factual`, `procedural`, `cross_document`, `ambiguous`, or
`out_of_scope`. The classification drives retrieval strategy — a factual
lookup against contracts uses a different approach than a procedural query
about maintenance steps. Out-of-scope queries are refused before any
retrieval occurs, preventing the generator from hallucinating answers to
questions the corpus cannot support.

### Layer 2 — Query Expansion

The original query is rephrased into three variants targeting different
vocabulary, specificity levels, and phrasing styles. All four queries are
used independently for dense retrieval, with results merged and
deduplicated by chunk ID.

A worker asking "what do I do if there's a gas leak on the forklift" uses
different vocabulary than the maintenance manual's "Apply soapy water to
connections — any bubbling indicates a gas leak." Query expansion bridges
that vocabulary gap without requiring the user to know the exact phrasing
in the source document.

### Layer 3 — Hybrid Retrieval

Two retrievers run in parallel:

**Dense retriever** — embeds all four query variants using
`text-embedding-3-small`, queries ChromaDB for nearest neighbours per
variant, merges results keeping the best cosine similarity score per chunk.

**Sparse retriever** — BM25 keyword search via `rank_bm25` over the same
corpus. Effective for exact matches that semantic search misses: document
IDs, form numbers, clause references, and specific technical identifiers.

Results are combined using **Reciprocal Rank Fusion** (RRF, k=60), which
rewards chunks that rank highly in both lists without requiring score
normalisation between the two systems.

### Layer 4 — Cross-Encoder Reranking

The merged hybrid results are reranked using
`cross-encoder/ms-marco-MiniLM-L-6-v2`. A cross-encoder sees the query
and each candidate chunk together and scores them jointly — significantly
more accurate than bi-encoder cosine similarity for final-stage ranking.
The cost is acceptable when operating on 20–40 candidates rather than the
full corpus.

### Layer 5 — Retrieval Critic

Before any generation occurs, an LLM-as-judge step evaluates whether the
reranked chunks are actually sufficient to answer the query. If the verdict
is insufficient, the critic generates a refined query and triggers a second
full retrieval pass. The generator only runs after the critic is satisfied.

This is the quality gate that prevents hallucination. The system will
refuse to answer rather than generate a plausible-sounding response from
weak evidence. In construction and facilities management — where an
incorrect answer about a safety procedure or contract clause has real
consequences — this is not optional.

### Layer 6 — Grounded Generation with Citations

GPT-4o generates the final answer under strict instructions: use only the
provided chunks, cite every factual claim by document ID, and return a
structured response with inline citations and verbatim excerpts. Every
answer is fully traceable to its source document and section.

---

## Document-Type-Aware Ingestion

The corpus contains five document types, each requiring a different
chunking strategy. Fixed-size naive chunking fails all of them.

| Document Type | Chunking Strategy | Key Challenge Solved |
|---|---|---|
| Safety SOPs | Section-aware, split on `━` separators | Subsections stay with parent section headers |
| Contracts | Schedule-level chunks, clause subdivision | Financial tables never split across chunks |
| Incident Emails | One chunk per message, full header preserved | Sender, date, subject co-embedded with body |
| Maintenance Manuals | Section-level, step sequences preserved | `STEP N` and sub-steps always stay together |
| Compliance Checklists | Section-level, tabular rows intact | Item codes stay with their descriptions |

Each document is classified on ingestion and routed to the appropriate
chunker. Chunk IDs are deterministic SHA-256 hashes of document ID and
content — re-ingesting an unchanged corpus is fully idempotent.

**This system is deliberately built without LangChain.** Every retrieval
and ingestion layer is a standalone module. Each architectural decision is
visible, explainable, and debuggable — without the abstraction overhead
that makes LangChain-based systems difficult to diagnose in production.

---

## Corpus

The BuildCore corpus contains 12 documents across five types, producing
90 chunks after ingestion:

**Safety SOPs (3)** — Fall Protection (SOP-001), Scaffold Safety (SOP-005),
Hazard Communication (SOP-007)

**Subcontractor Contracts (2)** — Harrington Electrical Services
(SC-2024-041, $366k scope), Apex Plumbing & Drainage (SC-2024-038, $191k
scope)

**Incident Emails (3)** — Forklift near-miss with investigation and
corrective actions (INC-2024-007), Laceration LTI (INC-2024-002), Epoxy
chemical spill (INC-2024-009)

**Maintenance Manuals (2)** — Toyota 8FGF25 forklift (MAINT-FLT-03),
Denyo DCA-45SPK3 generator (MAINT-GEN-01)

**Compliance Checklists (2)** — Daily site safety inspection (SSIC-001),
Subcontractor pre-mobilisation checklist (SC-PMCL-001)

The document mix is intentionally heterogeneous — five structurally
different document types, mixed formatting, tables, email threads, and
numbered procedures — to stress-test every layer of the ingestion and
retrieval pipeline.

---

## Evaluation Results

The system was evaluated against a hand-crafted test suite of 50 questions
across four difficulty levels: simple factual (15), procedural (15),
multi-hop cross-document reasoning (10), and out-of-scope refusals (10).

Each answer was scored by an LLM judge on three metrics: faithfulness to
source documents, citation presence, and refusal accuracy.

| Metric | BuildCore Intelligence | Naive Baseline | Delta |
|---|---|---|---|
| Faithfulness | 76.4% | 66.8% | +9.6pp |
| Citation presence | **80.0%** | **20.0%** | **+60.0pp** |
| Refusal accuracy | 80.0% | 80.0% | 0.0pp |
| **Overall** | **78.8%** | **55.6%** | **+23.2pp** |

![Evaluation Dashboard](docs/evaluation.png)

The citation presence result is the most significant finding. The naive
baseline generates answers without citing sources 80% of the time. The
full pipeline cites sources on 80% of answers — a 4x improvement. For
enterprise deployments where auditability matters, this is the difference
between a system that can be trusted and one that cannot.

The naive baseline used for comparison: dense-only retrieval, no query
expansion, no reranking, no retrieval critic, plain GPT-4o generation with
no structured output or citation requirements.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, uvicorn |
| Frontend | React 18, Vite |
| Vector store | ChromaDB (cosine similarity, persistent) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Query analysis, expansion, critic | GPT-4o-mini (structured output) |
| Final generation | GPT-4o |
| Sparse retrieval | BM25 via `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rank fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Containerisation | Docker Compose |

---

## Running Locally

**Prerequisites:** Docker Desktop, OpenAI API key

```bash
# 1. Clone and configure
git clone https://github.com/YOUR_USERNAME/buildcore-rag.git
cd buildcore-rag
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Start the system
docker-compose up --build

# 3. Run ingestion (first time only — builds the ChromaDB index)
docker exec buildcore-backend python -c \
  "from ingestion.pipeline import run_ingestion; \
   run_ingestion(data_dir='/app/data/raw')"

# 4. Open the UI
# http://localhost:5173
```

The ingestion step embeds all 12 documents and upserts 90 chunks into
ChromaDB. It runs once — the index persists across container restarts.
Re-run only when documents change.

---

## Project Structure

```
buildcore-rag/
├── backend/
│   ├── api/
│   │   └── routes/          # FastAPI SSE endpoints (query, evaluate, traces)
│   ├── ingestion/
│   │   ├── classifier.py    # Document type detection
│   │   ├── pipeline.py      # Ingestion orchestrator with PDF support
│   │   └── chunkers/        # Five type-specific chunkers + base class
│   ├── retrieval/
│   │   ├── query_analyzer.py    # Intent classification and routing
│   │   ├── query_expander.py    # Multi-query generation
│   │   ├── dense_retriever.py   # ChromaDB semantic search
│   │   ├── sparse_retriever.py  # BM25 keyword search
│   │   ├── hybrid_retriever.py  # RRF rank fusion
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── retrieval_critic.py  # LLM-as-judge sufficiency check
│   ├── generation/
│   │   ├── generator.py     # Grounded generation with citations
│   │   └── schemas.py       # Pydantic models for entire pipeline
│   └── evaluation/
│       ├── test_suite.json  # 50 hand-crafted Q&A pairs
│       ├── baseline.py      # Naive RAG comparison system
│       ├── metrics.py       # LLM-judge scoring functions
│       └── evaluator.py     # Full evaluation orchestrator
├── frontend/
│   └── src/
│       ├── pages/           # Query, Evaluation, Traces
│       ├── components/      # Reusable UI components
│       ├── hooks/           # useSSE, useEvaluation, useTraces
│       └── utils/           # API client, formatters
└── data/
    └── raw/                 # BuildCore corpus (5 document type folders)
```

---

## Key Design Decisions

**No LangChain.** LangChain abstracts away exactly the layers this system
is designed to make visible. Building without it means every decision is
explicit, every failure is traceable, and every component is replaceable
without framework constraints.

**Retrieval critic before generation.** Most RAG systems pass whatever
retrieval returns directly to the generator. This system inserts a
judgement step. The cost is one additional LLM call per query. The benefit
is a system that knows when it doesn't know enough — and says so instead
of hallucinating.

**Document-type-aware chunking over fixed-size splitting.** The ingestion
pipeline classifies each document before chunking. A safety SOP with
nested numbered sections is chunked differently from a contract with
financial tables, which is chunked differently from a multi-message email
thread. The retrieval quality difference is significant and measurable.

**Evaluation as a first-class component.** The test suite was written
before the evaluator, and the evaluator scores both the full pipeline and
a naive baseline. The +23.2pp overall improvement and the 4x citation
presence improvement are not estimated — they are measured against 50
hand-crafted test cases with an LLM judge.
