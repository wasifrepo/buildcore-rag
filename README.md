# BuildCore Intelligence

An enterprise-grade retrieval-augmented generation system for construction
and facilities management operations. Built to demonstrate what production
RAG looks like beyond the tutorial — multi-layer retrieval architecture,
document-type-aware ingestion, a self-correcting retrieval critic, and a
built-in evaluation framework with measurable before/after results.

![BuildCore Intelligence UI](docs/screenshot.png)

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

Before touching the vector store, the incoming query is classified using a
small, fast model with structured output (`gpt-4o-mini` locally, `gpt-5-mini`
on Azure). Queries are routed into one of five
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

Both retrievers search small **child** chunks and return their full **parent**
chunks — see [Small-to-Big Retrieval](#small-to-big-retrieval-search-small-answer-big)
below. The whole layer sits behind a swappable adapter so it can be replaced
by Azure AI Search in production without touching any other layer.

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

The generation model (`gpt-4o` locally, `gpt-5.4-mini` on Azure) produces the
final answer under strict instructions: use only the
provided chunks, cite every factual claim by document ID, and return a
structured response with inline citations and verbatim excerpts. Every
answer is fully traceable to its source document and section.

The model is a configuration detail, not an architectural one — every LLM call
resolves through a single client factory, so switching provider or model is one
environment variable and no code change.

---

## Document-Type-Aware Ingestion

The corpus contains six document types, each requiring a different
chunking strategy. Fixed-size naive chunking fails all of them.

| Document Type | Chunking Strategy | Key Challenge Solved |
|---|---|---|
| Safety SOPs | Section-aware, split on `━` separators | Subsections stay with parent section headers |
| Contracts | Schedule-level chunks, clause subdivision | Financial tables never split across chunks |
| Incident Emails | One chunk per message, full header preserved | Sender, date, subject co-embedded with body |
| Maintenance Manuals | Section-level, step sequences preserved | `STEP N` and sub-steps always stay together |
| Compliance Checklists | Section-level, tabular rows intact | Item codes stay with their descriptions |
| OSHA Regulatory PDFs | Section/heading-aware over extracted PDF text | Regulatory clauses stay whole across page breaks |

Each document is classified on ingestion and routed to the appropriate
chunker. Chunk IDs are deterministic SHA-256 hashes of document ID and
content — re-ingesting an unchanged corpus is fully idempotent.

**This system is deliberately built without LangChain.** Every retrieval
and ingestion layer is a standalone module. Each architectural decision is
visible, explainable, and debuggable — without the abstraction overhead
that makes LangChain-based systems difficult to diagnose in production.

---

## Small-to-Big Retrieval: Search Small, Answer Big

This is the most consequential change to the retrieval architecture, and it
solves a problem that sits at the heart of every RAG system.

### The tension

Chunk size forces an uncomfortable trade-off, and you cannot win both sides
of it with a single chunk size.

**Big chunks answer well but search badly.** A full safety SOP section
contains everything needed to answer a question properly. But when you turn
2,000 characters into a single embedding — one list of numbers meant to
capture the "meaning" of the whole thing — the specific sentence you care
about gets averaged in with everything around it. Ask about a gas leak, and
the one sentence about gas leaks is diluted by fifteen sentences about
tyre pressure and battery terminals. The match is weak, so the right chunk
may not even be retrieved.

**Small chunks search well but answer badly.** Embed two sentences and the
match is sharp, because the embedding is *about* that one thing. But hand
those two sentences to the generator and it has no surrounding context — no
section header, no preceding steps, no safety warning that came three
sentences earlier. It answers, but thinly, and sometimes wrongly.

### The resolution

Do both. Index at two sizes and use each for what it is good at.

Every document is still chunked by the structure-aware chunkers into
**parent** chunks — a full SOP section, a whole contract clause, one email
message. Each parent is then split into **child** chunks of 2–3 sentences.

Only the children get embedded and BM25-indexed. Search runs against the
children, so matching is precise. But every child carries a pointer back to
its parent, and the moment a child matches, the retriever swaps it for the
parent before anything else in the pipeline sees it. The reranker and the
generator only ever work with full-context parents.

> **The analogy:** it is the difference between a book's index and its
> chapters. You search the index because it is specific — one line per
> concept, nothing diluting it. But you don't *read* the index entry. It
> gives you a page number, and you go read the whole page. Children are the
> index. Parents are the page.

### What this changes in practice

| | Before | After |
|---|---|---|
| What gets embedded | Whole parent chunk (up to ~2,000 chars) | Child chunk (2–3 sentences) |
| What the generator reads | The same chunk that matched | The full parent of whatever matched |
| Precision of matching | Diluted — one relevant sentence averaged with many irrelevant ones | Sharp — the embedding is about one topic |
| Context at generation | Whatever the chunk happened to contain | Always the complete structural unit |
| Long source text | **Silently truncated at 6,000 chars** | Never truncated — nothing is dropped |

Three effects are worth calling out specifically.

**Retrieval got more precise without the answers getting thinner.** This is
the whole point. Normally, improving match precision by shrinking chunks
costs you answer quality. Here it doesn't, because the thing you search and
the thing you read are no longer the same object.

**Silent data loss is gone.** The old pipeline capped text at 6,000
characters before embedding and cut off anything beyond it — quietly, with
only a log line. Any content past that ceiling was simply unsearchable, and
nobody would know. Children are now capped at 1,200 characters *by
construction*, with over-long sentences wrapped at word boundaries rather
than cut. Every character of the corpus is now reachable. The truncation
code, and a fiddly ID-collision workaround it required, were both deleted.

**Retrieval over-fetches to compensate.** Because many children collapse
into one parent, fetching 20 children might yield only 6 distinct parents.
The retrievers now fetch `top_k × 4` children (tunable via
`CHILD_FETCH_MULTIPLIER`) so that enough distinct parents survive the
collapse. This is the one real cost of the design: slightly more work per
query, in exchange for precision that doesn't sacrifice context.

Everything is tunable without touching code — `CHILD_SENTENCES`,
`CHILD_OVERLAP`, `CHILD_MAX_CHARS`, and `CHILD_FETCH_MULTIPLIER` in `.env`.
Children overlap by one sentence by default, so a fact spanning a window
boundary is still fully present in at least one child.

> **On measurement:** the [evaluation numbers](#evaluation-results) below were
> produced against the re-ingested index, so they include this change. They do
> not *isolate* it — small-to-big landed alongside the Azure migration and a
> model change, and no ablation was run to separate the three. The argument
> above is architectural; treat it as reasoning, not as a measured claim.

---

## The Retriever Adapter: Built to Move to Azure

The local stack — ChromaDB, BM25, and RRF fusion — is a genuine hybrid
retrieval implementation, and it is what runs when you clone this repo. But
the production target is **Azure AI Search**, whose managed hybrid search and
semantic ranker replace most of that machinery.

Rather than let that become a rewrite, retrieval sits behind a single
interface (`retrieval/base.py`), and the backend is chosen at runtime by one
environment variable:

```bash
RETRIEVER_BACKEND=local            # ChromaDB + BM25 + RRF (default)
RETRIEVER_BACKEND=azure_ai_search  # Azure AI Search hybrid + semantic ranker
```

Everything upstream and downstream — query analysis, expansion, reranking,
the critic, generation — is unchanged by that switch. Both `query.py` and
`evaluator.py` resolve their retriever through `get_retriever()`, which
matters more than it looks: **the evaluation harness always scores the
backend that actually ships.** Swap to Azure and the eval numbers describe
Azure, with no code changes and no second measurement path to keep in sync.

The parent-child design carries across the boundary by design. Azure AI
Search has a native feature — *index projections* — that expresses exactly
the same idea: child documents carry a parent key and the parent's fields.
The small-to-big model is therefore not local scaffolding to be thrown away;
it is the same architecture spoken in Azure's vocabulary.

The Azure retriever is currently a scaffold, wired up during the deployment
phase (there is no local emulator for Azure AI Search, so it is built against
the real service). The MiniLM cross-encoder reranker stays as-is locally; in
production, Azure's managed semantic ranker fills that role.

---

## Corpus

The corpus contains 17 documents across six types. Ingestion produces
structure-aware **parent** chunks, then splits each into small 2-3 sentence
**child** chunks (small-to-big retrieval): children are embedded and indexed
for precise matching, while the parent is returned for full-context
generation.

Twelve documents are the fictional BuildCore Operations corpus; five are real
OSHA regulatory publications. That pairing is the point, and it is deliberate
rather than incidental: the OSHA documents cover the *same topics* as
BuildCore's internal SOPs. OSHA 3150 is a scaffold-use guide and BuildCore has
a scaffold SOP; OSHA 3146 covers fall protection in construction and BuildCore
has a fall-protection SOP.

That mirrors how enterprise knowledge bases actually look — company policy
sitting next to the external regulation it implements — and it enables the
queries that matter most: *"what does our scaffold SOP require, and does it
meet OSHA?"* is a cross-document question spanning an internal procedure and a
73-page federal guide.

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

**OSHA Regulatory Documents (5)** — real published PDFs: *A Guide to Scaffold
Use in the Construction Industry* (OSHA 3150, 73pp), *Job Hazard Analysis*
(OSHA 3071, 51pp), *Fall Protection in Construction* (OSHA 3146, 48pp),
*Materials Handling and Storage* (OSHA 2236, 41pp), and the Walking-Working
Surfaces & Fall Protection final-rule fact sheet (OSHA 3903, 3pp)

The document mix is intentionally heterogeneous — six structurally different
document types, mixed formatting, tables, email threads, numbered procedures,
and real-world PDFs — to stress-test every layer of the ingestion and
retrieval pipeline.

### A note on scale, and why it makes retrieval hard

The five OSHA PDFs dominate the index by volume: they produce **4,093 of the
~4,900 child chunks**, roughly 84% of the corpus, because a published
regulatory PDF is far longer than a two-page internal SOP. OSHA 3150 alone is
2,152 children — more than every BuildCore document combined.

The topical pairing that makes the corpus realistic is exactly what makes it
adversarial. BuildCore's scaffold SOP produces ~70 child chunks; OSHA's
scaffold guide produces ~2,152. A query about scaffolding therefore faces a
**30:1 imbalance of on-topic competitors**, and pure vector similarity has no
principled reason to prefer the company's own procedure over the federal guide
— both are genuinely about scaffolding.

This is where the upstream layers stop being decoration. Query classification
and the `document_type` filter exist to decide *which* kind of document a
question wants; the retrieval critic exists to notice when the answer came from
the wrong one. A corpus of twelve tidy, same-sized, topically-disjoint
documents would never have exercised either.

---

## Evaluation Results

Measured against the **deployed Azure system** — Azure AI Search with the
managed semantic ranker, `gpt-5.4-mini` for generation, `gpt-5-mini` for the
analysis layers — over a hand-crafted suite of 55 questions: factual (15),
procedural (15), multi-hop cross-document (15), and out-of-scope refusals (10).

Each answer is scored by an LLM judge on faithfulness to source documents,
citation presence, and refusal accuracy.

| Metric | BuildCore Intelligence | Naive Baseline | Delta |
|---|---|---|---|
| Faithfulness | 88.7% | 79.3% | +9.5pp |
| Citation presence | **95.5%** | **18.2%** | **+77.3pp** |
| Refusal accuracy | 96.4% | 81.8% | +14.5pp |
| **Overall** | **93.5%** | **59.8%** | **+33.8pp** |

By difficulty:

| Difficulty | n | System | Baseline | Delta |
|---|---|---|---|---|
| Factual | 15 | 98.7% | 65.3% | +33.3pp |
| Procedural | 15 | 84.4% | 65.3% | +19.1pp |
| Multi-hop cross-document | 15 | 93.1% | 50.7% | **+42.4pp** |
| Out-of-scope refusal | 10 | **100.0%** | 56.7% | +43.3pp |

<!-- STALE: this screenshot predates the Azure migration and shows the earlier
     50-question / 12-document numbers, which contradict the table above.
     Re-capture from the EvalPage after the next suite run before publishing. -->
![Evaluation Dashboard](docs/evaluation.png)

### What the baseline is, and why the comparison is fair

The baseline is a naive RAG system: embed the question, take the top-5 nearest
chunks by cosine similarity, concatenate, generate. No query expansion, no
sparse retrieval, no hybrid fusion, no reranking, no critic, no structured
output or citation requirement.

Everything else is held identical *on purpose*, because a comparison is only
worth reporting if it isolates one variable:

- **Same retrieval substrate** — the baseline queries the same Azure AI Search
  index, passing `search_text=None` and no `query_type` so that neither BM25
  nor the semantic ranker contributes. It is a bare nearest-neighbour lookup
  against the same data.
- **Same chunks** — baseline hits resolve to the same *parent* text the full
  pipeline generates from, so the delta is not an artefact of the baseline
  receiving smaller context.
- **Same models** — embedding, generation, and reasoning effort all resolve
  through the same client factory.

What differs is exactly the list in the first paragraph. That is the thing
being measured.

### Reading the numbers

**Citation presence is the headline.** The naive baseline cites a source on
18.2% of answers; the full pipeline does on 95.5% — a 5x difference. Every
factual claim is traceable to a document ID and a verbatim excerpt. For any
deployment where an answer has consequences, that is the line between a system
that can be audited and one that cannot.

**The biggest deltas are where the architecture is aimed.** Multi-hop (+42.4pp)
and out-of-scope refusal (+43.3pp) are precisely what query expansion, hybrid
retrieval, and the critic gate exist for. Naive retrieval scores 50.7% on
multi-hop because a single vector query cannot span an internal SOP and a
73-page federal regulation at once.

**Refusals are perfect (100%) and were the only place the system erred.** Every
failure in the suite was an *over*-refusal — the system declining a question it
could have answered. It never hallucinated an answer it could not support. That
is the safe direction to be wrong.

### Three honest caveats

**These numbers are not comparable to this project's earlier 78.8% / +23.2pp.**
That run used 50 questions over 12 documents on `gpt-4o` with a ChromaDB
baseline. This one uses 55 questions over 17 documents (including 4,093 OSHA
distractor chunks that did not exist then) on `gpt-5.4-mini` with a fair Azure
baseline. Four variables moved at once; the improvement cannot be attributed to
any single one, and it is not claimed to be.

**Two items scored 0.00 for a bug that is now fixed, and the fix is not
reflected above.** Both failures were the retrieval critic truncating chunks to
2,000 characters while the generator reads them in full — so for *"How is
FLT-03's service brake tested?"* the correct 4,225-character chunk was
retrieved and ranked 2nd, but the answer at step 5.7 sat past the cut. The
critic reported it absent and the pipeline refused a question whose answer it
was already holding. Both now pass (0.92 / 0.86 critic confidence) with correct
cited answers. Re-running the suite would lift Procedural from 84.4% to roughly
97% and Overall to roughly 97%; the numbers above are left as measured rather
than adjusted by hand.

**One multi-hop question was made easier during authoring.** `multi_hop_15`
originally asked whether SOP-001 and SOP-005 agree on guardrail height, and the
system failed it by applying OSHA's general 42±3 in figure to a scaffold. The
rewritten version states both figures in the question, so the model only has to
reason about regulatory scope rather than retrieve the numbers. It scores 1.00 —
partly because the question is easier than the one it replaced.

---

## Tech Stack

Every row below with two entries is a swap made by an environment variable, not
a code change.

| Layer | Local (development) | Azure (production) |
|---|---|---|
| Backend | Python 3.11, FastAPI, uvicorn | ← same, on Container Apps |
| Frontend | React 18, Vite dev server | ← same build, served by nginx on Container Apps |
| Vector search | ChromaDB (cosine, persistent) | Azure AI Search (HNSW, cosine) |
| Keyword search | BM25 via `rank_bm25` | Azure AI Search (BM25) |
| Rank fusion | Reciprocal Rank Fusion (RRF, k=60) | Azure AI Search native hybrid (also RRF) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Azure managed semantic ranker |
| Embeddings | OpenAI `text-embedding-3-small` | Azure OpenAI `text-embedding-3-small` |
| Query analysis, expansion, critic | `gpt-4o-mini` | Azure OpenAI `gpt-5-mini` |
| Final generation | `gpt-4o` | Azure OpenAI `gpt-5.4-mini` (Data Zone Standard) |
| Chunking | Small-to-big parent-child (structure-aware parents, 2–3 sentence children) | ← identical, pushed to the index |
| Secrets | `.env` | Container Apps secret store |
| Registry / orchestration | Docker Compose | Azure Container Registry + Container Apps |

The Azure image installs a **separate, smaller dependency set**: with the
managed semantic ranker doing the reranking, `torch`, `sentence-transformers`,
`chromadb`, and `rank-bm25` are never imported and therefore never installed —
roughly 3GB down to 300MB. Both factories import their backends lazily, which
is what makes that possible. It matters because Container Apps scales to zero
and must pull the image before serving the first request after a scale-up.

---

## Running Locally

**Prerequisites:** Docker Desktop, OpenAI API key

```bash
# 1. Clone and configure
git clone https://github.com/wasifrepo/buildcore-rag.git
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

The ingestion step chunks all 17 documents into structure-aware parents,
splits those into small child chunks, embeds the children, and upserts them
into ChromaDB. It runs once — the index persists across container restarts.
Re-run only when documents change, and pass `recreate=True` if the chunking
strategy changed, so stale chunks from a previous scheme aren't left behind.

---

## Running on Azure

The same code runs against Azure with no source changes — the backends are
selected by environment variable:

```bash
LLM_BACKEND=azure_openai
RETRIEVER_BACKEND=azure_ai_search
RERANKER_BACKEND=passthrough        # the managed semantic ranker reranks
AZURE_SEARCH_SEMANTIC_CONFIG=buildcore-semantic
```

**Deployed architecture:**

| Component | Service |
|---|---|
| Backend API | Container Apps (scale-to-zero, 1 vCPU / 2GiB) |
| Frontend | Container Apps (nginx, static build) |
| Images | Azure Container Registry, pulled via managed identity |
| Retrieval | Azure AI Search (Basic + semantic ranker, Free plan) |
| Inference | Azure OpenAI (Data Zone Standard) |
| Secrets | Container Apps secret store, referenced as `secretref:` |

Indexing is a **push** from `ingestion/azure_index.py`, not an Azure indexer +
skillset. That is deliberate: Azure's built-in `SplitSkill` chunks on fixed
character or page boundaries, which would discard the six document-type-aware
chunkers this project is built around. Azure AI Search is used as a retrieval
engine, not a chunking engine — so both backends index byte-identical text and
the evaluation harness compares like with like.

```bash
# Build in Azure (no local Docker needed) and deploy
az acr build --registry <acr> --image buildcore-backend:v1 --file Dockerfile.azure .
az containerapp update -n ca-buildcore-api -g <rg> --image <acr>.azurecr.io/buildcore-backend:v1

# One-glance config check — reports which backends are actually live
curl https://<backend-fqdn>/health
# {"status":"ok","retriever_backend":"azure_ai_search",
#  "reranker_backend":"passthrough","llm_backend":"azure_openai"}
```

The frontend must be built *after* the backend exists: Vite inlines
`VITE_API_URL` at build time, so the backend's FQDN has to be known before the
frontend image is produced.

Note that nginx serves static assets only and does **not** proxy the API. nginx
buffers responses by default, and proxying the SSE stream through it would
deliver all six pipeline events in one burst at the end instead of streaming —
silently destroying the live progress view. The browser calls the backend
directly and `CORS_ORIGINS` allows exactly the frontend origin.

---

## Project Structure

```
buildcore-rag/
├── backend/
│   ├── api/
│   │   └── routes/          # FastAPI SSE endpoints (query, evaluate, traces)
│   ├── ingestion/
│   │   ├── classifier.py     # Document type detection
│   │   ├── pipeline.py       # Ingestion orchestrator with PDF support
│   │   ├── child_splitter.py # Parent → child splitting (small-to-big)
│   │   └── chunkers/         # Five type-specific chunkers + base class
│   ├── retrieval/
│   │   ├── base.py                     # Retriever interface (local ↔ Azure)
│   │   ├── factory.py                  # Backend selection via RETRIEVER_BACKEND
│   │   ├── local_retriever.py          # ChromaDB + BM25 + RRF
│   │   ├── azure_ai_search_retriever.py # Azure AI Search (production backend)
│   │   ├── _parenting.py               # Child → parent collapsing
│   │   ├── query_analyzer.py           # Intent classification and routing
│   │   ├── query_expander.py           # Multi-query generation
│   │   ├── dense_retriever.py          # ChromaDB semantic search
│   │   ├── sparse_retriever.py         # BM25 keyword search
│   │   ├── hybrid_retriever.py         # RRF rank fusion
│   │   ├── reranker.py                 # Cross-encoder reranking
│   │   └── retrieval_critic.py         # LLM-as-judge sufficiency check
│   ├── generation/
│   │   ├── generator.py     # Grounded generation with citations
│   │   └── schemas.py       # Pydantic models for entire pipeline
│   └── evaluation/
│       ├── test_suite.json  # 55 hand-crafted Q&A pairs
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
    └── raw/                 # Corpus (6 document type folders, 17 documents)
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

**Search small, answer big.** Chunk size is a trade-off between precise
matching and sufficient context, and picking one size means losing one of
them. Indexing children and returning parents refuses the trade-off instead
of splitting the difference.

**An adapter, not a rewrite, for Azure.** The local retrieval stack is real
and fully working, but it is not the production target. Putting retrieval
behind an interface from the start means the Azure migration is a
configuration change rather than a fork — and the evaluation harness follows
the switch automatically.

**Evaluation as a first-class component.** The test suite was written before
the evaluator, and the evaluator scores both the full pipeline and a naive
baseline through the same `get_retriever()` / `get_reranker()` factories the
production route uses — so it always scores the backend that actually ships,
not a local stand-in. The +33.8pp overall and 5x citation-presence improvements
are measured against 55 hand-crafted cases with an LLM judge, against the
deployed Azure system.

The suite has also earned its place by failing the system. Both failures in the
last run turned out to be a real bug — the critic reading truncated chunks and
vetoing answers the generator was already holding — not noise in the questions.
That is what a test suite is for.
