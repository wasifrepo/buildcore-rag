"""Query analyser for the BuildCore RAG pipeline.

Receives a raw user query string and returns a :class:`~generation.schemas.QueryAnalysis`
Pydantic model that drives every subsequent retrieval decision.

Classification taxonomy
-----------------------
``factual``
    The user wants a specific fact, value, name, date, or definition that
    lives in a single document.  Example: *"What is the notice period in the
    Harrington contract?"*

``procedural``
    The user wants step-by-step instructions or a process explanation.
    Example: *"How do I perform a pre-start check on the Denyo generator?"*

``cross_document``
    Answering correctly requires information from two or more documents of
    the same or different types.  Example: *"Do any of the subcontractor
    contracts require the same insurance that the site safety checklist
    mandates?"*

``ambiguous``
    The query is too vague, underspecified, or could match multiple intents.
    The retrieval strategy should cast a wide net and the answer should
    surface clarifying context.  Example: *"Tell me about the incident."*

``out_of_scope``
    The query asks for information that is definitively not present in the
    BuildCore corpus — for example, federal OSHA regulations, general
    construction law, or anything unrelated to BuildCore's own documents.
    The pipeline will return a refusal rather than hallucinating an answer.

BuildCore corpus summary (for routing decisions)
-------------------------------------------------
The corpus contains six document categories:

* **Safety SOPs** — internal procedures: chemical spill response, working at
  height, hot-works permit, confined space entry, electrical isolation.
* **Contracts** — subcontractor services agreements for Apex Electrical
  (SC-2024-038) and Harrington Scaffolding (SC-2024-041), including schedules
  for scope, price, and general conditions.
* **Incident emails** — three incident threads: INC-2024-007 (forklift near-miss
  at Commerce Drive site, 5 February 2024), INC-2024-002 (laceration LTI — Sam
  Osei, Zone 3 Warehouse, 14 February 2024), INC-2024-009 (epoxy chemical spill,
  Zone 3, 22 February 2024).
* **Maintenance manuals** — Toyota 8FGF25 forklift (MAINT-FLT-03) and
  Denyo DCA-45SPK3 generator (MAINT-GEN-01) service and operating procedures.
* **Compliance checklists** — SSIC-001 daily site safety inspection checklist
  and SC-PMCL-001 subcontractor pre-mobilisation compliance checklist.
* **OSHA regulatory documents** — OSHA2236 (Materials Handling and Storage),
  OSHA3071 (Job Hazard Analysis), OSHA3146 (Fall Protection in Construction),
  OSHA3150 (Control of Hazardous Energy / Lockout-Tagout), OSHA3903 (Quick
  Start Guide to Safety Programs).

Any query that requires knowledge outside this corpus should be classified
``out_of_scope``. Queries about OSHA requirements, regulatory standards, or
any of the specific topics covered by the OSHA documents above must NOT be
classified as out_of_scope — those documents are part of the corpus.
"""

import os

from openai import OpenAI

from generation.schemas import QueryAnalysis

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the query analyser for BuildCore RAG, an enterprise retrieval system
for BuildCore Operations — a mid-size construction and facilities management
company.

Your job is to classify an incoming user query and return a structured
QueryAnalysis object that drives the retrieval pipeline downstream.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUILDCORE CORPUS — what documents exist
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SAFETY SOPs (5 documents)
   Internal standard operating procedures covering:
   - Chemical spill response and cleanup
   - Working at height / fall protection
   - Hot-works permit and fire watch
   - Confined space entry procedures
   - Electrical isolation (lockout / tagout)

2. SUBCONTRACTOR CONTRACTS (2 documents)
   - SC-2024-038: Apex Electrical Services — electrical fit-out scope,
     lump-sum price schedule, general conditions (payment, variations,
     insurance, termination)
   - SC-2024-041: Harrington Scaffolding Solutions — scaffolding scope,
     schedule of rates, 6-clause general conditions

3. INCIDENT EMAILS (3 threads)
   - INC-2024-007: Forklift near-miss at Commerce Drive site
     (5 February 2024).
   - INC-2024-002: Laceration LTI — Sam Osei, Zone 3 Warehouse
     (14 February 2024).
   - INC-2024-009: Epoxy chemical spill, Zone 3
     (22 February 2024).

4. MAINTENANCE MANUALS (2 documents)
   - MAINT-FLT-03: Toyota 8FGF25 forklift — pre-operation checks,
     operating procedures, refuelling, maintenance schedule, emergency
     procedures
   - MAINT-GEN-01: Denyo DCA-45SPK3 generator — pre-start checks,
     start/stop procedures, load connection, shutdown, fault indicators

5. COMPLIANCE CHECKLISTS (2 documents)
   - SSIC-001: Daily site safety inspection checklist (sections: site
     access, housekeeping, fall protection, scaffolding, plant &
     equipment, PPE, emergency preparedness)
   - SC-PMCL-001: Subcontractor pre-mobilisation compliance checklist
     (sections: legal & insurance, safety documentation, worker
     credentials, site induction, plant & equipment)

6. OSHA REGULATORY DOCUMENTS (5 documents)
   - OSHA2236: Materials Handling and Storage
   - OSHA3071: Job Hazard Analysis
   - OSHA3146: Fall Protection in Construction
   - OSHA3150: Control of Hazardous Energy (Lockout/Tagout)
   - OSHA3903: Quick Start Guide to Safety Programs
   NOTE: Queries about OSHA requirements, regulatory standards, or any of
   these specific topics must be classified as factual or procedural —
   NOT out_of_scope — because these documents are part of the corpus.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSIFICATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

query_type — choose exactly one:

  factual         The answer is a specific fact, value, name, date, clause, or
                  definition from a single document. No multi-step reasoning
                  required.
                  Examples: "What is the contract sum for Apex Electrical?",
                  "Who sent the first email about the chemical spill?"

  procedural      The answer is a sequence of steps, a process, or a how-to
                  drawn primarily from one document section.
                  Examples: "How do I start the Denyo generator?",
                  "What are the steps for confined space entry?"

  cross_document  Correctly answering requires synthesising information from
                  two or more documents (can be same or different types).
                  Examples: "Do both subcontractor contracts require the same
                  insurance minimums?", "Does the chemical spill SOP align
                  with the actions taken in the INC-2024-009 email thread?"

  ambiguous       The query is too vague, uses pronouns without clear
                  referents, or could plausibly match multiple intents. Cast a
                  wide retrieval net; surface clarifying context in the answer.
                  Examples: "What happened?", "Tell me about the incident."

  out_of_scope    The query requires knowledge that is definitively not present
                  in the BuildCore corpus. This includes: Australian federal
                  or state legislation (WHS Act, Fair Work Act), OSHA
                  regulations, general construction industry standards not
                  referenced in these documents, financial market data,
                  anything unrelated to BuildCore's own operations.
                  Return a high-confidence out_of_scope rather than guessing.

retrieval_strategy — brief instruction for the hybrid retriever. Examples:
  "Dense retrieval on contract documents; filter by document_type=contract"
  "Dense + sparse retrieval across all document types; rerank for relevance"
  "Dense retrieval on SOP and email documents; multi-hop synthesis required"
  "Broad dense retrieval; surface diverse chunks for ambiguity resolution"
  "No retrieval required — query is out of scope"

requires_multi_hop — true only when the answer requires reasoning across
  two or more distinct document chunks or document IDs.

confidence — your confidence (0.0–1.0) that you have correctly classified
  this query. Use < 0.5 for genuinely ambiguous cases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Never classify a query as out_of_scope just because it is difficult.
  Only use out_of_scope when the required information genuinely does not
  exist anywhere in the corpus described above.
- When uncertain between factual and procedural, prefer procedural if the
  user's phrasing implies they want to *do* something.
- When uncertain between cross_document and factual/procedural, prefer
  cross_document if synthesising across documents would produce a
  materially better answer.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_query(query: str) -> QueryAnalysis:
    """Classify a raw user query and return a structured QueryAnalysis.

    Calls the OpenAI Chat Completions API with structured output mode
    (``response_format`` set to the :class:`~generation.schemas.QueryAnalysis`
    Pydantic schema) so the model is constrained to return a valid, fully
    populated object with no post-processing required.

    The model used is ``gpt-4o-mini`` (configured via the ``ANALYSIS_MODEL``
    environment variable), which is fast and cost-efficient for this
    lightweight routing task.

    Args:
        query: Raw user query string exactly as received from the frontend.

    Returns:
        A fully populated :class:`~generation.schemas.QueryAnalysis` instance
        whose ``query_type``, ``intent_summary``, ``retrieval_strategy``,
        ``requires_multi_hop``, and ``confidence`` fields drive all subsequent
        pipeline steps.

    Raises:
        openai.OpenAIError: On any API-level failure (network, auth, rate
            limit).  The caller (pipeline orchestrator) is responsible for
            retry logic.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("ANALYSIS_MODEL", "gpt-4o-mini")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        response_format=QueryAnalysis,
    )

    return completion.choices[0].message.parsed
