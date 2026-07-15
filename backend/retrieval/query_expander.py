"""Query expander for the BuildCore RAG pipeline.

Generates three rephrased variants of the original user query to improve
retrieval coverage.  The variants deliberately target different angles:

- **Vocabulary shift** — replaces domain terms with synonyms or related
  jargon likely to appear in the corpus (e.g. "hot-works" ↔ "fire permit",
  "LOTO" ↔ "lockout/tagout").
- **Specificity change** — one variant broadens the query (useful when the
  answer is spread across sections) and one narrows it (useful for pinpointing
  a precise clause or step).
- **Phrasing style** — switches between question form, keyword-list form,
  and imperative/procedural form to match different writing styles in the
  corpus.

All four queries (original + 3 variants) are later embedded and used
independently for dense retrieval; BM25 sparse retrieval uses the original
query only.  The :class:`~generation.schemas.ExpandedQueries` model is
defined in ``generation/schemas.py``.
"""

import os

from common.llm_client import (
    get_analysis_model,
    get_llm_client,
    reasoning_extra_body,
)

from generation.schemas import ExpandedQueries

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a query expansion assistant for BuildCore RAG — an enterprise
retrieval system over construction and facilities management documents.

The corpus contains six document types:
  • Safety SOPs (chemical spill, working at height, hot-works, confined
    space, electrical isolation / lockout-tagout)
  • Subcontractor contracts (Apex Plumbing & Drainage SC-2024-038, Harrington
    Electrical Services SC-2024-041) with scope schedules, price schedules,
    and general conditions
  • Incident emails (INC-2024-007 forklift near-miss, INC-2024-002 laceration
    LTI, INC-2024-009 epoxy chemical spill)
  • Maintenance manuals (Toyota 8FGF25 forklift, Denyo DCA-45SPK3 generator)
  • Compliance checklists (daily site safety inspection, subcontractor
    pre-mobilisation checklist)
  • OSHA regulatory documents (OSHA2236 materials handling, OSHA3071 job
    hazard analysis, OSHA3146 fall protection, OSHA3150 lockout/tagout,
    OSHA3903 safety programs)

Your job: given the user's original query, produce exactly 3 rephrased
variants that will retrieve MORE relevant chunks from the corpus when used
independently alongside the original query.

Rules for the variants:
1. VOCABULARY SHIFT — Replace key terms with synonyms, acronyms, or
   domain jargon that might appear in the document text.  E.g.
   "lockout/tagout" ↔ "electrical isolation", "hot-works" ↔ "fire permit
   and fire watch", "EWP" ↔ "elevated work platform".
2. SPECIFICITY — Make one variant broader (useful when the answer spans
   multiple sections) and one more specific (useful for pinpointing a
   clause, step number, or checklist item).
3. PHRASING STYLE — Vary the style: question form, keyword phrase, or
   imperative/procedural phrasing.  Match the writing styles found in
   technical procedures, contract clauses, and email correspondence.

Important constraints:
- Do NOT change the factual intent of the query.
- Do NOT invent information not implied by the original query.
- Each variant must be a standalone query — no references to "the above".
- Keep variants concise (1–2 sentences or a keyword phrase).
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_query(query: str) -> ExpandedQueries:
    """Generate three retrieval-broadening variants of the original query.

    Calls the OpenAI Chat Completions API with structured output mode so the
    response is parsed directly into an :class:`~generation.schemas.ExpandedQueries`
    instance.  The ``original`` field is populated with the raw ``query``
    argument; the ``variants`` list contains exactly three rephrased strings.

    Args:
        query: Raw user query string exactly as received from the frontend.

    Returns:
        An :class:`~generation.schemas.ExpandedQueries` instance with
        ``original`` set to ``query`` and ``variants`` containing three
        rephrased alternatives for broader retrieval coverage.

    Raises:
        openai.OpenAIError: On any API-level failure (network, auth, rate
            limit).  The caller is responsible for retry logic.
    """
    client = get_llm_client()
    model = get_analysis_model()

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    "Return an ExpandedQueries object. "
                    "Set original to the exact text above. "
                    "Set variants to exactly 3 rephrased alternatives."
                ),
            },
        ],
        response_format=ExpandedQueries,
        extra_body=reasoning_extra_body("expansion"),
    )

    result: ExpandedQueries = completion.choices[0].message.parsed
    # Guarantee the original field exactly matches the caller's query string,
    # regardless of any model paraphrasing.
    result.original = query
    return result
