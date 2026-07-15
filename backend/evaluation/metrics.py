"""Scoring functions for evaluating BuildCore RAG pipeline outputs.

Three metrics are implemented:

``score_faithfulness``
    LLM-as-judge (GPT-4o-mini) that reads the system answer, the expected
    answer, and the hand-written evaluation rubric from the test suite, then
    returns a 0.0–1.0 score.  Uses structured output so the score is always
    a valid float.  Handles out-of-scope refusal cases explicitly: if the
    expected answer begins with "REFUSAL", a correct refusal by the system
    scores 1.0; an attempted answer scores 0.0.

``score_refusal_accuracy``
    Rule-based.  For ``out_of_scope`` questions the system must refuse
    (``refused=True``).  For all other difficulties the system must answer
    (``refused=False``).  Returns 1.0 or 0.0 — no partial credit.

``score_citation_presence``
    Fraction of the test-suite's ``source_documents`` list that appear
    (by ``document_id``) in the answer's ``citations`` list.  Returns 1.0
    when ``source_documents`` is empty (out-of-scope questions have no
    required citations).
"""

import os

from common.llm_client import (
    get_analysis_model,
    get_llm_client,
    reasoning_extra_body,
)
from pydantic import BaseModel, Field

from generation.schemas import GeneratedAnswer


# ---------------------------------------------------------------------------
# Pydantic model for the LLM judge output
# ---------------------------------------------------------------------------


class FaithfulnessScore(BaseModel):
    """Structured output produced by the LLM faithfulness judge."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Faithfulness score from 0.0 (completely wrong) to 1.0 (fully correct)",
    )
    reasoning: str = Field(
        description="One or two sentences explaining the score based on the rubric"
    )


# ---------------------------------------------------------------------------
# System prompt for the faithfulness judge
# ---------------------------------------------------------------------------

_FAITHFULNESS_JUDGE_PROMPT = """\
You are an evaluation judge for BuildCore RAG, an enterprise question-answering
system over construction and facilities management documents.

Your task: given a system-generated answer, a reference (expected) answer, and
a scoring rubric, assign a faithfulness score between 0.0 and 1.0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.0  All key facts in the rubric are present and correct.
0.8  Most key facts correct; one minor omission or imprecise detail.
0.6  Core answer is directionally correct but missing important specifics.
0.4  Partial answer — correct in some areas but wrong or absent in others.
0.2  Substantially incorrect or highly incomplete.
0.0  Wrong answer, hallucinated facts, or refused to answer when an answer
     was required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIAL RULE — OUT-OF-SCOPE / REFUSAL CASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If the reference answer begins with "REFUSAL", the correct system behaviour
is to refuse to answer.

  • System answer is a clear refusal → score 1.0
  • System answer attempts to answer the question → score 0.0
  • System answer partially refuses but also provides substantive content
    from outside the corpus → score 0.2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE ONLY THE RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Judge solely against the rubric (evaluation_notes) and reference answer.
Do not penalise for style, length, or extra correct details beyond the rubric.
Do not award marks for facts not in the rubric even if they are true.
"""


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------


def score_faithfulness(
    answer: str,
    expected: str,
    evaluation_notes: str,
) -> FaithfulnessScore:
    """Score how faithfully a system answer matches the expected answer.

    Calls GPT-4o-mini as an LLM judge with structured output, using the
    hand-written ``evaluation_notes`` from the test suite as the rubric.
    Returns a :class:`FaithfulnessScore` with both the numeric score and
    the judge's one-sentence reasoning.

    Args:
        answer: The system-generated answer string (plain text or the
            ``answer`` field extracted from a :class:`GeneratedAnswer`).
        expected: The reference answer from the test suite.
        evaluation_notes: The hand-written rubric specifying what a correct
            answer must include.

    Returns:
        A :class:`FaithfulnessScore` with ``score`` in ``[0.0, 1.0]`` and
        a brief ``reasoning`` string.
    """
    client = get_llm_client()
    model = get_analysis_model()

    user_message = (
        f"REFERENCE ANSWER:\n{expected}\n\n"
        f"RUBRIC (evaluation_notes):\n{evaluation_notes}\n\n"
        f"SYSTEM ANSWER:\n{answer}\n\n"
        "Score the system answer against the rubric and reference answer."
    )

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _FAITHFULNESS_JUDGE_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=FaithfulnessScore,
        extra_body=reasoning_extra_body("analysis"),
    )
    return completion.choices[0].message.parsed


def score_refusal_accuracy(
    answer: GeneratedAnswer,
    expected_difficulty: str,
) -> float:
    """Score whether the system correctly refused or answered the question.

    Rule-based (no LLM call).  Out-of-scope questions must produce a refusal;
    all other difficulties must produce an answer.  Returns 1.0 or 0.0.

    Args:
        answer: The :class:`~generation.schemas.GeneratedAnswer` produced by
            the pipeline.
        expected_difficulty: The ``difficulty`` field from the test suite item
            (``"factual"``, ``"procedural"``, ``"multi_hop"``, or
            ``"out_of_scope"``).

    Returns:
        ``1.0`` if the refusal behaviour is correct, ``0.0`` otherwise.
    """
    should_refuse = expected_difficulty == "out_of_scope"
    did_refuse = answer.refused

    return 1.0 if should_refuse == did_refuse else 0.0


def score_citation_presence(
    answer: GeneratedAnswer,
    source_documents: list[str],
) -> float:
    """Score what fraction of expected source documents appear in citations.

    Computes the proportion of the test suite's ``source_documents`` list
    that appear (by ``document_id``) in the answer's ``citations``.  If
    ``source_documents`` is empty (as with out-of-scope questions that have
    no required citations), returns 1.0 unconditionally.

    Args:
        answer: The :class:`~generation.schemas.GeneratedAnswer` produced by
            the pipeline.
        source_documents: List of document ID strings from the test suite item
            specifying which documents must be cited for a complete answer.

    Returns:
        A float in ``[0.0, 1.0]``.  ``1.0`` means all expected documents
        were cited; ``0.0`` means none were.
    """
    if not source_documents:
        return 1.0

    cited_ids: set[str] = {c.document_id for c in answer.citations}
    matched = sum(1 for doc_id in source_documents if doc_id in cited_ids)
    return matched / len(source_documents)
