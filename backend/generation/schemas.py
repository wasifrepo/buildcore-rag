from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """The six document categories present in the BuildCore corpus."""

    SAFETY_SOP = "safety_sop"
    CONTRACT = "contract"
    INCIDENT_EMAIL = "incident_email"
    MAINTENANCE_MANUAL = "maintenance_manual"
    COMPLIANCE_CHECKLIST = "compliance_checklist"
    REGULATORY_DOC = "regulatory_doc"


class QueryType(str, Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CROSS_DOCUMENT = "cross_document"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


class QueryAnalysis(BaseModel):
    query_type: QueryType
    intent_summary: str = Field(description="One sentence summary of what the user is asking")
    retrieval_strategy: str = Field(description="Which retrieval strategy to apply")
    requires_multi_hop: bool = Field(description="Whether the query requires reasoning across multiple documents")
    confidence: float = Field(ge=0.0, le=1.0)


class ExpandedQueries(BaseModel):
    original: str
    variants: list[str] = Field(description="3 rephrased variants for broader retrieval coverage")


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_type: str
    content: str
    metadata: dict
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None


class CriticVerdict(BaseModel):
    sufficient: bool = Field(description="Whether retrieved chunks are sufficient to answer the query")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Why the chunks are or are not sufficient")
    refined_query: Optional[str] = Field(
        default=None,
        description="Refined query for second retrieval pass if chunks are insufficient"
    )


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    excerpt: str = Field(description="Short excerpt from the chunk that supports the claim")


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    refused: bool = Field(default=False, description="True if query was out of scope and system declined to answer")
    refusal_reason: Optional[str] = None


class PipelineTrace(BaseModel):
    trace_id: str
    question: str
    query_analysis: QueryAnalysis
    expanded_queries: ExpandedQueries
    chunks_retrieved: list[Chunk]
    chunks_reranked: list[Chunk]
    critic_verdict: CriticVerdict
    second_pass_triggered: bool = False
    final_answer: GeneratedAnswer
    retrieval_passes: int = Field(default=1)
    total_latency_ms: float
