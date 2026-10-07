"""Pydantic request/response models shared across the API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
class DocumentOut(BaseModel):
    id: str
    filename: str
    original_filename: str
    doc_type: str
    status: str
    num_pages: int
    file_sha256: Optional[str] = None
    storage_encrypted: bool = False
    error_message: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentProcessResult(BaseModel):
    document_id: str
    status: str
    num_pages: int
    file_sha256: Optional[str] = None
    storage_encrypted: bool = False
    num_chunks: int
    ocr_pages: int
    tables_found: int
    extraction_id: Optional[str] = None
    error_message: Optional[str] = None


class DocumentDeleteResponse(BaseModel):
    id: str
    original_filename: str
    deleted: bool
    detail: str


# --------------------------------------------------------------------------
# Module 1 - Hybrid RAG + citations
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class CitationOut(BaseModel):
    document: str
    page: int
    chunk_id: Optional[str] = None
    excerpt: str
    claim_text: str
    supported: bool
    confidence: float
    reason: str


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    citations: List[CitationOut]
    retrieved_chunks: int
    latency_ms: float
    llm_mode: str  # "live" or "mock"


class QueryOut(BaseModel):
    query_id: str
    question: str
    answer: str
    mode: str
    num_citations: int
    num_verified_citations: int
    latency_ms: float
    created_at: datetime

    model_config = {"from_attributes": True}


# --------------------------------------------------------------------------
# Module 2 - Document intelligence
# --------------------------------------------------------------------------
class ValidationResult(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class ExtractionOut(BaseModel):
    id: str
    document_id: str
    doc_type: str
    fields: Dict[str, Any]
    valid: bool
    validation_errors: List[str]
    validation_warnings: List[str]
    confidence_score: float
    confidence_breakdown: Dict[str, float]
    status: str
    created_at: datetime
    field_evidence: List["FieldEvidenceOut"] = []

    model_config = {"from_attributes": True}


class FieldEvidenceOut(BaseModel):
    field_name: str
    page_number: Optional[int] = None
    source_text: str
    confidence: float

    model_config = {"from_attributes": True}


class ReviewRequest(BaseModel):
    action: str = Field(..., pattern="^(approve|reject|edit)$")
    edited_fields: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ReviewOut(BaseModel):
    id: str
    extraction_id: str
    action: str
    edited_fields: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# --------------------------------------------------------------------------
# Module 3 - Agentic RAG
# --------------------------------------------------------------------------
class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class RetrievalAttemptOut(BaseModel):
    attempt_number: int
    query_used: str
    query_analysis: Dict[str, Any]
    retrieved_chunk_ids: List[str]
    top_score: float
    evidence_sufficient: bool
    evidence_confidence: float
    missing_information: Optional[str]
    decision: str


class AgentRunOut(BaseModel):
    id: str
    question: str
    final_answer: Optional[str]
    succeeded: bool
    num_attempts: int
    latency_ms: float = 0.0
    attempts: List[RetrievalAttemptOut]
    citations: List[CitationOut] = []


# --------------------------------------------------------------------------
# Dashboard / evaluation
# --------------------------------------------------------------------------
class DashboardStats(BaseModel):
    documents_processed: int
    pages_indexed: int
    chunks_indexed: int
    review_queue_count: int
    queries_processed: int
    average_confidence: float
    llm_mode: str
    embedding_backend: str


class HealthOut(BaseModel):
    status: str
    database: str
    embedding_backend: str
    reranker_backend: str
    llm_mode: str
    version: str
