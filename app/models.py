"""Database models for the enterprise RAG platform.

The model is intentionally portable:
- SQLite stores vectors as JSON for zero-setup/offline development.
- PostgreSQL uses a real pgvector VECTOR column and can use ANN indexes.

The database also keeps provenance, permissions, evidence, claims, reviews,
and audit records so an answer can be traced back to the exact source data.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import settings
from app.database import Base

try:  # Optional at import time so SQLite/offline mode remains lightweight.
    from pgvector.sqlalchemy import Vector as PGVector
except ImportError:  # pragma: no cover
    PGVector = None


def _uuid() -> str:
    return uuid.uuid4().hex[:24]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JSONType(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else json.dumps(value, default=str)

    def process_result_value(self, value, dialect):
        return None if value is None else json.loads(value)


class VectorType(TypeDecorator):
    """Real pgvector on PostgreSQL; JSON fallback on SQLite."""

    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            if PGVector is None:
                raise RuntimeError(
                    "pgvector is required for PostgreSQL. "
                    "Install the 'pgvector' package."
                )
            return dialect.type_descriptor(
                PGVector(settings.embedding_dim)
            )

        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None

        if dialect.name == "postgresql":
            return list(value)

        return json.dumps(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None

        if dialect.name == "postgresql":
            return list(value)

        return json.loads(value)

    def coerce_compared_value(self, op, value):
        return self


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(32), default="USER")  # ADMIN / USER
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    original_filename: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str] = mapped_column(String(1024))
    file_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    storage_encrypted: Mapped[bool] = mapped_column(Boolean, default=True)
    owner_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    doc_type: Mapped[str] = mapped_column(String(64), default="generic")
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    num_pages: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    pages: Mapped[List["Page"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    extractions: Mapped[List["Extraction"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    permissions: Mapped[List["DocumentPermission"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentPermission(Base):
    __tablename__ = "document_permissions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    permission: Mapped[str] = mapped_column(String(16), default="READ")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    document: Mapped["Document"] = relationship(back_populates="permissions")


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, default="")
    used_ocr: Mapped[bool] = mapped_column(Boolean, default=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    tables_json: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="pages")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    page_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pages.id"), nullable=True, index=True)
    page_number: Mapped[int] = mapped_column(Integer, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_type: Mapped[str] = mapped_column(String(32), default="text")  # text/table/invoice_field
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
    PGVector(settings.embedding_dim) if PGVector is not None else VectorType(),
    nullable=True,
)
    chunk_metadata: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    doc_type: Mapped[str] = mapped_column(String(64), default="invoice")
    fields: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    valid: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_errors: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    validation_warnings: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_breakdown: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="NEEDS_REVIEW")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    document: Mapped["Document"] = relationship(back_populates="extractions")
    reviews: Mapped[List["Review"]] = relationship(back_populates="extraction", cascade="all, delete-orphan")
    field_evidence: Mapped[List["FieldEvidence"]] = relationship(back_populates="extraction", cascade="all, delete-orphan")


class FieldEvidence(Base):
    __tablename__ = "field_evidence"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    extraction_id: Mapped[str] = mapped_column(ForeignKey("extractions.id"), index=True)
    field_name: Mapped[str] = mapped_column(String(64))
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    extraction: Mapped["Extraction"] = relationship(back_populates="field_evidence")


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    extraction_id: Mapped[str] = mapped_column(ForeignKey("extractions.id"))
    action: Mapped[str] = mapped_column(String(32))
    edited_fields: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str] = mapped_column(String(128), default="local-user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    extraction: Mapped["Extraction"] = relationship(back_populates="reviews")


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(32), default="hybrid")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    num_citations: Mapped[int] = mapped_column(Integer, default=0)
    num_verified_citations: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    citations: Mapped[List["Citation"]] = relationship(back_populates="query", cascade="all, delete-orphan")


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    query_id: Mapped[Optional[str]] = mapped_column(ForeignKey("queries.id"), nullable=True)
    agent_run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("agent_runs.id"), nullable=True, index=True)
    chunk_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    document_name: Mapped[str] = mapped_column(String(512))
    page: Mapped[int] = mapped_column(Integer)
    claim_text: Mapped[str] = mapped_column(Text)
    excerpt: Mapped[str] = mapped_column(Text, default="")
    supported: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    query: Mapped[Optional["Query"]] = relationship(back_populates="citations")
    agent_run: Mapped[Optional["AgentRun"]] = relationship(back_populates="citations")


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    succeeded: Mapped[bool] = mapped_column(Boolean, default=False)
    num_attempts: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    attempts: Mapped[List["RetrievalAttempt"]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan", order_by="RetrievalAttempt.attempt_number"
    )
    citations: Mapped[List["Citation"]] = relationship(back_populates="agent_run", cascade="all, delete-orphan")


class RetrievalAttempt(Base):
    __tablename__ = "retrieval_attempts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)
    query_used: Mapped[str] = mapped_column(Text)
    query_analysis: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    retrieved_chunk_ids: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    top_score: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_sufficient: Mapped[bool] = mapped_column(Boolean, default=False)
    evidence_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    missing_information: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decision: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    agent_run: Mapped["AgentRun"] = relationship(back_populates="attempts")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    document_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    query_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    metadata_json: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
