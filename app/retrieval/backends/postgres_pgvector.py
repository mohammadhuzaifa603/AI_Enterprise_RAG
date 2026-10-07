"""PostgreSQL + pgvector dense retrieval backend.

This is the production-style path. Similarity is executed by PostgreSQL
using the vector operator instead of loading the entire corpus into Python.
"""
from __future__ import annotations
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.ingestion.embeddings import embed_text
from app.models import Chunk, Document, DocumentPermission, User
from app.retrieval.filters import DEFAULT_LOCAL_USER, get_or_create_local_user


def search(db: Session, query: str, top_k: int = 15, user_id: str | None = None):
    if db.bind is None or db.bind.dialect.name != "postgresql":
        raise RuntimeError("Postgres pgvector backend requires a PostgreSQL session.")
    user = db.get(User, user_id) if user_id else get_or_create_local_user(db)
    qv = embed_text(query)
    distance = Chunk.embedding.cosine_distance(qv)
    stmt = select(Chunk, (1.0 - distance).label("score")).join(Document, Chunk.document_id == Document.id)
    if user.role != "ADMIN":
        allowed = select(DocumentPermission.document_id).where(
            DocumentPermission.user_id == user.id,
            DocumentPermission.permission == "READ",
        )
        stmt = stmt.where((Document.owner_id == user.id) | Chunk.document_id.in_(allowed))
    stmt = stmt.where(Chunk.embedding.is_not(None)).order_by(distance).limit(top_k)
    return [(chunk, float(score)) for chunk, score in db.execute(stmt).all()]
