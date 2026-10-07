"""Offline brute-force dense retrieval backend."""
from __future__ import annotations
from typing import List
import numpy as np
from sqlalchemy.orm import Session
from app.ingestion.embeddings import embed_text
from app.models import Chunk
from app.retrieval.filters import accessible_chunk_query


def cosine_similarity(a, b) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def search(db: Session, query: str, top_k: int = 15, user_id: str | None = None):
    chunks: List[Chunk] = accessible_chunk_query(db, user_id).filter(Chunk.embedding.is_not(None)).all()
    if not chunks:
        return []
    qv = embed_text(query)
    scored = [(c, cosine_similarity(qv, c.embedding)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
