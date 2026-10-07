"""Dense retrieval facade.

PostgreSQL uses pgvector ANN/SQL similarity. SQLite uses a small-corpus
brute-force fallback for offline development and tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from sqlalchemy.orm import Session
from app.models import Chunk
from app.retrieval.backends.sqlite import search as sqlite_search
import numpy as np

def cosine_similarity(a, b) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(va, vb) / (na * nb))

@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def dense_search(db: Session, query: str, top_k: int = 15, user_id: str | None = None) -> List[ScoredChunk]:
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        from app.retrieval.backends.postgres_pgvector import search
        results = search(db, query, top_k, user_id)
    else:
        results = sqlite_search(db, query, top_k, user_id)
    return [ScoredChunk(chunk=c, score=s) for c, s in results]
