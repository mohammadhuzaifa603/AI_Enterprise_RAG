"""Retrieval backend contract."""
from __future__ import annotations
from typing import Protocol
from sqlalchemy.orm import Session

class DenseBackend(Protocol):
    def search(self, db: Session, query: str, top_k: int): ...
