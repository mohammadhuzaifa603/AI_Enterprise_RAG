from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class Evidence:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    text: str
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    chunk_type: str = "text"

@dataclass
class EvidenceDecision:
    sufficient: bool
    confidence: float
    missing_information: Optional[str] = None
    contradiction: bool = False
