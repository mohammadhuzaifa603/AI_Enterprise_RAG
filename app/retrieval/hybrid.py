"""Hybrid dense + BM25 retrieval with permission filtering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sqlalchemy.orm import Session

from app.config import settings
from app.retrieval.bm25 import bm25_search
from app.retrieval.dense import dense_search


@dataclass
class HybridCandidate:
    chunk: object
    dense_score: float
    bm25_score: float
    hybrid_score: float


def _rrf_fusion(
    dense_results,
    bm25_results,
    k: int = 60,
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion.

    RRF combines ranked lists without requiring dense and BM25
    scores to be on the same numerical scale.

    score = 1 / (k + rank)
    """

    fused: Dict[str, float] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk.id

        fused[chunk_id] = fused.get(chunk_id, 0.0) + (
            settings.dense_weight * (1.0 / (k + rank))
        )

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result.chunk.id

        fused[chunk_id] = fused.get(chunk_id, 0.0) + (
            settings.bm25_weight * (1.0 / (k + rank))
        )

    return fused


def hybrid_search(
    db: Session,
    query: str,
    top_k: int | None = None,
    user_id: str | None = None,
) -> List[HybridCandidate]:

    top_k = top_k or settings.hybrid_top_k

    # ---------------------------------------------------------
    # 1. Retrieve independently
    # ---------------------------------------------------------

    dense_results = dense_search(
        db,
        query,
        settings.dense_top_k,
        user_id,
    )

    bm25_results = bm25_search(
        db,
        query,
        settings.bm25_top_k,
        user_id,
    )

    # ---------------------------------------------------------
    # 2. Build raw score maps for observability
    # ---------------------------------------------------------

    dense_raw = {
        result.chunk.id: result.score
        for result in dense_results
    }

    bm25_raw = {
        result.chunk.id: result.score
        for result in bm25_results
    }

    # ---------------------------------------------------------
    # 3. Reciprocal Rank Fusion
    # ---------------------------------------------------------

    hybrid_scores = _rrf_fusion(
        dense_results,
        bm25_results,
    )

    # ---------------------------------------------------------
    # 4. Resolve chunks
    # ---------------------------------------------------------

    by_id = {
        result.chunk.id: result.chunk
        for result in (
            dense_results + bm25_results
        )
    }

    # ---------------------------------------------------------
    # 5. Build candidates
    # ---------------------------------------------------------

    candidates = []

    for chunk_id, hybrid_score in hybrid_scores.items():

        candidates.append(
            HybridCandidate(
                chunk=by_id[chunk_id],
                dense_score=dense_raw.get(
                    chunk_id,
                    0.0,
                ),
                bm25_score=bm25_raw.get(
                    chunk_id,
                    0.0,
                ),
                hybrid_score=hybrid_score,
            )
        )

    # ---------------------------------------------------------
    # 6. Sort by fused relevance
    # ---------------------------------------------------------

    candidates.sort(
        key=lambda candidate: candidate.hybrid_score,
        reverse=True,
    )

    return candidates[:top_k]