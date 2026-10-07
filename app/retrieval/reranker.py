"""
Reranking.

Two implementations behind one interface, selected via
`settings.reranker_backend`:

* "lexical" (default): a dependency-free query-chunk overlap scorer
  (token Jaccard + coverage), used because this sandboxed build has no
  network access to huggingface.co to download a real cross-encoder
  checkpoint.
* "cross-encoder": a real `sentence-transformers.CrossEncoder`
  (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2), loaded lazily. Falls
  back to the lexical reranker if the model cannot be loaded.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from app.config import settings
from app.retrieval.hybrid import HybridCandidate

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass
class RerankedChunk:
    chunk: object
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0


def _lexical_score(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    c_tokens = _tokenize(text)
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = q_tokens & c_tokens
    jaccard = len(overlap) / len(q_tokens | c_tokens)
    coverage = len(overlap) / len(q_tokens)  # how much of the query is addressed
    return 0.4 * jaccard + 0.6 * coverage


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder  # may raise ImportError

    return CrossEncoder(settings.cross_encoder_model_name)


def rerank(query: str, candidates: List[HybridCandidate], top_k: int | None = None) -> List[RerankedChunk]:
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []

    if settings.reranker_backend == "cross-encoder":
        try:
            model = _get_cross_encoder()
            pairs = [(query, c.chunk.text) for c in candidates]
            raw_scores = model.predict(pairs)
            reranked = [
                RerankedChunk(chunk=c.chunk, score=float(s), dense_score=c.dense_score, bm25_score=c.bm25_score)
                for c, s in zip(candidates, raw_scores)
            ]
            reranked.sort(key=lambda r: r.score, reverse=True)
            return reranked[:top_k]
        except Exception as exc:
            logger.warning(
                "Falling back to lexical reranker: could not load cross-encoder model (%s). "
                "This is expected without access to huggingface.co.",
                exc,
            )

    reranked = [
        RerankedChunk(
            chunk=c.chunk,
            score=_lexical_score(query, c.chunk.text),
            dense_score=c.dense_score,
            bm25_score=c.bm25_score,
        )
        for c in candidates
    ]
    reranked.sort(key=lambda r: r.score, reverse=True)
    return reranked[:top_k]
