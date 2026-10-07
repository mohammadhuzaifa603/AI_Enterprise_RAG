"""
Embedding backends.

Two implementations behind one interface:

* HashingEmbedder (default): a dependency-free, deterministic embedding
  built from word-hashing + TF weighting, L2-normalized. It needs no
  network access and no GPU, which makes the whole platform runnable
  offline / in sandboxed environments. It captures lexical overlap well
  enough to demonstrate the full retrieval pipeline end-to-end, but it
  is NOT a substitute for a trained semantic embedding model.

* SentenceTransformerEmbedder (optional, used automatically when
  `EMBEDDING_BACKEND=sentence-transformers` and the package + model
  weights are available): a real dense embedding model via
  sentence-transformers. If the model cannot be loaded (e.g. no
  internet access to huggingface.co), the platform logs a warning and
  transparently falls back to the hashing embedder rather than crashing.

This mirrors how a real enterprise system is often built: a cheap local
fallback for offline/dev, a stronger model in environments that can
reach the model hub.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache
from typing import List, Protocol

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class HashingEmbedder:
    """Deterministic, dependency-free embedding via feature hashing."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = _tokenize(text)
        if not tokens:
            return vec.tolist()

        # Term frequency
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        for tok, count in tf.items():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            weight = 1.0 + math.log(count)
            vec[idx] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """Real dense embedding model, loaded lazily."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # may raise ImportError

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the configured embedder, falling back to hashing on failure."""
    if settings.embedding_backend == "sentence-transformers":
        try:
            embedder = SentenceTransformerEmbedder(settings.embedding_model_name)
            logger.info("Loaded sentence-transformers embedder: %s", settings.embedding_model_name)
            return embedder
        except Exception as exc:
            logger.warning(
                "Falling back to HashingEmbedder: could not load sentence-transformers model (%s). "
                "This is expected in offline/sandboxed environments without access to huggingface.co.",
                exc,
            )
    return HashingEmbedder(dim=settings.embedding_dim)


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedder().embed(texts)


def embed_text(text: str) -> List[float]:
    return get_embedder().embed([text])[0]
