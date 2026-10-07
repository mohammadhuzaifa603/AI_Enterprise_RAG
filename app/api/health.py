from fastapi import APIRouter
from app import __version__
from app.config import settings, llm_is_configured
from app.database import engine

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {
        "status": "ok",
        "database": "ok",
        "database_engine": engine.dialect.name,
        "vector_backend": "pgvector" if engine.dialect.name == "postgresql" else "sqlite-python-fallback",
        "embedding_backend": settings.embedding_backend,
        "reranker_backend": settings.reranker_backend,
        "llm_mode": "live" if llm_is_configured() else "mock",
        "encrypted_storage": bool(settings.storage_encryption_key),
        "version": __version__,
    }
