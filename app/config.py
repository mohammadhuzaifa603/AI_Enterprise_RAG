"""Central application configuration."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'app.db'}"
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: float = 30.0

    embedding_backend: str = "hashing"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    reranker_backend: str = "lexical"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    dense_top_k: int = 15
    bm25_top_k: int = 15
    hybrid_top_k: int = 10
    rerank_top_k: int = 5
    dense_weight: float = 0.5
    bm25_weight: float = 0.5

    chunk_size_chars: int = 900
    chunk_overlap_chars: int = 150

    agent_max_attempts: int = 3
    agent_sufficiency_threshold: float = 0.40

    ocr_min_chars_per_page: int = 20
    tesseract_cmd: Optional[str] = None
    auto_approve_confidence: float = 0.85
    needs_review_confidence: float = 0.5

    upload_dir: Path = BASE_DIR / "data" / "uploads"
    max_upload_bytes: int = 25 * 1024 * 1024
    max_pdf_pages: int = 100
    storage_encryption_key: Optional[str] = None
    storage_encryption_required: bool = False

    allowed_origins: str = "http://localhost:8501,http://localhost:3000"
    app_name: str = "AI Enterprise RAG Platform"
    api_base_url: str = "http://localhost:8000"

settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)

def llm_is_configured() -> bool:
    return bool(settings.llm_api_key and settings.llm_api_key.strip())

def cors_origins() -> list[str]:
    return [x.strip() for x in settings.allowed_origins.split(",") if x.strip()]
