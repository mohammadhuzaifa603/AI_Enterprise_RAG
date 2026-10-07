from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Chunk, Document, Extraction, Page, Query
from app.schemas import DashboardStats
from app.services.llm_client import get_llm_client

router = APIRouter(tags=["evaluation"])

EVAL_RESULTS_PATH = Path(settings.upload_dir).parent / "evaluation_results.json"


@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(db: Session = Depends(get_db)):
    documents_processed = db.query(Document).filter(Document.status == "processed").count()
    pages_indexed = db.query(Page).count()
    chunks_indexed = db.query(Chunk).count()
    review_queue_count = db.query(Extraction).filter(Extraction.status.in_(["NEEDS_REVIEW", "FAILED"])).count()
    queries_processed = db.query(Query).count()
    avg_confidence = db.query(func.avg(Extraction.confidence_score)).scalar() or 0.0

    return DashboardStats(
        documents_processed=documents_processed,
        pages_indexed=pages_indexed,
        chunks_indexed=chunks_indexed,
        review_queue_count=review_queue_count,
        queries_processed=queries_processed,
        average_confidence=round(float(avg_confidence), 3),
        llm_mode=get_llm_client().mode,
        embedding_backend=settings.embedding_backend,
    )


@router.get("/evaluation")
def get_evaluation_results():
    """Returns the results of the most recent `scripts/evaluate.py` run.

    Never fabricates numbers: if the evaluation script has not been run
    yet, this returns a clear "not yet run" payload instead of made-up
    metrics.
    """
    if not EVAL_RESULTS_PATH.exists():
        return {
            "status": "not_run",
            "message": "No evaluation results yet. Run `python -m evaluation.runner` to generate real metrics.",
        }
    with EVAL_RESULTS_PATH.open() as f:
        return {"status": "ok", "results": json.load(f)}
