from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.document_ai.confidence import compute_confidence
from app.document_ai.validation import validate_invoice_fields
from app.models import Extraction, Review
from app.schemas import ExtractionOut, FieldEvidenceOut, ReviewOut, ReviewRequest

router = APIRouter(tags=["review"])


@router.get("/review-queue", response_model=List[ExtractionOut])
def list_review_queue(db: Session = Depends(get_db)):
    extractions = (
        db.query(Extraction)
        .filter(Extraction.status.in_(["NEEDS_REVIEW", "FAILED"]))
        .order_by(Extraction.created_at.desc())
        .all()
    )
    return [_extraction_to_schema(e) for e in extractions]


@router.get("/extractions/{extraction_id}", response_model=ExtractionOut)
def get_extraction(extraction_id: str, db: Session = Depends(get_db)):
    extraction = db.get(Extraction, extraction_id)
    if extraction is None:
        raise HTTPException(status_code=404, detail="Extraction not found.")
    return _extraction_to_schema(extraction)


@router.post("/review/{extraction_id}", response_model=ReviewOut)
def submit_review(extraction_id: str, payload: ReviewRequest, db: Session = Depends(get_db)):
    extraction = db.get(Extraction, extraction_id)
    if extraction is None:
        raise HTTPException(status_code=404, detail="Extraction not found.")

    review = Review(
        extraction_id=extraction_id,
        action=payload.action,
        edited_fields=payload.edited_fields,
        notes=payload.notes,
    )
    db.add(review)

    if payload.action == "approve":
        # Human confirmed the extraction as-is; this is an explicit
        # override, so it moves out of the queue even if the original
        # score was borderline.
        extraction.status = "AUTO_APPROVED"
    elif payload.action == "reject":
        extraction.status = "FAILED"
    elif payload.action == "edit" and payload.edited_fields:
        merged = dict(extraction.fields or {})
        merged.update(payload.edited_fields)
        extraction.fields = merged
        # Re-run deterministic validation + confidence on the corrected
        # fields rather than blindly marking it approved - a bad edit
        # should still surface as NEEDS_REVIEW.
        valid, errors, warnings = validate_invoice_fields(merged)
        score, breakdown, status = compute_confidence(merged, valid, errors, warnings, ocr_ratio=0.0)
        extraction.valid = valid
        extraction.validation_errors = errors
        extraction.validation_warnings = warnings
        extraction.confidence_score = score
        extraction.confidence_breakdown = breakdown
        extraction.status = status

    db.commit()
    db.refresh(review)
    return review


def _extraction_to_schema(e: Extraction) -> ExtractionOut:
    return ExtractionOut(
        id=e.id,
        document_id=e.document_id,
        doc_type=e.doc_type,
        fields=e.fields or {},
        valid=e.valid,
        validation_errors=e.validation_errors or [],
        validation_warnings=e.validation_warnings or [],
        confidence_score=e.confidence_score,
        confidence_breakdown=e.confidence_breakdown or {},
        status=e.status,
        created_at=e.created_at,
        field_evidence=[
            FieldEvidenceOut(
                field_name=fe.field_name,
                page_number=fe.page_number,
                source_text=fe.source_text,
                confidence=fe.confidence,
            )
            for fe in (e.field_evidence or [])
        ],
    )
