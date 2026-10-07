"""Transparent confidence scoring; explicitly heuristic, not probability."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from app.config import settings
from app.document_ai.validation import REQUIRED_FIELDS

ALL_TRACKED_FIELDS = ["vendor_name", "invoice_number", "invoice_date", "subtotal", "discount", "tax", "shipping", "fees", "total", "currency"]

def compute_confidence(fields: Dict[str, Any], valid: bool, errors: List[str], warnings: List[str], ocr_ratio: float = 0.0) -> Tuple[float, Dict[str, float], str]:
    field_scores = {f: (1.0 if fields.get(f) not in (None, "", []) else 0.0) for f in ALL_TRACKED_FIELDS}
    filled = sum(field_scores.values())
    completeness = filled / len(ALL_TRACKED_FIELDS)
    validation_score = 1.0 if valid else max(0.0, 1.0 - 0.35 * len(errors))
    validation_score = max(0.0, validation_score - 0.05 * len(warnings))
    ocr_quality = max(0.0, 1.0 - (0.3 * ocr_ratio))
    required_present = sum(field_scores[f] for f in REQUIRED_FIELDS)
    structured_validity = required_present / len(REQUIRED_FIELDS)
    score = round(max(0.0, min(1.0, 0.30 * completeness + 0.35 * validation_score + 0.15 * ocr_quality + 0.20 * structured_validity)), 3)
    breakdown = {"overall": score, "field_completeness": round(completeness, 3), "validation_success": round(validation_score, 3), "ocr_quality": round(ocr_quality, 3), "structured_validity": round(structured_validity, 3)}
    breakdown.update({f"field:{k}": v for k, v in field_scores.items()})
    if not fields or not filled:
        status = "FAILED"
    elif not valid:
        status = "NEEDS_REVIEW"
    elif score >= settings.auto_approve_confidence:
        status = "AUTO_APPROVED"
    elif score >= settings.needs_review_confidence:
        status = "NEEDS_REVIEW"
    else:
        status = "FAILED"
    return score, breakdown, status
