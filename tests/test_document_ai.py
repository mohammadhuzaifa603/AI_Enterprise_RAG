from decimal import Decimal

from app.document_ai.confidence import compute_confidence
from app.document_ai.extraction import extract_invoice_fields, ground_field_evidence
from app.document_ai.validation import validate_invoice_fields
from app.models import Document, Extraction, FieldEvidence
from app.services.ingestion_pipeline import run_invoice_extraction

INVOICE_TEXT = (
    "INVOICE\nVendor: BrightPath Office Supplies Inc.\n"
    "Invoice Number: INV-2026-04471\nInvoice Date: 2026-06-14\n"
    "Subtotal: $3,590.00\nTax: $296.18\nTotal: $3,886.18\nCurrency: USD\n"
)


def test_validation_passes_consistent_invoice():
    fields = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-001",
        "invoice_date": "2026-06-14",
        "subtotal": 100.0,
        "tax": 8.25,
        "total": 108.25,
        "currency": "USD",
    }
    valid, errors, warnings = validate_invoice_fields(fields)
    assert valid is True
    assert errors == []


def test_validation_catches_arithmetic_mismatch():
    fields = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-001",
        "invoice_date": "2026-06-14",
        "subtotal": 100.0,
        "tax": 8.25,
        "total": 500.00,  # wildly inconsistent
        "currency": "USD",
    }
    valid, errors, warnings = validate_invoice_fields(fields)
    assert valid is False
    assert any("does not match total" in e for e in errors)


def test_validation_flags_missing_required_fields():
    fields = {"subtotal": 10.0, "tax": 1.0, "total": 11.0}
    valid, errors, warnings = validate_invoice_fields(fields)
    assert valid is False
    assert any("vendor_name" in e for e in errors)
    assert any("invoice_number" in e for e in errors)


def test_validation_warns_on_bad_date_format():
    fields = {
        "vendor_name": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": "not-a-date",
        "subtotal": 10.0,
        "tax": 1.0,
        "total": 11.0,
        "currency": "USD",
    }
    valid, errors, warnings = validate_invoice_fields(fields)
    assert any("invoice_date" in w for w in warnings)


def test_confidence_high_for_complete_valid_invoice():
    fields = {
        "vendor_name": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": "2026-06-14",
        "subtotal": 100.0,
        "tax": 8.0,
        "total": 108.0,
        "currency": "USD",
    }
    valid, errors, warnings = validate_invoice_fields(fields)
    score, breakdown, status = compute_confidence(fields, valid, errors, warnings, ocr_ratio=0.0)
    assert status == "AUTO_APPROVED"
    assert score >= 0.85


def test_confidence_never_auto_approves_invalid_invoice():
    """A hard validation error must always route to review, regardless
    of how complete the fields otherwise look."""
    fields = {
        "vendor_name": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": "2026-06-14",
        "subtotal": 100.0,
        "tax": 8.0,
        "total": 500.0,  # inconsistent
        "currency": "USD",
    }
    valid, errors, warnings = validate_invoice_fields(fields)
    score, breakdown, status = compute_confidence(fields, valid, errors, warnings, ocr_ratio=0.0)
    assert valid is False
    assert status in ("NEEDS_REVIEW", "FAILED")
    assert status != "AUTO_APPROVED"


def test_confidence_failed_for_empty_extraction():
    fields = {k: None for k in ["vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency"]}
    valid, errors, warnings = validate_invoice_fields(fields)
    score, breakdown, status = compute_confidence(fields, valid, errors, warnings)
    assert status == "FAILED"


def test_confidence_degrades_with_high_ocr_ratio():
    fields = {
        "vendor_name": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": "2026-06-14",
        "subtotal": 100.0,
        "tax": 8.0,
        "total": 108.0,
        "currency": "USD",
    }
    valid, errors, warnings = validate_invoice_fields(fields)
    score_clean, _, _ = compute_confidence(fields, valid, errors, warnings, ocr_ratio=0.0)
    score_ocr, _, _ = compute_confidence(fields, valid, errors, warnings, ocr_ratio=1.0)
    assert score_ocr < score_clean


def test_extract_invoice_fields_mock_llm_extracts_known_values():
    text = (
        "INVOICE\nVendor: BrightPath Office Supplies Inc.\n"
        "Invoice Number: INV-2026-04471\nInvoice Date: 2026-06-14\n"
        "Subtotal: $3,590.00\nTax: $296.18\nTotal: $3,886.18\nCurrency: USD\n"
    )
    fields = extract_invoice_fields(text)
    assert fields["vendor_name"] == "BrightPath Office Supplies Inc."
    assert fields["invoice_number"] == "INV-2026-04471"
    assert float(fields["subtotal"]) == 3590.00
    assert float(fields["total"]) == 3886.18


def test_extract_invoice_fields_handles_missing_data_gracefully():
    fields = extract_invoice_fields("This document contains no invoice information at all.")
    # Should not raise, and should return the expected key shape with nulls.
    assert {"vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency"}.issubset(fields.keys())


# ---------------------------------------------------------------------------
# Field-level evidence grounding
# ---------------------------------------------------------------------------
from app.document_ai.extraction import (
    FieldEvidenceData,
    SEARCHABLE_FIELDS,
    _decimal_search_variants,
    _extract_line,
    _search_value_in_pages,
)
from app.models import Document


def test_ground_field_evidence_creates_evidence_for_found_fields():
    fields = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-001",
        "invoice_date": "2026-06-14",
        "subtotal": Decimal("100.00"),
        "tax": Decimal("8.00"),
        "total": Decimal("108.00"),
        "currency": "USD",
    }
    pages = [(1, "Vendor: Acme Corp\nInvoice Number: INV-001\nTotal: 108.00\nCurrency: USD\n")]
    evidence = ground_field_evidence(fields, pages)
    assert evidence["vendor_name"] is not None
    assert evidence["invoice_number"] is not None
    assert evidence["total"] is not None


def test_ground_field_evidence_correct_field_name():
    fields = {"invoice_number": "INV-001"}
    pages = [(1, "Invoice Number: INV-001")]
    evidence = ground_field_evidence(fields, pages)
    assert evidence["invoice_number"].field_name == "invoice_number"


def test_ground_field_evidence_contains_page_number():
    fields = {"vendor_name": "Acme Corp"}
    pages = [(1, "Vendor: Acme Corp"), (2, "Second page")]
    evidence = ground_field_evidence(fields, pages)
    assert evidence["vendor_name"].page_number == 1


def test_ground_field_evidence_contains_source_text():
    fields = {"invoice_number": "INV-001"}
    pages = [(1, "Some header\nInvoice Number: INV-001\nSome footer")]
    evidence = ground_field_evidence(fields, pages)
    assert "INV-001" in evidence["invoice_number"].source_text


def test_ground_field_evidence_confidence_high_for_single_match():
    fields = {"total": Decimal("108.00")}
    pages = [(1, "Total: 108.00")]
    evidence = ground_field_evidence(fields, pages)
    assert evidence["total"] is not None
    assert evidence["total"].confidence == 1.0


def test_ground_field_evidence_confidence_half_for_multiple_pages():
    fields = {"total": Decimal("108.00")}
    pages = [
        (1, "Total: 108.00"),
        (2, "Total: 108.00"),
    ]
    evidence = ground_field_evidence(fields, pages)
    assert evidence["total"] is not None
    assert evidence["total"].confidence == 0.5


def test_ground_field_evidence_none_for_missing_field_value():
    fields = {"vendor_name": None, "invoice_number": "INV-001"}
    pages = [(1, "Invoice Number: INV-001")]
    evidence = ground_field_evidence(fields, pages)
    assert evidence["vendor_name"] is None
    assert evidence["invoice_number"] is not None


def test_ground_field_evidence_none_for_empty_string_value():
    fields = {"vendor_name": "", "invoice_number": "INV-001"}
    pages = [(1, "Invoice Number: INV-001")]
    evidence = ground_field_evidence(fields, pages)
    assert evidence["vendor_name"] is None


def test_ground_field_evidence_none_for_value_not_in_any_page():
    fields = {"vendor_name": "Ghost Vendor"}
    pages = [(1, "Vendor: Acme Corp")]
    evidence = ground_field_evidence(fields, pages)
    assert evidence["vendor_name"] is None


def test_ground_field_evidence_none_when_no_pages():
    fields = {"vendor_name": "Acme Corp"}
    evidence = ground_field_evidence(fields, None)
    assert evidence["vendor_name"] is None
    evidence_no_pages = ground_field_evidence(fields, [])
    assert evidence_no_pages["vendor_name"] is None


def test_ground_field_evidence_decimal_variants_matched():
    fields = {"subtotal": Decimal("3590.00")}
    pages = [(1, "Subtotal: $3,590.00")]
    evidence = ground_field_evidence(fields, pages)
    assert evidence["subtotal"] is not None
    assert evidence["subtotal"].confidence == 1.0


def test_ground_field_evidence_db_persistence(db_session):
    fields = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-001",
        "invoice_date": "2026-06-14",
        "subtotal": Decimal("100.00"),
        "tax": Decimal("8.00"),
        "total": Decimal("108.00"),
        "currency": "USD",
    }
    pages = [(1, "Vendor: Acme Corp\nInvoice Number: INV-001\nTotal: 108.00\nCurrency: USD\n")]
    evidence_map = ground_field_evidence(fields, pages)
    doc = Document(
        filename="test_invoice.pdf",
        original_filename="test_invoice.pdf",
        file_path="/tmp/test_invoice.pdf",
        doc_type="invoice",
        status="processed",
        num_pages=1,
    )
    db_session.add(doc)
    db_session.flush()
    extraction = Extraction(
        document_id=doc.id,
        doc_type="invoice",
        fields=fields,
        valid=True,
        validation_errors=[],
        validation_warnings=[],
        confidence_score=0.95,
        confidence_breakdown={},
        status="AUTO_APPROVED",
    )
    db_session.add(extraction)
    db_session.flush()
    for field_name, evidence in evidence_map.items():
        if evidence is not None:
            db_session.add(FieldEvidence(
                extraction_id=extraction.id,
                field_name=evidence.field_name,
                page_number=evidence.page_number,
                source_text=evidence.source_text,
                confidence=evidence.confidence,
            ))
    db_session.commit()
    persisted = db_session.query(FieldEvidence).filter(
        FieldEvidence.extraction_id == extraction.id
    ).all()
    assert len(persisted) > 0
    for record in persisted:
        assert record.field_name in SEARCHABLE_FIELDS
        assert record.confidence in (1.0, 0.5)
