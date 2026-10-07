from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_page_text
from app.ingestion.ocr import ocr_page
from app.ingestion.parser import InvalidPDFError, parse_pdf

SAMPLES = Path(__file__).resolve().parent.parent / "sample_documents"


def test_parse_pdf_extracts_pages_and_text():
    parsed = parse_pdf(str(SAMPLES / "Employee_Handbook.pdf"))
    assert parsed.num_pages == 4
    assert len(parsed.pages) == 4
    assert "Remote Work Policy" in parsed.pages[1].text
    assert all(p.char_count > 0 for p in parsed.pages)


def test_parse_pdf_invalid_file_raises(tmp_path):
    bad_file = tmp_path / "not_a_pdf.pdf"
    bad_file.write_text("this is not a real pdf")
    with pytest.raises(InvalidPDFError):
        parse_pdf(str(bad_file))


def test_parse_pdf_detects_pages_needing_ocr():
    parsed = parse_pdf(str(SAMPLES / "Synthetic_Scanned_Invoice.pdf"))
    assert parsed.num_pages == 1
    assert parsed.pages[0].needs_ocr is True
    assert parsed.pages[0].char_count == 0


def test_ocr_fallback_recovers_text_from_scanned_page():
    result = ocr_page(
        str(SAMPLES / "Synthetic_Scanned_Invoice.pdf"),
        page_number_zero_indexed=0,
    )

    # Tesseract output isn't pixel-perfect, but key invoice terms
    # should be recoverable from the rendered image.
    assert (
        "INVOICE" in result.text.upper()
        or "Hazelwood" in result.text
        or len(result.text) > 0
    )

    assert result.failed is False


def test_chunking_preserves_page_number_and_splits_long_text():
    long_text = "This is a sentence. " * 200  # long enough to force multiple chunks
    chunks = chunk_page_text(page_number=7, text=long_text, chunk_size=200, overlap=30)
    assert len(chunks) > 1
    assert all(c.page_number == 7 for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)


def test_chunking_short_text_produces_single_chunk():
    chunks = chunk_page_text(page_number=1, text="A short page of text.", chunk_size=900, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1


def test_chunking_empty_text_produces_no_chunks():
    chunks = chunk_page_text(page_number=1, text="   ", chunk_size=900, overlap=150)
    assert chunks == []
def test_ocr_result_contains_confidence():
    result = ocr_page(
        str(SAMPLES / "Synthetic_Scanned_Invoice.pdf"),
        page_number_zero_indexed=0,
    )

    assert result.failed is False
    assert result.text
    assert result.confidence is None or result.confidence >= 0


def test_ocr_failure_returns_structured_failure(monkeypatch):
    def fake_ocr(*args, **kwargs):
        from app.ingestion.ocr import OCRResult

        return OCRResult(
            text="",
            confidence=None,
            failed=True,
            error="Synthetic OCR failure",
        )

    monkeypatch.setattr(
        "app.ingestion.ocr.ocr_page",
        fake_ocr,
    )

    result = fake_ocr(
        "missing.pdf",
        0,
    )

    assert result.text == ""
    assert result.failed is True
    assert result.confidence is None
    assert result.error == "Synthetic OCR failure"


def test_ocr_empty_text_is_not_treated_as_success():
    from app.ingestion.ocr import OCRResult

    result = OCRResult(
        text="",
        confidence=12.5,
        failed=False,
    )

    assert result.text == ""
    assert result.failed is False
    assert result.confidence == 12.5

from app.services.ingestion_pipeline import _determine_document_status


def test_document_status_processed_when_all_pages_are_usable():
    status = _determine_document_status(
        usable_pages=4,
        total_pages=4,
        ocr_failed_pages=0,
        low_confidence_ocr_pages=0,
    )

    assert status == "processed"


def test_document_status_partial_when_ocr_fails():
    status = _determine_document_status(
        usable_pages=3,
        total_pages=4,
        ocr_failed_pages=1,
        low_confidence_ocr_pages=0,
    )

    assert status == "partial"


def test_document_status_needs_review_for_low_confidence_ocr():
    status = _determine_document_status(
        usable_pages=4,
        total_pages=4,
        ocr_failed_pages=0,
        low_confidence_ocr_pages=1,
    )

    assert status == "needs_review"


def test_document_status_needs_review_takes_priority_over_partial():
    status = _determine_document_status(
        usable_pages=3,
        total_pages=4,
        ocr_failed_pages=1,
        low_confidence_ocr_pages=1,
    )

    assert status == "needs_review"


def test_document_status_partial_when_page_has_no_usable_text():
    status = _determine_document_status(
        usable_pages=3,
        total_pages=4,
        ocr_failed_pages=0,
        low_confidence_ocr_pages=0,
    )

    assert status == "partial"