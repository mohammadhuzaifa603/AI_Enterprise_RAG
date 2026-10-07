"""Regression tests for the ingest script storage/encryption fix.

These tests verify that ``scripts/ingest.py`` correctly uses the
``save_document()`` storage abstraction and produces ``Document`` records
whose ``storage_encrypted`` flag matches the actual file on disk, so that
``process_document()`` can materialise and parse them without encryption
errors.
"""
import io
from pathlib import Path

from app.config import settings
from app.models import Document
from app.storage import save_document
from scripts.ingest import guess_doc_type, ingest_pdf

SAMPLES = Path(__file__).resolve().parent.parent / "sample_documents"


def test_guess_doc_type_generic():
    assert guess_doc_type("handbook.pdf") == "generic"


def test_guess_doc_type_invoice():
    assert guess_doc_type("Invoice_Clean.pdf") == "invoice"
    assert guess_doc_type("inVOICE.pdf") == "invoice"


def test_ingest_pdf_sets_storage_encrypted_correctly(db_session):
    """The ``storage_encrypted`` column must reflect the real file state."""
    pdf_path = SAMPLES / "Remote_Work_Policy.pdf"
    assert pdf_path.exists(), "sample document must exist"

    document, result = ingest_pdf(db_session, pdf_path, "generic")

    assert document.file_sha256 is not None
    assert len(document.file_sha256) == 64

    stored_on_disk = Path(document.file_path).exists()
    assert stored_on_disk, f"stored file must exist at {document.file_path}"

    # storage_encrypted must be False when no STORAGE_ENCRYPTION_KEY is set
    # (and STORAGE_ENCRYPTION_REQUIRED is False).
    if not settings.storage_encryption_key:
        assert document.storage_encrypted is False
        assert document.file_path.endswith(".pdf")
    else:
        assert document.storage_encrypted is True
        assert document.file_path.endswith(".enc")


def test_ingest_pdf_processes_without_encryption_error(db_session):
    """End-to-end: ingest_pdf must produce a 'processed' document."""
    pdf_path = SAMPLES / "Remote_Work_Policy.pdf"
    document, result = ingest_pdf(db_session, pdf_path, "generic")
    assert result.status == "processed"
    assert document.status == "processed"
    assert document.num_pages == 2
    assert document.file_path.endswith(".pdf") or document.file_path.endswith(".enc")


def test_ingest_pdf_dedupes_same_content(db_session):
    """Two calls with identical PDF bytes produce one stored file."""
    pdf_path = SAMPLES / "Remote_Work_Policy.pdf"
    doc1, _ = ingest_pdf(db_session, pdf_path, "generic")
    doc2, _ = ingest_pdf(db_session, pdf_path, "generic")

    # Same content → same digest → same file path
    assert doc1.file_sha256 == doc2.file_sha256
    assert doc1.file_path == doc2.file_path
    assert Path(doc1.file_path).exists()
    assert Path(doc2.file_path).exists()
