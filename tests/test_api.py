from pathlib import Path

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import (
    Chunk,
    Document,
    DocumentPermission,
    Extraction,
    FieldEvidence,
    Page,
    Query,
    Review,
    User,
)
from app.retrieval.filters import DEFAULT_LOCAL_USER, get_or_create_local_user

client = TestClient(app)

SAMPLES = Path(__file__).resolve().parent.parent / "sample_documents"


def test_health_endpoint_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["database"] == "ok"
    assert data["llm_mode"] in ("mock", "live")


def test_root_endpoint_lists_modules():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["modules"]) == 3


def test_upload_rejects_non_pdf(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello")
    with bad_file.open("rb") as f:
        resp = client.post("/documents/upload", files={"file": ("notes.txt", f, "text/plain")})
    assert resp.status_code == 400


def test_upload_process_and_query_flow():
    pdf_path = SAMPLES / "Employee_Handbook.pdf"
    with pdf_path.open("rb") as f:
        upload_resp = client.post(
            "/documents/upload", files={"file": ("Employee_Handbook.pdf", f, "application/pdf")}, data={"doc_type": "generic"}
        )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    process_resp = client.post(f"/documents/process?document_id={doc_id}")
    assert process_resp.status_code == 200
    process_data = process_resp.json()
    assert process_data["status"] == "processed"
    assert process_data["num_chunks"] > 0

    query_resp = client.post("/query", json={"question": "How many days per week can employees work remotely?"})
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert "answer" in query_data
    assert query_data["retrieved_chunks"] > 0


def test_process_nonexistent_document_returns_404():
    resp = client.post("/documents/process?document_id=does-not-exist")
    assert resp.status_code == 404


def test_agent_query_endpoint_returns_trace():
    resp = client.post("/agent/query", json={"question": "How many days per week can employees work remotely?"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["attempts"]) >= 1
    assert data["num_attempts"] == len(data["attempts"])


# --------------------------------------------------------------------------
# Document deletion tests
# --------------------------------------------------------------------------

def _upload_and_process(filename: str = "Employee_Handbook.pdf"):
    pdf_path = SAMPLES / filename
    with pdf_path.open("rb") as f:
        upload_resp = client.post(
            "/documents/upload",
            files={"file": (filename, f, "application/pdf")},
            data={"doc_type": "generic"},
        )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    process_resp = client.post(f"/documents/process?document_id={doc_id}")
    assert process_resp.status_code == 200
    return doc_id


def test_document_deletion_success():
    """A document can be deleted and the response contains the right data."""
    doc_id = _upload_and_process()

    resp = client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == doc_id
    assert data["deleted"] is True
    assert "Employee_Handbook.pdf" in data["original_filename"]

    # Document no longer exists
    resp2 = client.get(f"/documents/{doc_id}")
    assert resp2.status_code == 404


def test_document_deletion_nonexistent_returns_404():
    """Deleting a document that does not exist returns 404."""
    resp = client.delete("/documents/nonexistent-id-12345")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_document_deletion_removes_child_records():
    """Deleting a document removes pages, chunks, and permissions via cascade."""
    doc_id = _upload_and_process()

    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        assert doc is not None
        assert db.query(Page).filter(Page.document_id == doc_id).count() > 0
        assert db.query(Chunk).filter(Chunk.document_id == doc_id).count() > 0
        assert db.query(DocumentPermission).filter(DocumentPermission.document_id == doc_id).count() > 0
    finally:
        db.close()

    resp = client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        assert db.get(Document, doc_id) is None
        assert db.query(Page).filter(Page.document_id == doc_id).count() == 0
        assert db.query(Chunk).filter(Chunk.document_id == doc_id).count() == 0
        assert db.query(DocumentPermission).filter(DocumentPermission.document_id == doc_id).count() == 0
    finally:
        db.close()


def test_document_deletion_removes_file():
    """Deleting a document removes the stored PDF file from disk."""
    doc_id = _upload_and_process()

    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        file_path = Path(doc.file_path)
        assert file_path.exists()
    finally:
        db.close()

    resp = client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200

    assert not file_path.exists()


def test_document_deletion_removes_extraction_children():
    """Deleting an invoice document removes extractions, field evidence, and reviews."""
    pdf_path = SAMPLES / "Synthetic_Invoice_Clean.pdf"
    with pdf_path.open("rb") as f:
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("Synthetic_Invoice_Clean.pdf", f, "application/pdf")},
            data={"doc_type": "invoice"},
        )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    process_resp = client.post(f"/documents/process?document_id={doc_id}")
    assert process_resp.status_code == 200

    db = SessionLocal()
    try:
        extraction_count = db.query(Extraction).filter(Extraction.document_id == doc_id).count()
        assert extraction_count > 0
        field_evidence_count = (
            db.query(FieldEvidence)
            .join(Extraction)
            .filter(Extraction.document_id == doc_id)
            .count()
        )
        assert field_evidence_count > 0
    finally:
        db.close()

    resp = client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        assert db.query(Extraction).filter(Extraction.document_id == doc_id).count() == 0
        assert (
            db.query(FieldEvidence).join(Extraction).filter(Extraction.document_id == doc_id).count() == 0
        )
        assert (
            db.query(Review).join(Extraction).filter(Extraction.document_id == doc_id).count() == 0
        )
    finally:
        db.close()


def test_other_documents_remain_after_deletion():
    """Deleting one document does not affect other documents."""
    doc_id_1 = _upload_and_process("Employee_Handbook.pdf")
    doc_id_2 = _upload_and_process("Remote_Work_Policy.pdf")

    assert doc_id_1 != doc_id_2

    resp = client.delete(f"/documents/{doc_id_1}")
    assert resp.status_code == 200

    # First document gone
    assert client.get(f"/documents/{doc_id_1}").status_code == 404

    # Second document still retrievable
    resp2 = client.get(f"/documents/{doc_id_2}")
    assert resp2.status_code == 200
    assert resp2.json()["original_filename"] == "Remote_Work_Policy.pdf"


def test_unauthorized_user_cannot_delete_document(monkeypatch):
    """A non-admin user who does not own the document gets 403."""
    doc_id = _upload_and_process()

    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        owner_id = doc.owner_id

        # Create a non-admin user who does NOT own this document
        other_user = User(username="other-user-12345", role="USER")
        db.add(other_user)
        db.commit()
        db.refresh(other_user)
    finally:
        db.close()

    # Patch get_or_create_local_user inside the documents module
    import app.api.documents as documents_module

    original = documents_module.get_or_create_local_user

    def _fake_user(db: Session):
        db.add(other_user)
        return other_user

    documents_module.get_or_create_local_user = _fake_user
    try:
        resp = client.delete(f"/documents/{doc_id}")
        assert resp.status_code == 403
        assert "not authorized" in resp.json()["detail"].lower()
    finally:
        documents_module.get_or_create_local_user = original

    # Clean up: delete the document as the admin (the real local user)
    client.delete(f"/documents/{doc_id}")


# --------------------------------------------------------------------------
# Query history tests
# --------------------------------------------------------------------------
def test_queries_recent_returns_history():
    """The /queries/recent endpoint returns previously run queries."""
    # Run a query to create a Query record
    resp = client.post("/query", json={"question": "What is the meaning of life?"})
    assert resp.status_code == 200
    query_id = resp.json()["query_id"]

    resp = client.get("/queries/recent?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    found = any(q["query_id"] == query_id for q in data)
    assert found
    first = data[0]
    assert "question" in first
    assert "answer" in first
    assert "mode" in first
    assert "num_citations" in first


def test_queries_recent_respects_limit():
    """The /queries/recent endpoint respects the limit parameter."""
    resp = client.get("/queries/recent?limit=1")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) <= 1


# --------------------------------------------------------------------------
# Invoice extraction tests
# --------------------------------------------------------------------------
def test_clean_invoice_extraction_succeeds():
    """Clean invoice PDF should extract fields and be auto-approved."""
    pdf_path = SAMPLES / "Synthetic_Invoice_Clean.pdf"
    with pdf_path.open("rb") as f:
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("Synthetic_Invoice_Clean.pdf", f, "application/pdf")},
            data={"doc_type": "invoice"},
        )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    process_resp = client.post(f"/documents/process?document_id={doc_id}")
    assert process_resp.status_code == 200
    process_data = process_resp.json()
    assert process_data["status"] == "processed"
    extraction_id = process_data.get("extraction_id")
    assert extraction_id is not None
    ext_resp = client.get(f"/extractions/{extraction_id}")
    assert ext_resp.status_code == 200
    ext = ext_resp.json()
    assert ext["status"] == "AUTO_APPROVED"
    assert ext["valid"] is True
    fields = ext["fields"]
    assert fields.get("total") is not None
    assert fields.get("vendor_name") is not None


def test_inconsistent_invoice_goes_to_review():
    """Inconsistent invoice should extract but fail validation → NEEDS_REVIEW."""
    pdf_path = SAMPLES / "Synthetic_Invoice_Inconsistent_Total.pdf"
    with pdf_path.open("rb") as f:
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("Synthetic_Invoice_Inconsistent_Total.pdf", f, "application/pdf")},
            data={"doc_type": "invoice"},
        )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    process_resp = client.post(f"/documents/process?document_id={doc_id}")
    assert process_resp.status_code == 200
    process_data = process_resp.json()
    assert process_data["status"] in ("processed", "needs_review", "partial")
    extraction_id = process_data.get("extraction_id")
    assert extraction_id is not None
    ext_resp = client.get(f"/extractions/{extraction_id}")
    assert ext_resp.status_code == 200
    ext = ext_resp.json()
    assert ext["status"] in ("NEEDS_REVIEW", "FAILED")
    assert ext["valid"] is False
    assert len(ext["validation_errors"]) > 0
