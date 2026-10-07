from decimal import Decimal
from app.document_ai.validation import validate_invoice_fields
from app.evidence.sufficiency import evaluate_evidence
from app.evidence.models import Evidence
from app.models import Document, DocumentPermission, User, Chunk
from app.retrieval.filters import accessible_chunk_query


def test_evidence_evaluation_uses_actual_passages():
    evidence = [Evidence("c1", "d1", "policy.pdf", 2, "Employees may work remotely three days per week.")]
    decision = evaluate_evidence("How many days can employees work remotely?", evidence)
    assert decision.sufficient is True
    assert decision.confidence > 0


def test_invoice_validation_uses_decimal_safe_arithmetic():
    valid, errors, _ = validate_invoice_fields({
        "vendor_name": "Acme", "invoice_number": "INV-1", "subtotal": Decimal("100.00"),
        "tax": Decimal("10.00"), "total": Decimal("110.00"), "currency": "USD"
    })
    assert valid is True
    assert errors == []


def test_non_admin_retrieval_is_permission_aware(db_session):
    user = User(username="finance-user", role="USER")
    other = User(username="other-user", role="USER")
    db_session.add_all([user, other]); db_session.flush()
    d1 = Document(filename="finance.pdf", original_filename="finance.pdf", file_path="/tmp/a", owner_id=user.id, status="processed")
    d2 = Document(filename="private.pdf", original_filename="private.pdf", file_path="/tmp/b", owner_id=other.id, status="processed")
    db_session.add_all([d1, d2]); db_session.flush()
    db_session.add(Chunk(document_id=d1.id, page_number=1, chunk_index=0, text="finance data"))
    db_session.add(Chunk(document_id=d2.id, page_number=1, chunk_index=0, text="private data"))
    db_session.flush()
    ids = {c.document_id for c in accessible_chunk_query(db_session, user.id).all()}
    assert d1.id in ids
    assert d2.id not in ids

def test_evidence_sufficiency_rejects_unrelated_generic_overlap():
    from app.evidence import Evidence, evaluate_evidence

    evidence = [
        Evidence(
            chunk_id="c1",
            document_id="d1",
            document_name="Employee_Handbook.pdf",
            page=2,
            text=(
                "Section 3: Remote Work Policy. "
                "The company allows eligible employees to work remotely "
                "up to three days per week."
            ),
            retrieval_score=0.9,
            rerank_score=0.9,
            chunk_type="text",
        )
    ]

    decision = evaluate_evidence(
        "What is the company's maternity leave policy for employees working in Antarctica?",
        evidence,
    )

    assert decision.sufficient is False


def test_direct_query_refuses_when_evidence_is_insufficient(db_session):
    from app.models import Document, Chunk
    from app.rag.generator import answer_question, NO_EVIDENCE_ANSWER
    from app.ingestion.embeddings import embed_text

    doc = Document(
        filename="Employee_Handbook.pdf",
        original_filename="Employee_Handbook.pdf",
        file_path="/tmp/handbook.pdf",
        status="processed",
    )

    db_session.add(doc)
    db_session.flush()

    text = (
        "Section 3: Remote Work Policy. "
        "The company allows eligible employees to work remotely "
        "up to three days per week."
    )

    db_session.add(
        Chunk(
            document_id=doc.id,
            page_number=2,
            chunk_index=0,
            text=text,
            embedding=embed_text(text),
        )
    )

    db_session.flush()

    result = answer_question(
        db_session,
        "What is the company's maternity leave policy for employees working in Antarctica?",
    )

    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.citations == []