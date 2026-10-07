from app.rag.citations import EvidenceItem, ground_citations, validate_citation
from app.rag.verifier import verify_claim


def _evidence():
    return [
        EvidenceItem(chunk_id="c1", document_name="Handbook.pdf", page=2, text="Employees may work remotely up to three days per week."),
        EvidenceItem(chunk_id="c2", document_name="Handbook.pdf", page=5, text="PTO accrues at 20 days per year."),
    ]


def test_validate_citation_matches_by_chunk_id():
    items = _evidence()
    citation = {"chunk_id": "c1", "document": "Handbook.pdf", "page": 2}
    match = validate_citation(citation, items)
    assert match is not None
    assert match.chunk_id == "c1"


def test_validate_citation_matches_by_document_and_page_fallback():
    items = _evidence()
    citation = {"chunk_id": "wrong-id", "document": "Handbook.pdf", "page": 5}
    match = validate_citation(citation, items)
    assert match is not None
    assert match.chunk_id == "c2"


def test_validate_citation_rejects_fabricated_reference():
    items = _evidence()
    citation = {"chunk_id": "does-not-exist", "document": "Made Up Document.pdf", "page": 999}
    match = validate_citation(citation, items)
    assert match is None


def test_ground_citations_drops_ungrounded_and_keeps_valid():
    items = _evidence()
    raw = [
        {"chunk_id": "c1", "document": "Handbook.pdf", "page": 2, "claim_text": "Remote work is allowed 3 days/week."},
        {"chunk_id": "fabricated", "document": "Nonexistent.pdf", "page": 42, "claim_text": "Invented fact."},
    ]
    grounded = ground_citations(raw, items)
    assert len(grounded) == 1
    assert grounded[0]["chunk_id"] == "c1"
    assert grounded[0]["excerpt"].startswith("Employees may work remotely")


def test_verify_claim_supported_when_terms_overlap():
    result = verify_claim(
        claim_text="Employees may work remotely up to three days per week.",
        source_text="Employees may work remotely up to three days per week, per company policy.",
    )
    assert result.supported is True
    assert result.confidence > 0.5


def test_verify_claim_unsupported_when_terms_dont_overlap():
    result = verify_claim(
        claim_text="The company offers unlimited vacation days.",
        source_text="Employees may work remotely up to three days per week.",
    )
    assert result.supported is False


def test_verify_claim_handles_empty_input_gracefully():
    result = verify_claim(claim_text="", source_text="Some source text.")
    assert result.supported is False
    assert result.confidence == 0.0
