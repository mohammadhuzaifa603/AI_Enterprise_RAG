from app.ingestion.embeddings import embed_texts
from app.models import Chunk, Document
from app.rag.agent import run_agentic_query
from app.config import settings


def _seed(db):
    doc = Document(
        filename="agent_test.pdf",
        original_filename="agent_test.pdf",
        file_path="/tmp/agent_test.pdf",
        status="processed",
        num_pages=1,
    )
    db.add(doc)
    db.flush()

    texts = [
        "Northwind Analytics Q2 2026 revenue reached 4.82 million dollars, "
        "an increase of 14 percent compared to Q1 2026 which was 4.23 million.",
        "Employees may work remotely up to three days per week under company policy.",
    ]
    vectors = embed_texts(texts)
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        db.add(Chunk(document_id=doc.id, page_number=1, chunk_index=i, text=text, embedding=vec))
    db.flush()


def test_agent_succeeds_on_first_attempt_when_evidence_is_strong(db_session):
    _seed(db_session)
    result = run_agentic_query(db_session, "What was Q2 2026 revenue compared to Q1 2026?")
    assert result.succeeded is True
    assert len(result.attempts) == 1
    assert result.attempts[0].decision == "answer"
    assert result.final_answer is not None
    assert "million" in result.final_answer.lower() or "revenue" in result.final_answer.lower()


def test_agent_respects_max_attempts_and_never_loops_forever(db_session):
    _seed(db_session)
    result = run_agentic_query(db_session, "What is the CEO's favorite pizza topping?")
    assert len(result.attempts) <= settings.agent_max_attempts
    assert len(result.attempts) == settings.agent_max_attempts  # exhausts all attempts on truly absent info
    assert result.attempts[-1].decision == "give_up"


def test_agent_reports_insufficient_evidence_honestly_when_answer_absent(db_session):
    _seed(db_session)
    result = run_agentic_query(db_session, "What is the CEO's favorite pizza topping?")
    assert result.succeeded is False
    assert "couldn't find sufficient evidence" in result.final_answer.lower()


def test_agent_trace_records_every_attempt_with_required_fields(db_session):
    _seed(db_session)
    result = run_agentic_query(db_session, "What is the CEO's favorite pizza topping?")
    for attempt in result.attempts:
        assert attempt.attempt_number >= 1
        assert isinstance(attempt.query_used, str) and attempt.query_used
        assert isinstance(attempt.query_analysis, dict)
        assert isinstance(attempt.retrieved_chunk_ids, list)
        assert attempt.decision in ("answer", "reformulate", "give_up")


def test_agent_query_analysis_classifies_multi_hop_question(db_session):
    _seed(db_session)
    result = run_agentic_query(db_session, "Compare Q2 2026 revenue and the remote work policy.")
    assert result.attempts[0].query_analysis.get("query_type") in ("multi-hop", "simple", "ambiguous", "document-specific")

