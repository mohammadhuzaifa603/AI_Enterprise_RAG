"""Tests for persistent Query/Agent execution traces.

Verifies that the full auditable execution trace is persisted:
  QueryRun (AgentRun) -> Attempts (RetrievalAttempt) -> Citations (with verification)
  plus final answer/success/refusal, reformulations, latency, and trace API access.
"""
from fastapi.testclient import TestClient

from app.api.query import persist_agent_run
from app.config import settings
from app.database import SessionLocal
from app.ingestion.embeddings import embed_texts
from app.main import app
from app.models import AgentRun, Chunk, Citation, Document, RetrievalAttempt
from app.rag.agent import run_agentic_query
from app.rag.generator import NO_EVIDENCE_ANSWER
from app.retrieval.filters import get_or_create_local_user

client = TestClient(app)


def _seed(db):
    """Create a document with chunks for trace tests."""
    doc = Document(
        filename="trace_test.pdf",
        original_filename="trace_test.pdf",
        file_path="/tmp/trace_test.pdf",
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
    return doc


def _run_and_persist(db, question):
    """Run the agent and persist its trace, returning (result, agent_run_row)."""
    user = get_or_create_local_user(db)
    result = run_agentic_query(db, question, user_id=user.id)
    run = persist_agent_run(db, result, user.id)
    db.flush()
    return result, run


# ---------------------------------------------------------------------------
# AgentRun (QueryRun) creation
# ---------------------------------------------------------------------------
def test_query_run_creation(db_session):
    _seed(db_session)
    result, run = _run_and_persist(db_session, "What was Q2 2026 revenue compared to Q1 2026?")
    assert result.succeeded is True
    persisted = db_session.get(AgentRun, run.id)
    assert persisted is not None
    assert persisted.question == "What was Q2 2026 revenue compared to Q1 2026?"
    assert persisted.succeeded is True
    assert persisted.num_attempts == len(result.attempts)
    assert persisted.latency_ms >= 0.0
    assert persisted.final_answer is not None
    assert persisted.user_id is not None


# ---------------------------------------------------------------------------
# Attempt persistence
# ---------------------------------------------------------------------------
def test_attempt_persistence(db_session):
    _seed(db_session)
    result, run = _run_and_persist(db_session, "What was Q2 2026 revenue compared to Q1 2026?")
    attempts = db_session.query(RetrievalAttempt).filter(
        RetrievalAttempt.agent_run_id == run.id
    ).order_by(RetrievalAttempt.attempt_number).all()
    assert len(attempts) == len(result.attempts)
    assert all(a.query_used for a in attempts)
    assert all(a.decision in ("answer", "reformulate", "give_up") for a in attempts)
    assert attempts[0].query_used == "What was Q2 2026 revenue compared to Q1 2026?"
    for a in attempts:
        assert isinstance(a.query_analysis, dict)
        assert a.evidence_confidence >= 0.0


# ---------------------------------------------------------------------------
# Evidence (retrieved chunk IDs) persistence
# ---------------------------------------------------------------------------
def test_evidence_persistence(db_session):
    _seed(db_session)
    result, run = _run_and_persist(db_session, "What was Q2 2026 revenue compared to Q1 2026?")
    attempt = db_session.query(RetrievalAttempt).filter(
        RetrievalAttempt.agent_run_id == run.id
    ).first()
    assert attempt is not None
    assert isinstance(attempt.retrieved_chunk_ids, list)
    assert len(attempt.retrieved_chunk_ids) > 0
    chunk_ids = {c.id for c in db_session.query(Chunk.id).all()}
    for cid in attempt.retrieved_chunk_ids:
        assert cid in chunk_ids


# ---------------------------------------------------------------------------
# Citation persistence (with verification results)
# ---------------------------------------------------------------------------
def test_citation_persistence(db_session):
    _seed(db_session)
    result, run = _run_and_persist(db_session, "What was Q2 2026 revenue compared to Q1 2026?")
    assert result.succeeded is True
    assert len(result.citations) > 0
    citations = db_session.query(Citation).filter(Citation.agent_run_id == run.id).all()
    assert len(citations) == len(result.citations)
    for c in citations:
        assert c.chunk_id is not None
        assert c.claim_text
        assert 0.0 <= c.confidence <= 1.0
        assert isinstance(c.supported, bool)
        assert c.reason  # verification reason recorded


# ---------------------------------------------------------------------------
# Successful run
# ---------------------------------------------------------------------------
def test_successful_run_recorded(db_session):
    _seed(db_session)
    result, run = _run_and_persist(db_session, "What was Q2 2026 revenue compared to Q1 2026?")
    persisted = db_session.get(AgentRun, run.id)
    assert persisted.succeeded is True
    assert persisted.final_answer is not None
    assert "million" in persisted.final_answer.lower() or "revenue" in persisted.final_answer.lower()
    cit_count = db_session.query(Citation).filter(Citation.agent_run_id == run.id).count()
    assert cit_count > 0


# ---------------------------------------------------------------------------
# Refused run
# ---------------------------------------------------------------------------
def test_refused_run_recorded(db_session):
    _seed(db_session)
    result, run = _run_and_persist(db_session, "What is the CEO's favorite pizza topping?")
    assert result.succeeded is False
    persisted = db_session.get(AgentRun, run.id)
    assert persisted.succeeded is False
    assert persisted.num_attempts == settings.agent_max_attempts
    assert persisted.final_answer == NO_EVIDENCE_ANSWER
    last = db_session.query(RetrievalAttempt).filter(
        RetrievalAttempt.agent_run_id == run.id
    ).order_by(RetrievalAttempt.attempt_number.desc()).first()
    assert last.decision == "give_up"
    cit_count = db_session.query(Citation).filter(Citation.agent_run_id == run.id).count()
    assert cit_count == 0


# ---------------------------------------------------------------------------
# Reformulation recording
# ---------------------------------------------------------------------------
def test_reformulated_queries_recorded(db_session):
    _seed(db_session)
    result, run = _run_and_persist(db_session, "What is the CEO's favorite pizza topping?")
    attempts = db_session.query(RetrievalAttempt).filter(
        RetrievalAttempt.agent_run_id == run.id
    ).order_by(RetrievalAttempt.attempt_number).all()
    assert len(attempts) == settings.agent_max_attempts
    # First attempt uses the original question
    assert attempts[0].query_used == "What is the CEO's favorite pizza topping?"
    # Intermediate attempts reformulate (decision="reformulate"), recording
    # the reformulated query as query_used
    reformulated = [a for a in attempts[1:] if a.decision == "reformulate"]
    assert len(reformulated) > 0
    for a in reformulated:
        assert a.query_used != attempts[0].query_used
    # Last attempt gives up when evidence remains insufficient
    assert attempts[-1].decision == "give_up"


# ---------------------------------------------------------------------------
# Trace API
# ---------------------------------------------------------------------------
def test_trace_api_returns_trace():
    resp = client.post("/agent/query", json={"question": "What is the meaning of life?"})
    assert resp.status_code == 200
    run_id = resp.json()["id"]
    resp = client.get(f"/agent/trace/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == run_id
    assert len(data["attempts"]) >= 1
    assert data["num_attempts"] == len(data["attempts"])
    assert data["latency_ms"] >= 0.0
    assert data["question"] == "What is the meaning of life?"


def test_trace_api_returns_404_for_unknown_run():
    resp = client.get("/agent/trace/does-not-exist")
    assert resp.status_code == 404


def test_trace_api_latency_recorded():
    resp = client.post("/agent/query", json={"question": "What is the meaning of life?"})
    assert resp.status_code == 200
    run_id = resp.json()["id"]
    resp = client.get(f"/agent/trace/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Trace API authorization
# ---------------------------------------------------------------------------
def test_trace_api_allows_owner_access():
    """The owner of an agent run can retrieve their trace."""
    resp = client.post("/agent/query", json={"question": "What is the meaning of life?"})
    assert resp.status_code == 200
    run_id = resp.json()["id"]
    resp = client.get(f"/agent/trace/{run_id}")
    assert resp.status_code == 200


def test_trace_api_denies_unauthorized_access():
    """A trace owned by a different user is not accessible (403)."""
    db = SessionLocal()
    try:
        run = AgentRun(
            user_id="intruder-nonexistent-user",
            question="secret question",
            final_answer="no",
            succeeded=False,
            num_attempts=1,
            latency_ms=0.0,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id
    finally:
        db.close()
    try:
        resp = client.get(f"/agent/trace/{run_id}")
        assert resp.status_code == 403
    finally:
        db = SessionLocal()
        try:
            db.query(AgentRun).filter(AgentRun.id == run_id).delete()
            db.commit()
        finally:
            db.close()
