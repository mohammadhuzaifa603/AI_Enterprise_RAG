"""Shared pytest fixtures: an isolated SQLite DB per test session, with
each individual test running inside a transaction that is rolled back
afterwards so tests never see data left behind by other tests."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point at a throwaway DB file BEFORE importing anything from `app`.
_TMP_DIR = tempfile.mkdtemp(prefix="ai_enterprise_rag_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DIR}/test.db"
os.environ["LLM_API_KEY"] = ""
from app.database import SessionLocal, engine, init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    init_db()
    yield


@pytest.fixture()
def db_session():
    """A session bound to a transaction that is rolled back after the
    test. It also clears document/chunk data at the start (within that
    same transaction) so each test sees a clean slate regardless of
    what other test modules have committed to the shared test DB file -
    none of this is visible outside the test since it is never committed."""
    from app.models import (
        AgentRun, AuditLog, Chunk, Citation,
        Document, DocumentPermission, Extraction, FieldEvidence,
        Page, Query, RetrievalAttempt, Review,
    )

    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        session.query(Citation).delete()
        session.query(RetrievalAttempt).delete()
        session.query(FieldEvidence).delete()
        session.query(Review).delete()
        session.query(Extraction).delete()
        session.query(AgentRun).delete()
        session.query(Query).delete()
        session.query(AuditLog).delete()
        session.query(DocumentPermission).delete()
        session.query(Chunk).delete()
        session.query(Page).delete()
        session.query(Document).delete()
        session.flush()
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
