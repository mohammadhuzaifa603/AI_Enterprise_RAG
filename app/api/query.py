from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.audit import log_event
from app.database import get_db
from app.models import AgentRun, Citation, Query, RetrievalAttempt
from app.rag.agent import AgentResult, run_agentic_query
from app.rag.generator import answer_question
from app.retrieval.filters import get_or_create_local_user
from app.schemas import AgentQueryRequest, AgentRunOut, CitationOut, QueryOut, QueryRequest, QueryResponse, RetrievalAttemptOut

router = APIRouter(tags=["query"])

def persist_agent_run(db: Session, result: AgentResult, user_id: str) -> AgentRun:
    """Persist a completed agent execution trace to the database.

    Returns the created AgentRun (flushed but not committed).
    """
    run_row = AgentRun(
        user_id=user_id,
        question=result.question,
        final_answer=result.final_answer,
        succeeded=result.succeeded,
        num_attempts=len(result.attempts),
        latency_ms=result.latency_ms,
    )
    db.add(run_row)
    db.flush()
    for attempt in result.attempts:
        db.add(RetrievalAttempt(
            agent_run_id=run_row.id,
            attempt_number=attempt.attempt_number,
            query_used=attempt.query_used,
            query_analysis=attempt.query_analysis,
            retrieved_chunk_ids=attempt.retrieved_chunk_ids,
            top_score=attempt.top_score,
            evidence_sufficient=attempt.evidence_sufficient,
            evidence_confidence=attempt.evidence_confidence,
            missing_information=attempt.missing_information,
            decision=attempt.decision,
        ))
    for c in result.citations:
        db.add(Citation(
            agent_run_id=run_row.id,
            chunk_id=c.get("chunk_id"),
            document_name=c.get("document", ""),
            page=c.get("page", 0),
            claim_text=c.get("claim_text", ""),
            excerpt=c.get("excerpt", ""),
            supported=c.get("supported", False),
            confidence=c.get("confidence", 0.0),
            reason=c.get("reason", ""),
        ))
    return run_row

@router.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest, db: Session = Depends(get_db)):
    user = get_or_create_local_user(db)
    result = answer_question(db, payload.question, top_k=payload.top_k)
    row = Query(user_id=user.id, question=payload.question, answer=result.answer, mode="hybrid", latency_ms=result.latency_ms, num_citations=len(result.citations), num_verified_citations=sum(1 for c in result.citations if c.get("supported")))
    db.add(row); db.flush()
    outs=[]
    for c in result.citations:
        db.add(Citation(query_id=row.id, chunk_id=c.get("chunk_id"), document_name=c.get("document", ""), page=c.get("page", 0), claim_text=c.get("claim_text", ""), excerpt=c.get("excerpt", ""), supported=c.get("supported", False), confidence=c.get("confidence", 0.0), reason=c.get("reason", "")))
        outs.append(CitationOut(**c))
    log_event(db, "QUERY_EXECUTED", user.id, query_id=row.id, metadata={"mode":"hybrid", "retrieved_chunks":result.retrieved_chunks})
    db.commit()
    return QueryResponse(query_id=row.id, question=payload.question, answer=result.answer, citations=outs, retrieved_chunks=result.retrieved_chunks, latency_ms=result.latency_ms, llm_mode=result.llm_mode)

@router.get("/queries/recent", response_model=list[QueryOut])
def get_recent_queries(limit: int = 20, db: Session = Depends(get_db)):
    """Return the most recent query/answer pairs for dashboard display."""
    rows = (
        db.query(Query)
        .order_by(Query.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        QueryOut(
            query_id=r.id,
            question=r.question,
            answer=r.answer,
            mode=r.mode,
            num_citations=r.num_citations,
            num_verified_citations=r.num_verified_citations,
            latency_ms=r.latency_ms,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.post("/agent/query", response_model=AgentRunOut)
def agent_query(payload: AgentQueryRequest, db: Session = Depends(get_db)):
    user = get_or_create_local_user(db)
    result = run_agentic_query(db, payload.question, user.id)
    run_row = persist_agent_run(db, result, user.id)
    log_event(db, "AGENT_QUERY_EXECUTED", user.id, query_id=run_row.id, metadata={"attempts":len(result.attempts), "succeeded":result.succeeded})
    db.commit(); db.refresh(run_row)
    return _agent_run_to_schema(run_row)

@router.get("/agent/trace/{agent_run_id}", response_model=AgentRunOut)
def get_agent_trace(agent_run_id: str, db: Session = Depends(get_db)):
    user = get_or_create_local_user(db)
    row = db.get(AgentRun, agent_run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Agent run not found.")
    if row.user_id is not None and row.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this trace.")
    return _agent_run_to_schema(row)

def _agent_run_to_schema(row: AgentRun) -> AgentRunOut:
    citations = [CitationOut(document=c.document_name, page=c.page, chunk_id=c.chunk_id, excerpt=c.excerpt, claim_text=c.claim_text, supported=c.supported, confidence=c.confidence, reason=c.reason) for c in row.citations]
    return AgentRunOut(
        id=row.id,
        question=row.question,
        final_answer=row.final_answer,
        succeeded=row.succeeded,
        num_attempts=row.num_attempts,
        latency_ms=row.latency_ms,
        attempts=[
            RetrievalAttemptOut(
                attempt_number=a.attempt_number,
                query_used=a.query_used,
                query_analysis=a.query_analysis or {},
                retrieved_chunk_ids=a.retrieved_chunk_ids or [],
                top_score=a.top_score,
                evidence_sufficient=a.evidence_sufficient,
                evidence_confidence=a.evidence_confidence,
                missing_information=a.missing_information,
                decision=a.decision,
            )
            for a in row.attempts
        ],
        citations=citations,
    )
