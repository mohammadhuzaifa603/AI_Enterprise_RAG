"""Controlled agentic RAG state machine.

Bounded, auditable loop:
ANALYZE -> PLAN (when needed) -> RETRIEVE -> RERANK -> EVALUATE ACTUAL EVIDENCE
-> GENERATE -> VERIFY CLAIMS, otherwise REFORMULATE and retry (max 3).
"""
from __future__ import annotations
import json, logging, re, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.config import settings
from app.evidence import Evidence, evaluate_evidence
from app.models import Document
from app.rag.citations import build_evidence_items, format_evidence_block, ground_citations
from app.rag.generator import GENERATION_SYSTEM_PROMPT, NO_EVIDENCE_ANSWER
from app.rag.verifier import verify_answer_claims
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.services.llm_client import LLMError, get_llm_client

logger = logging.getLogger(__name__)
QUERY_ANALYSIS_PROMPT = """TASK: query_analysis
Analyze QUESTION. Return JSON: {\"query_type\":\"simple|ambiguous|multi-hop|document-specific\",\"entities\":[],\"keywords\":[],\"dates\":[],\"important_concepts\":[]}"""
PLAN_PROMPT = """TASK: planning
Break a multi-hop document question into at most 3 retrieval sub-questions.
Return JSON: {\"subqueries\":[\"...\"]}"""
REFORMULATION_PROMPT = """TASK: reformulation
Given ORIGINAL_QUESTION and MISSING_INFORMATION, return JSON {\"reformulated_query\":\"...\"}."""


def _extract_json_object(raw: str) -> Any:
    """Extract a valid JSON object from a raw LLM response.

    Handles markdown-wrapped JSON, reasoning text before the JSON,
    and a JSON object nested inside the ``answer`` field.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise json.JSONDecodeError("No JSON object found in response", raw, 0)

@dataclass
class AttemptRecord:
    attempt_number: int
    query_used: str
    query_analysis: Dict
    retrieved_chunk_ids: List[str]
    top_score: float
    evidence_sufficient: bool
    evidence_confidence: float
    missing_information: Optional[str]
    decision: str

@dataclass
class AgentResult:
    question: str
    final_answer: Optional[str]
    succeeded: bool
    attempts: List[AttemptRecord] = field(default_factory=list)
    citations: List[Dict] = field(default_factory=list)
    llm_mode: str = "mock"
    latency_ms: float = 0.0

def _json_call(system: str, user: str, fallback):
    try:
        raw = get_llm_client().complete(system, user, json_mode=True)
        if not raw or not raw.strip():
            raise LLMError("Agent helper returned an empty response.")
        return _extract_json_object(raw)
    except (LLMError, json.JSONDecodeError) as exc:
        logger.warning("Agent helper failed: %s", exc)
        return fallback

def _analyze_query(question: str) -> Dict:
    return _json_call(QUERY_ANALYSIS_PROMPT, f"QUESTION: {question}", {"query_type":"simple","entities":[],"keywords":[],"dates":[],"important_concepts":[]})

def _plan(question: str, analysis: Dict) -> list[str]:
    if analysis.get("query_type") != "multi-hop":
        return [question]
    data = _json_call(PLAN_PROMPT, f"QUESTION: {question}\nANALYSIS: {json.dumps(analysis)}", {"subqueries": [question]})
    subs = [str(x).strip() for x in data.get("subqueries", []) if str(x).strip()][:3]
    return subs or [question]

def _reformulate(original_question: str, missing_information: str) -> str:
    data = _json_call(REFORMULATION_PROMPT, f"ORIGINAL_QUESTION: {original_question}\nMISSING_INFORMATION: {missing_information}", {"reformulated_query": original_question})
    return str(data.get("reformulated_query") or original_question)

def run_agentic_query(db: Session, question: str, user_id: str | None = None) -> AgentResult:
    start = time.perf_counter()
    client = get_llm_client()
    attempts: List[AttemptRecord] = []
    current_query = question
    best_reranked = []
    for attempt_number in range(1, settings.agent_max_attempts + 1):
        analysis = _analyze_query(question)
        subqueries = _plan(current_query, analysis)
        merged = {}
        for subquery in subqueries:
            for candidate in rerank(subquery, hybrid_search(db, subquery, user_id=user_id)):
                merged[candidate.chunk.id] = candidate
        reranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)[:settings.rerank_top_k]
        top_score = reranked[0].score if reranked else 0.0
        doc_ids = {r.chunk.document_id for r in reranked}
        doc_names = {d.id: d.original_filename for d in db.query(Document).filter(Document.id.in_(doc_ids)).all()}
        evidence = [Evidence(r.chunk.id, r.chunk.document_id, doc_names.get(r.chunk.document_id, "Unknown Document"), r.chunk.page_number, r.chunk.text, r.hybrid_score if hasattr(r, "hybrid_score") else 0.0, r.score, getattr(r.chunk, "chunk_type", "text")) for r in reranked]
        decision = evaluate_evidence(question, evidence)
        action = "answer" if decision.sufficient else ("reformulate" if attempt_number < settings.agent_max_attempts else "give_up")
        attempts.append(AttemptRecord(attempt_number, current_query, analysis, [e.chunk_id for e in evidence], top_score, decision.sufficient, decision.confidence, decision.missing_information, action))
        best_reranked = reranked
        if decision.sufficient:
            break
        if attempt_number < settings.agent_max_attempts:
            current_query = _reformulate(question, decision.missing_information or "")

    if not attempts or not attempts[-1].evidence_sufficient or not best_reranked:
        return AgentResult(question, NO_EVIDENCE_ANSWER, False, attempts, [], client.mode, latency_ms=(time.perf_counter() - start) * 1000)

    doc_ids = {r.chunk.document_id for r in best_reranked}
    doc_names = {d.id: d.original_filename for d in db.query(Document).filter(Document.id.in_(doc_ids)).all()}
    evidence_items = build_evidence_items([(r.chunk, doc_names.get(r.chunk.document_id, "Unknown Document")) for r in best_reranked])
    try:
        raw = client.complete(GENERATION_SYSTEM_PROMPT, f"{format_evidence_block(evidence_items)}\n\nQUESTION: {question}", json_mode=True)
        if not raw or not raw.strip():
            raise LLMError("Agent generation returned an empty response.")
        data = _extract_json_object(raw)
        answer = str(data.get("answer", "")).strip() or NO_EVIDENCE_ANSWER
        grounded = ground_citations(data.get("citations", []), evidence_items)
        answer, citations, verified = verify_answer_claims(answer, grounded) if grounded else (NO_EVIDENCE_ANSWER, [], False)
        if not verified:
            return AgentResult(question, NO_EVIDENCE_ANSWER, False, attempts, citations, client.mode, latency_ms=(time.perf_counter() - start) * 1000)
        return AgentResult(question, answer, True, attempts, citations, client.mode, latency_ms=(time.perf_counter() - start) * 1000)
    except (LLMError, json.JSONDecodeError) as exc:
        logger.warning("Agent generation failed: %s", exc)
        return AgentResult(question, NO_EVIDENCE_ANSWER, False, attempts, [], client.mode, latency_ms=(time.perf_counter() - start) * 1000)
