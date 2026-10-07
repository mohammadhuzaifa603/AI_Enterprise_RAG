"""
Evaluation runner.

Orchestrates retrieval, answer generation, citation verification, and
agent execution against an evaluation dataset.  Computes aggregate
metrics across all cases and returns a structured result.

The runner keeps evaluation logic separate from production query code:
it calls the existing ``answer_question`` / ``run_agentic_query``
functions and records their outputs, never duplicating retrieval or
generation logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.rag.generator import NO_EVIDENCE_ANSWER, answer_question
from app.rag.agent import run_agentic_query

from evaluation.dataset import EvalCase
from evaluation.metrics import (
    answer_contains_keywords,
    agent_first_pass_success,
    agent_recovery_rate,
    agent_refusal_accuracy,
    citation_precision,
    citation_recall,
    citation_support_rate,
    is_refusal_answer,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    refusal_accuracy,
)


@dataclass
class CaseResult:
    question: str
    category: str
    should_refuse: bool

    # retrieval
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    relevant_chunk_ids: Set[str] = field(default_factory=set)
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    # answer
    answer: str = ""
    answer_correctness: float = 0.0
    is_refused: bool = False
    refusal_acc: float = 0.0

    # citations
    citation_ids: Set[str] = field(default_factory=set)
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_support: Optional[float] = None

    # agent
    agent_succeeded: bool = False
    agent_num_attempts: int = 0
    agent_first_pass: float = 0.0
    agent_recovery: float = 0.0
    agent_refusal_acc: float = 0.0

    # latency
    latency_ms: float = 0.0


@dataclass
class EvaluationResult:
    num_cases: int
    retrieval: Dict[str, float]
    answer_quality: Dict[str, float]
    citations: Dict[str, float]
    refusal: Dict[str, float]
    agent: Dict[str, float]
    overall_latency_ms: float
    per_case: List[CaseResult] = field(default_factory=list)


def _resolve_relevant_chunk_ids(
    db: Session,
    relevant_sources: Optional[List[Dict[str, Any]]],
) -> Set[str]:
    """Resolve ground-truth relevant chunk IDs from stable document/page references.

    Looks up chunks whose document ``original_filename`` and ``page_number``
    match the annotations in *relevant_sources*.  This avoids fragile
    substring matching against PDF-extracted text, which can differ due to
    whitespace, line-breaks, or OCR artefacts.
    """
    if not relevant_sources:
        return set()

    doc_names = {s["document_name"] for s in relevant_sources}
    page_by_doc: Dict[str, set] = {}
    for s in relevant_sources:
        page_by_doc.setdefault(s["document_name"], set()).update(
            s.get("page_numbers", [])
        )

    docs = db.query(Document).filter(Document.original_filename.in_(doc_names)).all()
    relevant_ids: Set[str] = set()
    for doc in docs:
        pages = page_by_doc.get(doc.original_filename, set())
        if pages:
            rows = db.query(Chunk).filter(
                Chunk.document_id == doc.id,
                Chunk.page_number.in_(pages),
            ).all()
        else:
            rows = db.query(Chunk).filter(Chunk.document_id == doc.id).all()
        relevant_ids.update(r.id for r in rows)
    return relevant_ids


def evaluate_case(
    db: Session,
    case: EvalCase,
) -> CaseResult:
    """Run a single evaluation case and compute its metrics."""
    result = CaseResult(
        question=case.question,
        category=case.category,
        should_refuse=case.should_refuse,
    )

    # ------------------------------------------------------------------
    # Module 1: Hybrid RAG (answer_question)
    # ------------------------------------------------------------------
    rag_result = answer_question(db, case.question, top_k=case.top_k)
    result.retrieved_chunk_ids = [
        item.chunk_id for item in rag_result.evidence_items
    ]
    result.answer = rag_result.answer
    result.latency_ms = rag_result.latency_ms

    # Determine ground-truth relevant chunk IDs from stable doc/page refs
    result.relevant_chunk_ids = _resolve_relevant_chunk_ids(
        db, case.relevant_sources
    )

    relevant = result.relevant_chunk_ids
    retrieved = result.retrieved_chunk_ids

    result.recall_at_5 = recall_at_k(relevant, retrieved, 5)
    result.recall_at_10 = recall_at_k(relevant, retrieved, 10)
    result.mrr = mean_reciprocal_rank(relevant, retrieved)
    result.ndcg_at_10 = ndcg_at_k(relevant, retrieved, 10)

    # Answer correctness
    result.answer_correctness = answer_contains_keywords(
        rag_result.answer, case.expected_answer_contains or []
    )

    # Refusal
    result.is_refused = is_refusal_answer(rag_result.answer)
    result.refusal_acc = refusal_accuracy(
        result.is_refused, case.should_refuse
    )

    # Citations
    cited_ids = {
        c.get("chunk_id") for c in rag_result.citations
        if c.get("chunk_id")
    }
    result.citation_ids = cited_ids
    result.citation_precision = citation_precision(cited_ids, relevant)
    result.citation_recall = citation_recall(cited_ids, relevant)
    result.citation_support = citation_support_rate(rag_result.citations)

    # ------------------------------------------------------------------
    # Module 3: Agentic RAG (run_agentic_query)
    # ------------------------------------------------------------------
    agent_result = run_agentic_query(db, case.question)
    result.agent_succeeded = agent_result.succeeded
    result.agent_num_attempts = len(agent_result.attempts)
    result.agent_first_pass = agent_first_pass_success(
        agent_result.succeeded, result.agent_num_attempts
    )
    result.agent_recovery = agent_recovery_rate(
        agent_result.succeeded, result.agent_num_attempts
    )
    result.agent_refusal_acc = agent_refusal_accuracy(
        agent_result.succeeded, case.should_refuse
    )

    return result


def run_evaluation(
    db: Session,
    cases: Sequence[EvalCase],
) -> EvaluationResult:
    """Run all cases and return aggregate metrics."""
    per_case: List[CaseResult] = []
    for case in cases:
        per_case.append(evaluate_case(db, case))

    n = len(per_case)
    if n == 0:
        return EvaluationResult(
            num_cases=0,
            retrieval={},
            answer_quality={},
            citations={},
            refusal={},
            agent={},
            overall_latency_ms=0.0,
        )

    def _avg(attr: str) -> float:
        vals = [getattr(c, attr) for c in per_case]
        return round(sum(vals) / n, 4)

    retrieval = {
        "recall_at_5": _avg("recall_at_5"),
        "recall_at_10": _avg("recall_at_10"),
        "mrr": _avg("mrr"),
        "ndcg_at_10": _avg("ndcg_at_10"),
    }

    answer_quality = {
        "answer_correctness": _avg("answer_correctness"),
        "avg_latency_ms": round(_avg("latency_ms"), 2),
    }

    cited = [c for c in per_case if c.citation_support is not None]
    citation_support_avg = (
        sum(c.citation_support for c in cited) / len(cited)
        if cited
        else 0.0
    )
    citations = {
        "precision": _avg("citation_precision"),
        "recall": _avg("citation_recall"),
        "support_rate": round(citation_support_avg, 4),
    }

    refusal = {
        "accuracy": _avg("refusal_acc"),
    }

    agent = {
        "first_pass_success_rate": _avg("agent_first_pass"),
        "recovery_rate": _avg("agent_recovery"),
        "refusal_accuracy": _avg("agent_refusal_acc"),
        "failure_rate": round(
            sum(1 for c in per_case if not c.agent_succeeded) / n, 4
        ),
        "average_attempts": _avg("agent_num_attempts"),
    }

    return EvaluationResult(
        num_cases=n,
        retrieval=retrieval,
        answer_quality=answer_quality,
        citations=citations,
        refusal=refusal,
        agent=agent,
        overall_latency_ms=round(_avg("latency_ms"), 2),
        per_case=per_case,
    )

if __name__ == "__main__":
    from app.database import SessionLocal
    from evaluation.dataset import EVAL_DATASET

    db = SessionLocal()

    try:
        result = run_evaluation(db, EVAL_DATASET)

        print("\n===== EVALUATION RESULTS =====")
        print(result)
        print("===== END EVALUATION =====")
    finally:
        db.close()