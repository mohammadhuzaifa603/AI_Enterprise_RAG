"""
Module 1 pipeline orchestrator: Hybrid RAG + citation verification.

question -> hybrid_search -> rerank -> build evidence -> evidence sufficiency
-> LLM answer -> ground citations -> verify each citation
"""
from __future__ import annotations
import re
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document
from app.rag.citations import (
    EvidenceItem,
    build_evidence_items,
    format_evidence_block,
    ground_citations,
)
from app.rag.verifier import verify_answer_claims
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.services.llm_client import LLMError, get_llm_client
from app.evidence import Evidence, evaluate_evidence

logger = logging.getLogger(__name__)


GENERATION_SYSTEM_PROMPT = """TASK: generation
You are an enterprise document assistant. Answer the user's QUESTION
using ONLY the provided EVIDENCE blocks.

Rules:
- Do NOT use any knowledge outside the EVIDENCE.
- Every factual claim must be supported by the EVIDENCE.
- Cite the evidence chunk that supports each factual claim.
- Use ONLY exact chunk_id values from the EVIDENCE.
- Do NOT invent chunk IDs, documents, or pages.
- For each citation, include the specific factual claim it supports
  in claim_text.
- If the evidence does not contain enough information, return an empty
  answer and an empty citations list.
- Keep the answer concise and directly answer the question.

Respond ONLY with valid JSON.
Do not use markdown.
Do not include any text before or after the JSON.

Format:
{"answer":"concise answer","citations":[{"chunk_id":"exact_chunk_id","claim_text":"specific factual claim supported by this chunk"}]}
"""


@dataclass
class RagResult:
    answer: str
    citations: List[Dict] = field(default_factory=list)
    retrieved_chunks: int = 0
    latency_ms: float = 0.0
    llm_mode: str = "mock"
    evidence_items: List[EvidenceItem] = field(default_factory=list)


NO_EVIDENCE_ANSWER = (
    "I couldn't find sufficient evidence in the available documents "
    "to answer this reliably."
)


def answer_question(
    db: Session,
    question: str,
    top_k: int | None = None,
) -> RagResult:

    start = time.perf_counter()
    top_k = top_k or settings.rerank_top_k

    # ---------------------------------------------------------
    # 1. Hybrid retrieval
    # ---------------------------------------------------------
    candidates = hybrid_search(db, question)
    reranked = rerank(question, candidates, top_k=top_k)

    client = get_llm_client()

    if not reranked:
        return RagResult(
            answer=NO_EVIDENCE_ANSWER,
            citations=[],
            retrieved_chunks=0,
            latency_ms=(time.perf_counter() - start) * 1000,
            llm_mode=client.mode,
        )

    # ---------------------------------------------------------
    # 2. Resolve document names
    # ---------------------------------------------------------
    doc_ids = {r.chunk.document_id for r in reranked}

    documents = (
        db.query(Document)
        .filter(Document.id.in_(doc_ids))
        .all()
    )

    doc_names = {
        d.id: d.original_filename
        for d in documents
    }

    # ---------------------------------------------------------
    # 3. Build evidence pairs
    # ---------------------------------------------------------
    evidence_pairs = [
        (
            r.chunk,
            doc_names.get(
                r.chunk.document_id,
                "Unknown Document",
            ),
        )
        for r in reranked
    ]

    evidence_items = build_evidence_items(evidence_pairs)

    # ---------------------------------------------------------
    # 4. Convert retrieved chunks into Evidence objects
    # ---------------------------------------------------------
    evidence = [
        Evidence(
            chunk.id,
            chunk.document_id,
            doc_name,
            chunk.page_number,
            chunk.text,
            next(
                (
                    r.dense_score
                    for r in reranked
                    if r.chunk.id == chunk.id
                ),
                0.0,
            ),
            next(
                (
                    r.score
                    for r in reranked
                    if r.chunk.id == chunk.id
                ),
                0.0,
            ),
            getattr(
                chunk,
                "chunk_type",
                "text",
            ),
        )
        for chunk, doc_name in evidence_pairs
    ]

    # ---------------------------------------------------------
    # 5. Evidence sufficiency gate
    # ---------------------------------------------------------
    decision = evaluate_evidence(
        question,
        evidence,
    )

    if not decision.sufficient:
        return RagResult(
            answer=NO_EVIDENCE_ANSWER,
            citations=[],
            retrieved_chunks=len(reranked),
            latency_ms=(time.perf_counter() - start) * 1000,
            llm_mode=client.mode,
            evidence_items=evidence_items,
        )

    # ---------------------------------------------------------
    # 6. Build evidence prompt
    # ---------------------------------------------------------
    evidence_block = format_evidence_block(evidence_items)

    user_prompt = (
        f"{evidence_block}\n\n"
        f"QUESTION: {question}"
    )

    # ---------------------------------------------------------
    # 7. Generate answer
    # ---------------------------------------------------------
    try:
        raw = client.complete(
            GENERATION_SYSTEM_PROMPT,
            user_prompt,
            json_mode=True,
        )

        # Some providers can return HTTP 200 with an empty body/content.
        if not raw or not raw.strip():
            raise LLMError(
                "Generation returned an empty response."
            )

        # First attempt: strict JSON parsing.
        try:
            data = json.loads(raw)

        # Fallback: some models prepend reasoning/text before JSON.
        except json.JSONDecodeError:
            match = re.search(
                r"\{.*\}",
                raw,
                re.DOTALL,
            )

            if not match:
                raise

            data = json.loads(match.group(0))

        # Some models may wrap the actual JSON object inside
        # the "answer" field.
        if (
            isinstance(data, dict)
            and isinstance(data.get("answer"), str)
            and data["answer"].lstrip().startswith("{")
        ):
            try:
                nested = json.loads(data["answer"])

                if isinstance(nested, dict):
                    data = nested

            except json.JSONDecodeError:
                pass

        if not isinstance(data, dict):
            raise LLMError(
                "Generation returned JSON that was not an object."
            )

        answer = str(
            data.get("answer", "")
        ).strip()

        raw_citations = data.get(
            "citations",
            [],
        )

        if not isinstance(raw_citations, list):
            raw_citations = []

    except (
        LLMError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        logger.warning(
            "Generation failed: %s",
            exc,
        )

        return RagResult(
            answer=(
                "The answer could not be generated because "
                "the language model returned an invalid response. "
                "Please retry."
            ),
            citations=[],
            retrieved_chunks=len(reranked),
            latency_ms=(
                time.perf_counter() - start
            ) * 1000,
            llm_mode=client.mode,
            evidence_items=evidence_items,
        )

    # If the model explicitly returned no answer, refuse safely.
    if not answer:
        return RagResult(
            answer=NO_EVIDENCE_ANSWER,
            citations=[],
            retrieved_chunks=len(reranked),
            latency_ms=(
                time.perf_counter() - start
            ) * 1000,
            llm_mode=client.mode,
            evidence_items=evidence_items,
        )

    # ---------------------------------------------------------
    # 8. Ground citations against actual evidence
    # ---------------------------------------------------------
    grounded = ground_citations(
        raw_citations,
        evidence_items,
    )

    if not grounded:
        return RagResult(
            answer=NO_EVIDENCE_ANSWER,
            citations=[],
            retrieved_chunks=len(reranked),
            latency_ms=(
                time.perf_counter() - start
            ) * 1000,
            llm_mode=client.mode,
            evidence_items=evidence_items,
        )

    # ---------------------------------------------------------
    # 9. Verify claims
    # ---------------------------------------------------------
    verified_answer, verified_citations, verified = (
        verify_answer_claims(
            answer,
            grounded,
        )
    )

    if not verified:
        verified_answer = (
            "I couldn't verify the generated claims against "
            "the retrieved source evidence, so I won't present "
            "them as reliable facts."
        )

    # ---------------------------------------------------------
    # 10. Return final RAG result
    # ---------------------------------------------------------
    return RagResult(
        answer=verified_answer,
        citations=verified_citations,
        retrieved_chunks=len(reranked),
        latency_ms=(
            time.perf_counter() - start
        ) * 1000,
        llm_mode=client.mode,
        evidence_items=evidence_items,
    )