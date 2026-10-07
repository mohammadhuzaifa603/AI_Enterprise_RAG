"""Tests for the evaluation runner.

Uses the ``db_session`` fixture (isolated SQLite) with seeded chunks,
matching the pattern in ``test_retrieval.py`` and ``test_agent.py``.
"""
from evaluation.runner import evaluate_case, run_evaluation, _resolve_relevant_chunk_ids
from evaluation.dataset import EvalCase, EVAL_DATASET
from app.ingestion.embeddings import embed_texts
from app.models import Chunk, Document


def _seed_relevant_chunks(db):
    """Seed a small corpus with known chunk IDs for deterministic tests."""
    doc = Document(
        filename="eval_test.pdf",
        original_filename="eval_test.pdf",
        file_path="/tmp/eval_test.pdf",
        status="processed",
        num_pages=1,
    )
    db.add(doc)
    db.flush()

    texts = [
        "Northwind Analytics Q2 2026 revenue reached 4.82 million dollars, "
        "an increase of 14 percent compared to Q1 2026 which was 4.23 million.",
        "Employees may work remotely up to three days per week under company policy.",
        "Employees accrue twenty days of paid time off per year.",
        "The espresso machine in the kitchen is broken again.",
    ]
    vectors = embed_texts(texts)
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        db.add(Chunk(
            document_id=doc.id,
            page_number=1,
            chunk_index=i,
            text=text,
            embedding=vec,
        ))
    db.flush()
    return doc


# ---------------------------------------------------------------------------
# Ground-truth resolution tests (stable document/page references)
# ---------------------------------------------------------------------------
def test_resolve_relevant_chunk_ids_by_doc_and_page(db_session):
    """Chunks should be resolved by document name + page number, not text."""
    doc = _seed_relevant_chunks(db_session)

    relevant = _resolve_relevant_chunk_ids(
        db_session,
        [{"document_name": "eval_test.pdf", "page_numbers": [1]}],
    )
    # All 4 seeded chunks are on page 1 of eval_test.pdf
    assert len(relevant) == 4
    assert all(isinstance(cid, str) for cid in relevant)


def test_resolve_relevant_chunk_ids_empty_sources(db_session):
    """Empty relevant_sources -> empty set."""
    _seed_relevant_chunks(db_session)
    assert _resolve_relevant_chunk_ids(db_session, []) == set()
    assert _resolve_relevant_chunk_ids(db_session, None) == set()


def test_resolve_relevant_chunk_ids_specific_page(db_session):
    """Only chunks on the specified page should be returned."""
    doc = Document(
        filename="multi_page.pdf",
        original_filename="multi_page.pdf",
        file_path="/tmp/multi_page.pdf",
        status="processed",
        num_pages=3,
    )
    db_session.add(doc)
    db_session.flush()

    vectors = embed_texts(["Page one text", "Page two text", "Page three text"])
    for i, (text, vec) in enumerate(zip(["Page one text", "Page two text", "Page three text"], vectors)):
        db_session.add(Chunk(
            document_id=doc.id,
            page_number=i + 1,
            chunk_index=i,
            text=text,
            embedding=vec,
        ))
    db_session.flush()

    relevant_page2 = _resolve_relevant_chunk_ids(
        db_session,
        [{"document_name": "multi_page.pdf", "page_numbers": [2]}],
    )
    assert len(relevant_page2) == 1
    chunk = db_session.query(Chunk).filter(Chunk.page_number == 2).first()
    assert chunk.id in relevant_page2


# ---------------------------------------------------------------------------
# End-to-end case evaluation tests
# ---------------------------------------------------------------------------
def test_evaluate_case_simple_question(db_session):
    _seed_relevant_chunks(db_session)
    case = EvalCase(
        question="What was Q2 2026 revenue compared to Q1?",
        category="simple",
        expected_answer_contains=["4.82", "million"],
        relevant_sources=[{"document_name": "eval_test.pdf", "page_numbers": [1]}],
        should_refuse=False,
    )
    result = evaluate_case(db_session, case)
    assert result.question == case.question
    assert len(result.retrieved_chunk_ids) > 0
    assert len(result.relevant_chunk_ids) > 0
    assert result.recall_at_5 > 0.0
    assert result.recall_at_10 > 0.0
    assert result.mrr > 0.0
    assert result.ndcg_at_10 > 0.0
    assert result.answer_correctness > 0.0
    assert result.is_refused is False
    assert result.refusal_acc == 1.0


def test_evaluate_case_not_present_question(db_session):
    _seed_relevant_chunks(db_session)
    case = EvalCase(
        question="What is the CEO's favorite color?",
        category="not_present",
        expected_answer_contains=[],
        relevant_sources=[],
        should_refuse=True,
    )
    result = evaluate_case(db_session, case)
    assert result.is_refused is True
    assert result.refusal_acc == 1.0
    assert result.relevant_chunk_ids == set()
    assert result.recall_at_5 == 0.0
    assert result.mrr == 0.0


def test_evaluate_case_answer_contains_keywords(db_session):
    _seed_relevant_chunks(db_session)
    case = EvalCase(
        question="What was Q2 2026 revenue compared to Q1?",
        category="simple",
        expected_answer_contains=["4.82", "million"],
        relevant_sources=[{"document_name": "eval_test.pdf", "page_numbers": [1]}],
        should_refuse=False,
    )
    result = evaluate_case(db_session, case)
    # The mock LLM should answer with keywords from the retrieved evidence.
    assert result.answer_correctness > 0.0


def test_run_evaluation_aggregates_metrics(db_session):
    _seed_relevant_chunks(db_session)
    cases = [
        EvalCase(
            question="What was Q2 2026 revenue compared to Q1?",
            category="simple",
            expected_answer_contains=["4.82", "million"],
            relevant_sources=[{"document_name": "eval_test.pdf", "page_numbers": [1]}],
            should_refuse=False,
        ),
        EvalCase(
            question="What is the CEO's favorite color?",
            category="not_present",
            expected_answer_contains=[],
            relevant_sources=[],
            should_refuse=True,
        ),
    ]
    result = run_evaluation(db_session, cases)
    assert result.num_cases == 2
    assert "recall_at_5" in result.retrieval
    assert "answer_correctness" in result.answer_quality
    assert "precision" in result.citations
    assert "accuracy" in result.refusal
    assert "first_pass_success_rate" in result.agent
    assert result.overall_latency_ms > 0.0
    assert len(result.per_case) == 2


def test_run_evaluation_empty_cases(db_session):
    result = run_evaluation(db_session, [])
    assert result.num_cases == 0


def test_evaluate_case_includes_agent_metrics(db_session):
    _seed_relevant_chunks(db_session)
    case = EvalCase(
        question="What was Q2 2026 revenue compared to Q1?",
        category="simple",
        expected_answer_contains=["4.82", "million"],
        relevant_sources=[{"document_name": "eval_test.pdf", "page_numbers": [1]}],
        should_refuse=False,
    )
    result = evaluate_case(db_session, case)
    assert result.agent_num_attempts >= 1
    assert result.agent_first_pass in (0.0, 1.0)
    assert result.agent_refusal_acc in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset integrity tests
# ---------------------------------------------------------------------------
def test_eval_dataset_has_16_cases():
    assert len(EVAL_DATASET) == 16


def test_eval_dataset_cases_have_required_fields():
    for case in EVAL_DATASET:
        assert case.question
        assert case.category in ("simple", "multi-hop", "weak_evidence", "not_present")
        assert isinstance(case.should_refuse, bool)
        assert isinstance(case.relevant_sources, list) or case.relevant_sources is None


def test_eval_dataset_refusal_cases_have_no_relevant_sources():
    """Cases marked should_refuse=True must have empty relevant_sources."""
    for case in EVAL_DATASET:
        if case.should_refuse:
            assert not case.relevant_sources or len(case.relevant_sources) == 0


def test_eval_dataset_answerable_cases_have_relevant_sources():
    """Cases marked should_refuse=False must have relevant_sources."""
    for case in EVAL_DATASET:
        if not case.should_refuse:
            assert case.relevant_sources and len(case.relevant_sources) > 0