"""
Deterministic evaluation dataset.

Each case is a self-contained question with ground-truth annotations.
Cases are derived from the actual content of the documents in
``sample_documents/`` (see ``scripts/make_sample_documents.py``).

Annotation fields:

* ``question``                    – the query string.
* ``category``                    – ``"simple"`` | ``"multi-hop"`` |
  ``"weak_evidence"`` | ``"not_present"``.
* ``expected_answer_contains``    – keywords that *should* appear in the
  generated answer (used for answer-correctness scoring).  Answer quality is
  evaluated separately from retrieval relevance.
* ``relevant_sources``            – list of ``{document_name, page_numbers}``
  dicts identifying the documents/pages that contain the answer.  The runner
  resolves these to concrete chunk IDs via the database, avoiding fragile
  substring matching against extracted text.
* ``should_refuse``               – whether the system *should* refuse
  (answer is not supported by the corpus).
* ``top_k``                       – retrieval cutoff for ranking metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class EvalCase:
    question: str
    category: str
    expected_answer_contains: Optional[List[str]] = None
    relevant_sources: Optional[List[Dict[str, Any]]] = None
    should_refuse: bool = False
    top_k: int = 10


# ---------------------------------------------------------------------------
# Dataset — 16 cases covering all four annotation categories.
# Document/page references are derived from the exact content of
# sample_documents/*.pdf (see scripts/make_sample_documents.py).
# ---------------------------------------------------------------------------
EVAL_DATASET: List[EvalCase] = [
    # -- simple (answer directly present) -------------------------------
    EvalCase(
        question="How many days can eligible employees work remotely per week?",
        category="simple",
        expected_answer_contains=["three", "3", "days per week"],
        relevant_sources=[
            {"document_name": "Remote_Work_Policy.pdf", "page_numbers": [1]},
            {"document_name": "Employee_Handbook.pdf", "page_numbers": [2]},
        ],
        should_refuse=False,
    ),
    EvalCase(
        question="How many PTO days do full-time employees accrue per year?",
        category="simple",
        expected_answer_contains=["20", "twenty days"],
        relevant_sources=[
            {"document_name": "Employee_Handbook.pdf", "page_numbers": [3]},
        ],
        should_refuse=False,
    ),
    EvalCase(
        question="What was Q2 2026 revenue?",
        category="simple",
        expected_answer_contains=["4.82", "million"],
        relevant_sources=[
            {"document_name": "Q2_2026_Sales_Report.pdf", "page_numbers": [1]},
        ],
        should_refuse=False,
    ),
    EvalCase(
        question="Which data warehouses does Northwind Insights support?",
        category="simple",
        expected_answer_contains=["Snowflake", "BigQuery", "Redshift", "Postgres"],
        relevant_sources=[
            {"document_name": "Northwind_Insights_Product_Guide.pdf", "page_numbers": [2]},
        ],
        should_refuse=False,
    ),
    EvalCase(
        question="What is the price of the Northwind Insights Starter tier?",
        category="simple",
        expected_answer_contains=["$39", "39"],
        relevant_sources=[
            {"document_name": "Northwind_Insights_Product_Guide.pdf", "page_numbers": [2]},
        ],
        should_refuse=False,
    ),
    EvalCase(
        question="How many sick days do employees receive per year?",
        category="simple",
        expected_answer_contains=["10"],
        relevant_sources=[
            {"document_name": "Employee_Handbook.pdf", "page_numbers": [3]},
        ],
        should_refuse=False,
    ),
    EvalCase(
        question="What is the company's home office stipend for remote workers?",
        category="simple",
        expected_answer_contains=["$300"],
        relevant_sources=[
            {"document_name": "Remote_Work_Policy.pdf", "page_numbers": [2]},
        ],
        should_refuse=False,
    ),
    # -- multi-hop (requires combining information from multiple passages)
    EvalCase(
        question="What was the Q2 2026 revenue and how does it compare to Q1 2026?",
        category="multi-hop",
        expected_answer_contains=["4.82", "million", "14 percent", "4.23"],
        relevant_sources=[
            {"document_name": "Q2_2026_Sales_Report.pdf", "page_numbers": [1]},
        ],
        should_refuse=False,
    ),
    EvalCase(
        question="Compare Financial Services and Healthcare revenue in Q2 2026.",
        category="multi-hop",
        expected_answer_contains=["Financial Services", "Healthcare", "1.8", "1.3"],
        relevant_sources=[
            {"document_name": "Q2_2026_Sales_Report.pdf", "page_numbers": [2]},
        ],
        should_refuse=False,
    ),
    EvalCase(
        question="How does the remote work stipend relate to eligibility?",
        category="multi-hop",
        expected_answer_contains=["90-day", "probationary", "stipend"],
        relevant_sources=[
            {"document_name": "Remote_Work_Policy.pdf", "page_numbers": [1, 2]},
        ],
        should_refuse=False,
    ),
    # -- weak evidence (partially covered / ambiguous) -------------------
    # These questions have no clear answer in any sample document.
    EvalCase(
        question="What is the company's dress code policy?",
        category="weak_evidence",
        expected_answer_contains=[],
        relevant_sources=[],
        should_refuse=True,
    ),
    EvalCase(
        question="What was the profit margin in Q2 2026?",
        category="weak_evidence",
        expected_answer_contains=[],
        relevant_sources=[],
        should_refuse=True,
    ),
    EvalCase(
        question="How many employees does Northwind Analytics have?",
        category="weak_evidence",
        expected_answer_contains=[],
        relevant_sources=[],
        should_refuse=True,
    ),
    # -- answer not present in corpus ------------------------------------
    EvalCase(
        question="What is the CEO's favorite color?",
        category="not_present",
        expected_answer_contains=[],
        relevant_sources=[],
        should_refuse=True,
    ),
    EvalCase(
        question="What was the company's revenue in 2015?",
        category="not_present",
        expected_answer_contains=[],
        relevant_sources=[],
        should_refuse=True,
    ),
    EvalCase(
        question="Who won the Q2 2026 employee of the quarter award?",
        category="not_present",
        expected_answer_contains=[],
        relevant_sources=[],
        should_refuse=True,
    ),
]
