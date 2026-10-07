"""
Evaluation script.

Runs a real (>=20 question) evaluation dataset against the live pipeline
and computes actual metrics for all three modules. Never hardcodes
results - every number here comes from an actual run against whatever
documents are currently ingested.

Usage:
    python scripts/ingest.py          # make sure sample docs are loaded
    python scripts/evaluate.py

Writes results to data/evaluation_results.json (read by GET /evaluation).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.database import init_db, session_scope
from app.document_ai.confidence import compute_confidence
from app.document_ai.extraction import extract_invoice_fields
from app.document_ai.validation import validate_invoice_fields
from app.models import Chunk, Document
from app.rag.agent import run_agentic_query
from app.rag.generator import answer_question

RESULTS_PATH = Path(settings.upload_dir).parent / "evaluation_results.json"

# ---------------------------------------------------------------------
# Module 1 & 3 evaluation dataset (>=20 questions across 4 categories).
# This is a genuine test set for the synthetic sample_documents/ corpus.
# ---------------------------------------------------------------------
EVAL_QUESTIONS = [
    # -- simple --------------------------------------------------------
    {"q": "How many days of PTO do full-time employees accrue per year?", "category": "simple"},
    {"q": "How many paid sick days do employees receive per year?", "category": "simple"},
    {"q": "What is Northwind Insights priced at for the Starter tier?", "category": "simple"},
    {"q": "Which data warehouses does Northwind Insights support?", "category": "simple"},
    {"q": "What percent did Q2 2026 revenue increase compared to Q1 2026?", "category": "simple"},
    # -- multi-hop -------------------------------------------------------
    {"q": "What was Q2 2026 revenue and how does that compare to Q1 2026?", "category": "multi-hop"},
    {"q": "Compare Financial Services and Healthcare revenue growth in Q2 2026.", "category": "multi-hop"},
    {"q": "How does the remote work equipment stipend relate to eligibility requirements?", "category": "multi-hop"},
    {"q": "How do the sick leave and PTO carryover rules relate to each other?", "category": "multi-hop"},
    {"q": "Compare the Starter and Professional pricing tiers of Northwind Insights.", "category": "multi-hop"},
    # -- weak evidence (partially covered / ambiguous) --------------------
    {"q": "What is the company's dress code policy?", "category": "weak_evidence"},
    {"q": "What was the profit margin in Q2 2026?", "category": "weak_evidence"},
    {"q": "How many employees does Northwind Analytics have?", "category": "weak_evidence"},
    {"q": "What security certifications does Northwind Insights hold?", "category": "weak_evidence"},
    {"q": "What is the average tenure of Northwind employees?", "category": "weak_evidence"},
    # -- answer not present in corpus ------------------------------------
    {"q": "What is the CEO's favorite color?", "category": "not_present"},
    {"q": "What was the company's revenue in 2015?", "category": "not_present"},
    {"q": "Who won the Q2 2026 employee of the quarter award?", "category": "not_present"},
    {"q": "What is the office parking policy?", "category": "not_present"},
    {"q": "What programming languages does the engineering team use?", "category": "not_present"},
]

# ---------------------------------------------------------------------
# Module 2 evaluation: known ground truth for the two text-based
# synthetic invoices shipped in sample_documents/.
# ---------------------------------------------------------------------
INVOICE_GROUND_TRUTH = {
    "Synthetic_Invoice_Clean.pdf": {
        "vendor_name": "BrightPath Office Supplies Inc.",
        "invoice_number": "INV-2026-04471",
        "subtotal": 3590.00,
        "tax": 296.18,
        "total": 3886.18,
        "currency": "USD",
    },
    "Synthetic_Invoice_Inconsistent_Total.pdf": {
        "vendor_name": "Lockhart Print & Signage LLC",
        "invoice_number": "INV-2026-00931",
        "subtotal": 1610.00,
        "tax": 132.83,
        "total": 1900.00,  # intentionally inconsistent - validation should flag this
        "currency": "USD",
    },
}


def ensure_corpus_ingested():
    with session_scope() as db:
        chunk_count = db.query(Chunk).count()
    if chunk_count > 0:
        return
    print("No chunks found in the database - ingesting sample_documents/ first...")
    import subprocess

    subprocess.run([sys.executable, str(Path(__file__).parent / "ingest.py")], check=True)


def evaluate_module1_and_3():
    module1_latencies = []
    module1_hit = 0
    total_citations = 0
    supported_citations = 0

    agent_first_pass = 0
    agent_recovered = 0
    agent_failed = 0
    agent_attempts_total = 0
    insufficient_correctly_detected = 0
    not_present_count = 0

    per_question_results = []

    with session_scope() as db:
        for item in EVAL_QUESTIONS:
            question, category = item["q"], item["category"]

            start = time.perf_counter()
            rag_result = answer_question(db, question)
            latency = (time.perf_counter() - start) * 1000
            module1_latencies.append(latency)
            if rag_result.retrieved_chunks > 0:
                module1_hit += 1
            total_citations += len(rag_result.citations)
            supported_citations += sum(1 for c in rag_result.citations if c.get("supported"))

            agent_result = run_agentic_query(db, question)
            agent_attempts_total += len(agent_result.attempts)
            if agent_result.succeeded and len(agent_result.attempts) == 1:
                agent_first_pass += 1
            elif agent_result.succeeded and len(agent_result.attempts) > 1:
                agent_recovered += 1
            else:
                agent_failed += 1

            if category == "not_present":
                not_present_count += 1
                if not agent_result.succeeded:
                    insufficient_correctly_detected += 1

            per_question_results.append(
                {
                    "question": question,
                    "category": category,
                    "module1_retrieved_chunks": rag_result.retrieved_chunks,
                    "module1_num_citations": len(rag_result.citations),
                    "module1_latency_ms": round(latency, 2),
                    "agent_succeeded": agent_result.succeeded,
                    "agent_num_attempts": len(agent_result.attempts),
                }
            )

    n = len(EVAL_QUESTIONS)
    module1_metrics = {
        "retrieval_hit_rate": round(module1_hit / n, 3),
        "citation_support_rate": round(supported_citations / total_citations, 3) if total_citations else None,
        "unsupported_citation_count": total_citations - supported_citations,
        "total_citations": total_citations,
        "avg_latency_ms": round(sum(module1_latencies) / n, 2),
        "num_questions": n,
    }

    module3_metrics = {
        "first_pass_success_rate": round(agent_first_pass / n, 3),
        "successful_recovery_rate": round(agent_recovered / n, 3),
        "failure_rate": round(agent_failed / n, 3),
        "average_attempts": round(agent_attempts_total / n, 3),
        "insufficient_evidence_detection_rate": (
            round(insufficient_correctly_detected / not_present_count, 3) if not_present_count else None
        ),
        "num_questions": n,
    }

    return module1_metrics, module3_metrics, per_question_results


def evaluate_module2():
    field_matches = 0
    field_total = 0
    validation_pass = 0
    review_count = 0
    n_invoices = 0
    per_invoice = []

    with session_scope() as db:
        for filename, ground_truth in INVOICE_GROUND_TRUTH.items():
            document = db.query(Document).filter(Document.original_filename == filename).first()
            if document is None:
                continue
            n_invoices += 1
            # Re-run extraction fresh against stored page text for a clean measurement.
            from app.models import Page

            pages = db.query(Page).filter(Page.document_id == document.id).order_by(Page.page_number).all()
            full_text = "\n".join(p.text for p in pages)
            fields = extract_invoice_fields(full_text)
            valid, errors, warnings = validate_invoice_fields(fields)
            score, breakdown, status = compute_confidence(fields, valid, errors, warnings)

            matches = 0
            for key, expected in ground_truth.items():
                field_total += 1
                actual = fields.get(key)
                is_match = (
                    actual is not None
                    and (
                        (isinstance(expected, float) and isinstance(actual, (int, float)) and abs(actual - expected) < 0.01)
                        or (str(actual).strip().lower() == str(expected).strip().lower())
                    )
                )
                if is_match:
                    field_matches += 1
                    matches += 1

            if status != "FAILED":
                validation_pass += 1 if valid else 0
            if status in ("NEEDS_REVIEW", "FAILED"):
                review_count += 1

            per_invoice.append(
                {"filename": filename, "fields_matched": matches, "fields_total": len(ground_truth), "status": status, "valid": valid}
            )

    return {
        "field_extraction_accuracy": round(field_matches / field_total, 3) if field_total else None,
        "validation_success_rate": round(validation_pass / n_invoices, 3) if n_invoices else None,
        "review_rate": round(review_count / n_invoices, 3) if n_invoices else None,
        "num_invoices_evaluated": n_invoices,
        "per_invoice": per_invoice,
    }


def main():
    init_db()
    ensure_corpus_ingested()

    print("Running Module 1 + Module 3 evaluation over", len(EVAL_QUESTIONS), "questions...")
    module1_metrics, module3_metrics, per_question = evaluate_module1_and_3()

    print("Running Module 2 evaluation over synthetic invoices...")
    module2_metrics = evaluate_module2()

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "module_1_hybrid_rag": module1_metrics,
        "module_2_document_intelligence": module2_metrics,
        "module_3_agentic_rag": module3_metrics,
        "per_question_detail": per_question,
        "note": "All metrics are computed live against the current database contents "
        "(see sample_documents/) using the configured embedding/LLM backends "
        "(see /health for current backend). These are NOT scientifically "
        "calibrated benchmarks - they demonstrate the pipeline is functioning "
        "end-to-end on a small synthetic corpus.",
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_PATH}\n")
    print(json.dumps({"module_1_hybrid_rag": module1_metrics, "module_2_document_intelligence": {k: v for k, v in module2_metrics.items() if k != "per_invoice"}, "module_3_agentic_rag": module3_metrics}, indent=2))


if __name__ == "__main__":
    main()
