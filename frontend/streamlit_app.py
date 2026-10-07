"""
AI Enterprise RAG Platform - Streamlit frontend.

A single, polished multi-page Streamlit app that talks to the FastAPI
backend over HTTP. Run with:

    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60

st.set_page_config(
    page_title="AI Enterprise RAG Platform",
    page_icon="\U0001F4DA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2.5rem; max-width: 1200px; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-right: 6px;
    }
    .badge-green { background: #e3f6e8; color: #1a7d3a; }
    .badge-amber { background: #fdf2e0; color: #9a6300; }
    .badge-red { background: #fbe4e4; color: #b3261e; }
    .badge-gray { background: #eef0f3; color: #52606d; }
    .empty-state {
        text-align: center; padding: 3rem 1rem; color: #9ca3af;
    }
    div[data-testid="stMetric"] {
        background-color: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def api_get(path: str, **kwargs) -> Optional[Any]:
    try:
        resp = httpx.get(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(
            f"Could not reach the backend at {API_BASE_URL}. "
            "Make sure it's running."
        )
    except httpx.HTTPStatusError as exc:
        st.error(f"API error ({exc.response.status_code}): {exc.response.text}")
    except Exception as exc:
        st.error(f"Unexpected error calling API: {exc}")
    return None


def api_post(path: str, **kwargs) -> Optional[Any]:
    try:
        resp = httpx.post(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(
            f"Could not reach the backend at {API_BASE_URL}. "
            "Make sure it's running."
        )
    except httpx.HTTPStatusError as exc:
        st.error(f"API error ({exc.response.status_code}): {exc.response.text}")
    except Exception as exc:
        st.error(f"Unexpected error calling API: {exc}")
    return None


def api_delete(path: str, **kwargs) -> Optional[Any]:
    try:
        resp = httpx.delete(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(
            f"Could not reach the backend at {API_BASE_URL}. "
            "Make sure it's running."
        )
    except httpx.HTTPStatusError as exc:
        st.error(f"API error ({exc.response.status_code}): {exc.response.text}")
    except Exception as exc:
        st.error(f"Unexpected error calling API: {exc}")
    return None


def status_badge(status: str) -> str:
    mapping = {
        "processed": "badge-green", "AUTO_APPROVED": "badge-green",
        "uploaded": "badge-gray", "processing": "badge-amber",
        "NEEDS_REVIEW": "badge-amber",
        "failed": "badge-red", "FAILED": "badge-red",
    }
    css_class = mapping.get(status, "badge-gray")
    return f'<span class="badge {css_class}">{status}</span>'


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("\U0001F4DA AI Enterprise RAG Platform")
st.sidebar.caption("Hybrid RAG \u00b7 Document Intelligence \u00b7 Agentic Retrieval")

health = api_get("/health")
API_AVAILABLE = health is not None

if API_AVAILABLE:
    mode_label = "\U0001F7E2 Live LLM" if health["llm_mode"] == "live" else "\U0001F7E1 Mock LLM (offline)"
    st.sidebar.markdown(f"**Backend:** {mode_label}")
    db_status = health.get("database", "unknown")
    st.sidebar.caption(
        f"Database: `{health.get('database_engine', 'N/A')}` \u00b7 "
        f"Embeddings: `{health['embedding_backend']}` \u00b7 Reranker: `{health['reranker_backend']}`"
    )
else:
    st.sidebar.warning("Backend unreachable.")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Documents",
        "Ask AI",
        "Agent Trace",
        "Document AI / Invoices",
        "Evaluation",
    ],
)

st.sidebar.divider()
st.sidebar.caption(f"API: {API_BASE_URL}")


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
def render_dashboard():
    st.title("AI Enterprise RAG Platform")
    st.caption(
        "Hybrid RAG + citation verification, multimodal document intelligence, "
        "and agentic retrieval\u2014sharing one document database and retrieval stack."
    )

    # ------------------------------------------------------------------
    # Connection & database status
    # ------------------------------------------------------------------
    health = api_get("/health")
    if health:
        col_db, col_mode, col_ver = st.columns([2, 2, 1])
        with col_db:
            db_ok = health.get("database") == "ok"
            icon = "\u2705" if db_ok else "\u274c"
            st.markdown(
                f"{icon} **Database** {health.get('database_engine', '').upper()}"
            )
        with col_mode:
            st.markdown(f"\U0001F5A6 **LLM mode**: {health.get('llm_mode', 'N/A')}")
        with col_ver:
            st.markdown(f"\U0001F3F7\ufe0f **v{health.get('version', '—')}**")
    else:
        st.warning("Backend is not reachable. Dashboard data is unavailable.")
        return

    st.divider()

    # ------------------------------------------------------------------
    # Metrics (3-column × 2-row grid using live backend data)
    # ------------------------------------------------------------------
    stats = api_get("/dashboard/stats")
    documents = api_get("/documents") or []

    if stats and documents is not None:
        processed_docs = sum(1 for d in documents if d["status"] == "processed")
        pages_indexed = stats.get("pages_indexed", 0)
        chunks_indexed = stats.get("chunks_indexed", 0)
        review_queue = stats.get("review_queue_count", 0)
        queries_processed = stats.get("queries_processed", 0)
        avg_confidence = stats.get("average_confidence", 0.0)

        st.markdown("### Metrics")
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Documents processed", processed_docs)
        r1c2.metric("Pages indexed", pages_indexed)
        r1c3.metric("Chunks indexed", chunks_indexed)

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Review queue", review_queue)
        r2c2.metric("Queries processed", queries_processed)
        r2c3.metric("Avg extraction confidence", f"{avg_confidence:.2f}")
    else:
        _render_empty_state("No dashboard data available.")
        return

    # ------------------------------------------------------------------
    # Recent queries (from the Query model via the API)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Recent queries")
    recent = api_get("/queries/recent?limit=10")
    if recent:
        for q in recent:
            with st.expander(
                f"{q['question'][:80]}..." if len(q.get("question", "")) > 80
                else q.get("question", "Untitled"),
                expanded=False,
            ):
                st.markdown(f"**Q:** {q.get('question', '')}")
                st.markdown(f"**A:** {q.get('answer', '')[:300]}")
                st.caption(
                    f"Mode: {q.get('mode', 'N/A')} "
                    f"\u00b7 Citations: {q.get('num_citations', 0)} "
                    f"\u00b7 Verified: {q.get('num_verified_citations', 0)} "
                    f"\u00b7 {q.get('latency_ms', 0):.0f} ms "
                    f"\u00b7 {q.get('created_at', '')[:19].replace('T', ' ')}"
                )
    else:
        st.info("No queries have been run yet. Ask a question on the Ask AI page.")


def _render_empty_state(message: str):
    st.markdown(
        f'<div class="empty-state">{message}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
def render_documents():
    st.title("Documents")
    st.caption("Upload PDFs and run them through the shared ingestion pipeline.")

    _render_upload_section()

    st.divider()
    st.subheader("Existing documents")
    _render_document_list()


def _render_upload_section():
    with st.form("upload_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
            )
        with col2:
            doc_type = st.selectbox("Document type", ["generic", "invoice"])
        submitted = st.form_submit_button("Upload & Process", type="primary")

    if not submitted or not uploaded_files:
        return

    progress = st.progress(0.0, text="Starting upload...")
    for i, uf in enumerate(uploaded_files):
        progress.progress((i) / len(uploaded_files), text=f"Uploading {uf.name}...")
        files = {"file": (uf.name, uf.getvalue(), "application/pdf")}
        upload_result = api_post(
            "/documents/upload", files=files, data={"doc_type": doc_type}
        )
        if not upload_result:
            progress.empty()
            st.error(f"Upload failed for {uf.name}.")
            return

        doc_id = upload_result["id"]
        progress.progress((i + 0.5) / len(uploaded_files), text=f"Processing {uf.name}...")
        process_result = api_post(
            f"/documents/process?document_id={doc_id}"
        )
        if process_result:
            if process_result.get("status") == "failed" and process_result.get("error_message"):
                st.error(
                    f"**{uf.name}** — failed: {process_result['error_message']}"
                )
            else:
                st.success(
                    f"**{uf.name}**: {process_result['num_pages']} pages, "
                    f"{process_result['num_chunks']} chunks, "
                    f"{process_result['ocr_pages']} OCR page(s), "
                    f"{process_result['tables_found']} table(s) found."
                )
                st.caption(f"Document ID: `{doc_id}`")
        else:
            st.error(f"Processing failed for {uf.name}.")
    progress.progress(1.0, text="Done.")
    st.rerun()


def _render_document_list():
    documents = api_get("/documents") or []
    extractions = api_get("/review-queue") or []

    extraction_by_doc = {
        e["document_id"]: e for e in extractions if e.get("document_id")
    }

    if not documents:
        st.info("No documents uploaded yet. Upload a PDF on the Documents page.")
        return

    for doc in documents:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(
                    f"**{doc['original_filename']}**  \n"
                    f"`{doc['doc_type']}` \u00b7 {doc['num_pages']} page(s)"
                )
                st.caption(
                    f"ID: `{doc['id']}` \u00b7 Created: {doc['created_at'][:19].replace('T', ' ')}"
                )
            with c2:
                st.markdown(status_badge(doc["status"]), unsafe_allow_html=True)
                ext = extraction_by_doc.get(doc["id"])
                if ext:
                    st.markdown(status_badge(ext["status"]), unsafe_allow_html=True)
            with c3:
                if doc.get("processed_at"):
                    st.caption(f"Processed: {doc['processed_at'][:19].replace('T', ' ')}")

            if doc.get("error_message"):
                st.error(doc["error_message"])

            # ------------------------------------------------------------------
            # Delete with two-step confirmation
            # ------------------------------------------------------------------
            doc_id = doc["id"]
            confirm_key = f"_delete_confirm_{doc_id}"
            if st.session_state.get(confirm_key):
                st.warning(
                    f"Are you sure you want to delete "
                    f"**{doc['original_filename']}**? This action cannot be undone."
                )
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes, delete", key=f"_delete_yes_{doc_id}", type="primary"):
                        _delete_document(doc_id, doc["original_filename"])
                        st.session_state[confirm_key] = False
                        st.rerun()
                with col_no:
                    if st.button("Cancel", key=f"_delete_cancel_{doc_id}"):
                        st.session_state[confirm_key] = False
                        st.rerun()
            else:
                if st.button("Delete", key=f"_delete_btn_{doc_id}"):
                    st.session_state[confirm_key] = True
                    st.rerun()


def _delete_document(doc_id: str, original_filename: str) -> None:
    """Call the DELETE endpoint and show feedback."""
    result = api_delete(f"/documents/{doc_id}")
    if result:
        st.success(f"Deleted **{original_filename}** successfully.")


NO_EVIDENCE_ANSWER = (
    "I couldn't find sufficient evidence in the available documents "
    "to answer this reliably."
)


def render_ask_ai():
    st.title("Ask AI")
    st.caption(
        "Ask questions about your documents. Every claim is grounded to "
        "verified citations with transparent confidence scoring."
    )

    if not API_AVAILABLE:
        st.warning("Backend is not reachable. Ask AI requires a running backend.")
        return

    # ------------------------------------------------------------------
    # Question input
    # ------------------------------------------------------------------
    default_question = st.session_state.get("ask_ai_question", "")
    question = st.text_area(
        "Question",
        value=default_question,
        placeholder="e.g. How many days can employees work remotely?",
        height=120,
        key="ask_ai_input",
    )
    ask = st.button("Ask AI", type="primary", disabled=not question.strip())

    if not ask:
        if not default_question:
            st.info("Enter a question and click **Ask AI** to get started.")
        return

    # ------------------------------------------------------------------
    # Submit & loading
    # ------------------------------------------------------------------
    with st.spinner("Retrieving evidence and generating an answer..."):
        result = api_post("/query", json={"question": question, "top_k": 10})

    if not result:
        return

    answer = result.get("answer", "")
    is_refusal = answer.strip() == NO_EVIDENCE_ANSWER or (
        "insufficient evidence" in answer.lower() and "couldn't find" in answer.lower()
    )

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------
    if is_refusal:
        st.warning(answer)
    else:
        st.markdown("### Answer")
        st.write(answer)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    st.divider()
    st.caption(
        f"\U0001F50D {result.get('retrieved_chunks', 'N/A')} chunks retrieved "
        f"\u00b7 \u23F1 {result.get('latency_ms', 0):.0f} ms "
        f"\u00b7 LLM: {result.get('llm_mode', 'N/A')}"
    )

    # ------------------------------------------------------------------
    # Citations / evidence
    # ------------------------------------------------------------------
    citations = result.get("citations", [])
    if not citations:
        st.info("No citations were returned for this query.")
        return

    st.subheader("Supporting evidence")
    for i, c in enumerate(citations, 1):
        verified = c.get("supported", False)
        verify_label = "\u2705 Verified" if verified else "\u26A0\uFE0F Needs review"
        verify_class = "badge-green" if verified else "badge-amber"

        with st.container(border=True):
            st.markdown(f"**{i}.** {c.get('document', 'Unknown document')}")
            meta_cols = st.columns([1, 1, 1, 2])
            with meta_cols[0]:
                st.caption(f"Page {c.get('page', 'N/A')}")
            with meta_cols[1]:
                st.caption(f"Chunk: {c.get('chunk_id') or 'N/A'}")
            with meta_cols[2]:
                st.markdown(
                    f'<span class="badge {verify_class}">{verify_label}</span>',
                    unsafe_allow_html=True,
                )
            with meta_cols[3]:
                st.markdown(
                    f"Confidence: **{c.get('confidence', 0.0):.2f}**"
                )

            st.markdown("**Claim**: " + (c.get("claim_text") or "—"))
            st.markdown("**Evidence**: " + (c.get("excerpt") or "—"))

            if c.get("reason"):
                st.caption(c["reason"])


def render_agent_trace():
    st.title("Agent Trace")
    st.caption(
        "Execution trace for an agentic retrieval run. Enter an Agent Run ID "
        "to inspect the step-by-step retrieval decisions, evidence sufficiency, "
        "and verified citations."
    )

    if not API_AVAILABLE:
        st.warning("Backend is not reachable. Agent trace requires a running backend.")
        return

    # ------------------------------------------------------------------
    # Run a new agent query (so users don't need to use curl/Postman)
    # ------------------------------------------------------------------
    st.subheader("Run a new agent query")
    default_question = st.session_state.get("agent_question", "")
    agent_question = st.text_area(
        "Agent question",
        placeholder="e.g. What is the eligibility requirement for requesting remote work?",
        height=100,
        key="agent_question_input",
        value=default_question,
    )
    run_agent = st.button("Run Agent", type="primary", disabled=not agent_question.strip())

    if run_agent:
        with st.spinner("Running agentic retrieval..."):
            result = api_post("/agent/query", json={"question": agent_question})
        if result:
            run_id = result.get("id", "")
            st.session_state["agent_trace_run_id"] = run_id
            st.session_state["ask_ai_question"] = agent_question
            st.success(f"Agent run **{run_id[:12]}…** created. Use it below.")
            st.rerun()
        elif result is not None:
            st.error("Agent query failed. See details above.")
        return

    # ------------------------------------------------------------------
    # Retrieve existing trace
    # ------------------------------------------------------------------
    run_id = st.text_input(
        "Agent Run ID",
        placeholder="e.g. a1b2c3d4e5f6...",
        help="Paste the run ID returned by the /agent/query endpoint.",
        value=st.session_state.get("agent_trace_run_id", ""),
    )
    retrieve = st.button("Retrieve trace", type="primary", disabled=not run_id.strip())

    if not retrieve:
        if not st.session_state.get("agent_trace_run_id"):
            st.info("Run an agent query above or enter a known Agent Run ID and click **Retrieve trace**.")
        return

    with st.spinner("Fetching agent trace..."):
        trace = api_get(f"/agent/trace/{run_id.strip()}")

    if not trace:
        return

    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------
    succeeded = trace.get("succeeded", False)
    outcome = "\u2705 Succeeded" if succeeded else "\u26A0\uFE0F Insufficient evidence"
    st.subheader(f"Execution Trace — {outcome}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Run ID**: `{trace.get('id', 'N/A')}`")
    with c2:
        st.markdown(f"**Attempts**: {trace.get('num_attempts', 0)}")
    with c3:
        st.markdown(f"**Latency**: {trace.get('latency_ms', 0):.0f} ms")

    st.markdown("#### Question")
    st.write(trace.get("question", ""))

    st.markdown("#### Final answer")
    final_answer = trace.get("final_answer")
    if final_answer:
        st.write(final_answer)
    else:
        st.markdown("*No answer was generated.*")

    st.markdown(f"**LLM mode**: `{trace.get('llm_mode', 'N/A')}`")

    # ------------------------------------------------------------------
    # Attempts
    # ------------------------------------------------------------------
    attempts = trace.get("attempts", [])
    if not attempts:
        st.info("No retrieval attempts were recorded for this run.")
        return

    st.divider()
    st.subheader("Retrieval attempts")

    for attempt in attempts:
        decision = attempt.get("decision", "unknown")
        decision_labels = {
            "answer": "\U0001F5A6 Answer",
            "reformulate": "\U0001F504 Reformulate",
            "give_up": "\u274C Give up",
        }
        decision_label = decision_labels.get(decision, decision)

        with st.expander(
            f"Attempt {attempt.get('attempt_number', '?')}: {decision_label} "
            f"\u2014 query: {attempt.get('query_used', '')[:80]}",
            expanded=attempt.get("attempt_number", 0) == len(attempts),
        ):
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                st.metric(
                    "Evidence sufficient",
                    "Yes" if attempt.get("evidence_sufficient") else "No",
                )
            with ac2:
                st.metric(
                    "Evidence confidence",
                    f"{attempt.get('evidence_confidence', 0.0):.2f}",
                )
            with ac3:
                st.metric("Top score", f"{attempt.get('top_score', 0.0):.3f}")
            with ac4:
                st.markdown(f"**Decision**: {decision_label}")

            # Query analysis
            analysis = attempt.get("query_analysis") or {}
            if analysis:
                with st.expander("Query analysis", expanded=False):
                    st.json(analysis)

            # Retrieved evidence
            chunk_ids = attempt.get("retrieved_chunk_ids", [])
            with st.expander(
                f"Retrieved evidence ({len(chunk_ids)} chunk(s))", expanded=False
            ):
                if chunk_ids:
                    for cid in chunk_ids:
                        st.markdown(f"- `{cid}`")
                    st.caption(
                        "Per-chunk document/page metadata is not available in the "
                        "trace API response. Use the Citations section below for "
                        "verified source references."
                    )
                else:
                    st.info("No chunks were retrieved in this attempt.")

            # Missing information
            missing = attempt.get("missing_information")
            if missing:
                st.markdown(f"**Missing information**: {missing}")

    # ------------------------------------------------------------------
    # Final citations
    # ------------------------------------------------------------------
    citations = trace.get("citations", [])
    if citations:
        st.divider()
        st.subheader("Verified citations")
        for i, c in enumerate(citations, 1):
            verified = c.get("supported", False)
            badge_class = "badge-green" if verified else "badge-amber"
            badge_label = "\u2705 Verified" if verified else "\u26A0\uFE0F Needs review"
            with st.container(border=True):
                st.markdown(f"**{i}.** {c.get('document', 'Unknown document')}")
                ccol1, ccol2, ccol3 = st.columns([1, 1, 2])
                with ccol1:
                    st.caption(f"Page {c.get('page', 'N/A')}")
                with ccol2:
                    st.markdown(
                        f'<span class="badge {badge_class}">{badge_label}</span>',
                        unsafe_allow_html=True,
                    )
                with ccol3:
                    st.markdown(
                        f"Confidence: **{c.get('confidence', 0.0):.2f}**"
                    )
                st.markdown(f"**Claim**: {c.get('claim_text') or '\u2014'}")
                st.markdown(f"**Evidence**: {c.get('excerpt') or '\u2014'}")
                if c.get("reason"):
                    st.caption(c["reason"])
    else:
        st.divider()
        st.info("No verified citations were attached to this run.")


INVOICE_FIELD_LABELS = {
    "vendor_name": "Vendor",
    "invoice_number": "Invoice number",
    "invoice_date": "Invoice date",
    "currency": "Currency",
    "line_items": "Line items",
    "subtotal": "Subtotal",
    "discount": "Discount",
    "tax": "Tax",
    "shipping": "Shipping",
    "fees": "Fees",
    "total": "Total",
}

INVOICE_FIELD_ORDER = [
    "vendor_name", "invoice_number", "invoice_date", "currency",
    "line_items", "subtotal", "discount", "tax", "shipping", "fees", "total",
]


def render_document_ai():
    st.title("Document AI / Invoices")
    st.caption(
        "Structured extraction, validation, and field-level evidence "
        "grounding for invoice documents (Module 2)."
    )

    if not API_AVAILABLE:
        st.warning("Backend is not reachable. Document AI requires a running backend.")
        return

    _render_invoice_upload()

    st.divider()
    st.subheader("Extractions needing review")
    _render_review_queue()


def _render_invoice_upload():
    st.subheader("Upload invoice")
    with st.form("invoice_upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "PDF invoice", type=["pdf"], accept_multiple_files=False, key="invoice_upload"
        )
        submitted = st.form_submit_button("Upload & Process", type="primary")

    if not submitted or not uploaded_file:
        return

    with st.spinner("Uploading and processing invoice..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        upload_result = api_post("/documents/upload", files=files, data={"doc_type": "invoice"})

    if not upload_result:
        return

    doc_id = upload_result["id"]
    with st.spinner("Running extraction pipeline..."):
        process_result = api_post(f"/documents/process?document_id={doc_id}")

    if not process_result:
        return

    if process_result.get("status") == "failed" and process_result.get("error_message"):
        st.error(f"**{uploaded_file.name}** — failed: {process_result['error_message']}")
        return

    st.success(
        f"**{uploaded_file.name}**: {process_result['num_pages']} pages, "
        f"{process_result['num_chunks']} chunks, "
        f"{process_result['ocr_pages']} OCR page(s), "
        f"{process_result['tables_found']} table(s) found."
    )
    st.caption(f"Document ID: `{doc_id}`")

    extraction_id = process_result.get("extraction_id")
    if extraction_id:
        with st.spinner("Fetching extraction results..."):
            extraction = api_get(f"/extractions/{extraction_id}")
        if extraction:
            _render_extraction_detail(extraction, key_prefix="upload")
        else:
            st.warning("Extraction was created but could not be retrieved.")
    else:
        st.info("No extraction was generated for this document.")


def _render_review_queue():
    extractions = api_get("/review-queue") or []
    if not extractions:
        st.info("No invoice extractions need review right now.")
        return

    for idx, ext in enumerate(extractions):
        with st.expander(
            f"Invoice extraction `{ext['id'][:12]}` \u2014 {status_badge(ext['status'])}",
            expanded=False,
        ):
            _render_extraction_detail(ext, key_prefix=f"queue_{idx}")


def _render_extraction_detail(extraction: dict, key_prefix: str = "default"):
    fields = extraction.get("fields", {})
    status = extraction.get("status", "")
    valid = extraction.get("valid", False)
    errors = extraction.get("validation_errors", [])
    warnings = extraction.get("validation_warnings", [])
    confidence = extraction.get("confidence_score", 0.0)

    # ------------------------------------------------------------------
    # Extraction fields
    # ------------------------------------------------------------------
    st.markdown("#### Extracted fields")
    for key in INVOICE_FIELD_ORDER:
        label = INVOICE_FIELD_LABELS.get(key, key)
        value = fields.get(key)
        if key == "line_items" and value:
            with st.expander(f"{label} ({len(value)} item(s))", expanded=False):
                for j, item in enumerate(value, 1):
                    st.markdown(f"- **Item {j}**: {item}")
        elif key == "due_date":
            st.markdown(f"**{label}**: N/A")
        elif value is not None and value != "":
            display = _format_field_value(key, value)
            st.markdown(f"**{label}**: {display}")
        else:
            st.markdown(f"**{label}**: N/A")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    st.markdown("#### Validation")
    v_status = "UNKNOWN"
    v_class = "badge-gray"
    if valid and not errors:
        if warnings:
            v_status = "WARNING"
            v_class = "badge-amber"
        else:
            v_status = "VALID"
            v_class = "badge-green"
    elif errors:
        v_status = "INVALID"
        v_class = "badge-red"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'<span class="badge {v_class}">{v_status}</span>', unsafe_allow_html=True)
        st.markdown(f"Confidence: **{confidence:.2f}**")
    with col2:
        breakdown = extraction.get("confidence_breakdown", {})
        if breakdown:
            breakdown_cols = st.columns(len(breakdown))
            for i, (bk, bv) in enumerate(breakdown.items()):
                breakdown_cols[i].caption(f"{bk}: {bv:.2f}")

    if errors:
        st.error("Validation errors:\n" + "\n".join(f"- {e}" for e in errors))
    if warnings:
        st.warning("Validation warnings:\n" + "\n".join(f"- {w}" for w in warnings))
    if not errors and not warnings:
        st.success("No validation issues.")

    # ------------------------------------------------------------------
    # Field evidence
    # ------------------------------------------------------------------
    evidence_list = extraction.get("field_evidence", [])
    if evidence_list:
        st.markdown("#### Field evidence")
        with st.expander(f"Show {len(evidence_list)} evidence item(s)", expanded=False):
            for fe in evidence_list:
                st.markdown(
                    f"**{fe.get('field_name', 'N/A')}** "
                    f"\u2014 page {fe.get('page_number') or 'N/A'} "
                    f"\u2014 confidence {fe.get('confidence', 0.0):.2f}"
                )
                if fe.get("source_text"):
                    st.markdown(f"_{fe['source_text']}_")
                st.caption(f"Source: {fe.get('source_text', '')}")
    else:
        st.info("No field evidence available for this extraction.")

    # ------------------------------------------------------------------
    # Review workflow (only for NEEDS_REVIEW / FAILED)
    # ------------------------------------------------------------------
    if status in ("NEEDS_REVIEW", "FAILED"):
        _render_review_form(extraction, extraction_id=extraction.get("id"), key_prefix=key_prefix)
    elif status == "AUTO_APPROVED":
        st.success("This extraction has been auto-approved.")
    else:
        st.caption(f"Status: {status_badge(status)}", unsafe_allow_html=True)


def _format_field_value(field_name: str, value) -> str:
    """Render a field value for display without modifying it."""
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def _render_review_form(extraction: dict, extraction_id: str, key_prefix: str = "default"):
    st.markdown("#### Human review")
    fields = dict(extraction.get("fields", {}))

    with st.form(f"review_form_{key_prefix}_{extraction_id}"):
        notes = st.text_input("Reviewer notes", key=f"notes_{key_prefix}_{extraction_id}")
        st.markdown("**Edit fields (leave blank to keep original)**")
        edited: dict = {}
        numeric_fields = {"subtotal", "discount", "tax", "shipping", "fees", "total"}
        for key, value in fields.items():
            label = INVOICE_FIELD_LABELS.get(key, key)
            default = "" if value is None else str(value)
            if key == "line_items":
                continue
            new_val = st.text_input(label, value=default, key=f"edit_{key_prefix}_{extraction_id}_{key}")
            if new_val != default and new_val.strip():
                if key in numeric_fields:
                    try:
                        edited[key] = float(new_val)
                    except ValueError:
                        st.warning(f"Could not parse '{new_val}' as a number for {label}.")
                        edited[key] = new_val
                else:
                    edited[key] = new_val

        b1, b2, b3 = st.columns(3)
        approve = b1.form_submit_button("\u2705 Approve", use_container_width=True, key=f"approve_{key_prefix}_{extraction_id}")
        save_edit = b2.form_submit_button("\U0001F4BE Save edits", use_container_width=True, key=f"saveedit_{key_prefix}_{extraction_id}")
        reject = b3.form_submit_button("\u274C Reject", use_container_width=True, key=f"reject_{key_prefix}_{extraction_id}")

        action = None
        if approve:
            action = "approve"
        elif reject:
            action = "reject"
        elif save_edit:
            action = "edit"

        if action:
            payload: dict = {"action": action, "notes": notes}
            if action == "edit" and edited:
                payload["edited_fields"] = edited
            with st.spinner(f"Submitting review..."):
                review_result = api_post(
                    f"/review/{extraction_id}", json=payload
                )
            if review_result:
                st.success(f"Review saved: {action}")
                st.rerun()
            else:
                st.error("Failed to save review.")


def render_evaluation():
    st.title("Evaluation")
    st.caption("Run a reproducible evaluation benchmark against the current corpus.")
    ev = api_get("/evaluation")
    if ev and ev.get("status") == "ok":
        st.json(ev["results"])
    elif ev and ev.get("status") == "not_run":
        st.info(
            "No evaluation results yet. Run the following command from the "
            "project root to generate metrics:\n\n"
            "`python -m evaluation.runner`"
        )
    else:
        st.warning("Could not retrieve evaluation status from the backend.")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
PAGES = {
    "Dashboard": render_dashboard,
    "Documents": render_documents,
    "Ask AI": render_ask_ai,
    "Agent Trace": render_agent_trace,
    "Document AI / Invoices": render_document_ai,
    "Evaluation": render_evaluation,
}

PAGES[page]()
