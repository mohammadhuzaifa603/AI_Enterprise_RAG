# Screenshots

This build environment is a headless container with no browser available,
so automatic screenshots could not be captured. The application was
verified to run correctly (FastAPI backend + Streamlit frontend, both
tested via HTTP requests — see `BUILD_REPORT.md`), but no image files
were generated here to avoid claiming a screenshot exists when it doesn't.

To capture real screenshots once you run the app locally
(`uvicorn app.main:app --reload` + `streamlit run frontend/streamlit_app.py`),
grab these 6 screens and drop them into this folder with these filenames
(the README doesn't hard-link them, but these names keep things tidy):

1. **01_dashboard.png** — the Dashboard page (`http://localhost:8501`, "Dashboard" in the sidebar) after running `python scripts/ingest.py` and `python scripts/evaluate.py`, showing the live metric cards and evaluation tabs.
2. **02_document_upload.png** — the Documents page mid-upload or just after processing a PDF, showing the per-file success message and the document list with status badges.
3. **03_document_intelligence.png** — the Document Intelligence page with an invoice expanded, showing extracted fields, confidence breakdown, and any validation warnings.
4. **04_review_queue.png** — the Review Queue page with `Synthetic_Invoice_Inconsistent_Total.pdf`'s extraction open, showing the validation error and the Approve / Save edits / Reject actions.
5. **05_chat_citations.png** — the Ask Documents page after asking *"How many days per week can employees work remotely?"*, showing the answer and the ✓ Verified citation card.
6. **06_agent_trace.png** — the Agent Trace page after running *"What was Q2 2026 revenue and how does it compare to Q1?"*, showing the attempt-by-attempt expandable trace.
