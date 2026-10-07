# AI Enterprise RAG & Document Intelligence Platform

A self-hostable enterprise RAG and document intelligence platform for **grounded question answering, document intelligence, and controlled agentic retrieval**.

The project is designed as a serious AI-engineering portfolio system: it keeps document provenance end-to-end, evaluates the actual retrieved evidence before generation, verifies claims after generation, supports scanned PDFs and tables, and refuses when reliable evidence is unavailable.

## Core capabilities

### 1. Hybrid RAG + citation verification
- Dense semantic retrieval + BM25 keyword retrieval
- Configurable reranking
- Authorization-aware retrieval
- Page-level provenance
- Evidence sufficiency evaluation
- Claim-level citation verification
- Safe refusal when claims cannot be grounded

### 2. Multimodal document intelligence
- Native PDF text extraction
- OCR fallback for scanned pages
- Table extraction
- First-class table chunks for retrieval
- Invoice field extraction
- `Decimal`-safe monetary validation
- Field-level confidence signals
- Human review queue

### 3. Controlled agentic RAG
- Query analysis
- Bounded multi-hop planning
- Evidence-driven retrieval loop
- Query reformulation
- Maximum 3 retrieval attempts
- Persistent attempt trace
- Persistent citations/evidence relationships
- Refusal after repeated evidence failure

## Architecture

```text
Streamlit UI
    |
    v
FastAPI API
    |
    v
Application Services
    |
    +-------------------+--------------------+
    |                   |                    |
Document Intelligence  Retrieval          RAG/Agent
    |                   |                    |
Parse/OCR/Tables   Dense + BM25         Plan/Retrieve
Chunk/Embed        Rerank               Evaluate
    |                   |                Generate
    +-------------------+                    |
                |                            |
                v                            v
          Evidence Layer <-------------------+
          |  sufficiency
          |  provenance
          |  claims
          |  verification
          |  citations
                |
                v
      PostgreSQL + pgvector
      SQLite offline fallback
                |
                v
       Private/encrypted storage
```

## Data flow

```text
PDF upload
  -> security validation
  -> private storage
  -> page parsing / OCR
  -> table extraction
  -> text + table chunks
  -> embeddings
  -> vector + BM25 indexes
  -> authorization-aware retrieval
  -> reranking
  -> evidence evaluation
  -> answer generation
  -> claim extraction / citation grounding
  -> independent verification
  -> cited answer OR refusal
  -> audit record
```

## Storage and security

The portfolio build intentionally keeps security focused rather than implementing a full enterprise IAM product.

- Local user/role model (`ADMIN` / `USER`)
- Document ownership and explicit read permissions
- Retrieval is filtered **before** dense/BM25 search
- Upload size, extension, PDF magic-byte and page-limit checks
- Configurable CORS instead of wildcard credentials
- Secrets loaded from environment variables
- Optional Fernet encryption for uploaded documents
- SHA-256 document fingerprinting
- Audit events for uploads and queries

For a real deployment, supply `STORAGE_ENCRYPTION_KEY` from a secret manager/KMS. The repository never contains secrets or runtime databases.

## Database modes

### SQLite
Zero-setup development and deterministic tests. Dense search uses a brute-force Python fallback.

### PostgreSQL + pgvector
Production-style retrieval. The embedding column is a real `VECTOR(n)` column and `init_db()` creates an HNSW cosine index. Dense similarity is executed in PostgreSQL rather than loading the full corpus into Python.

Start PostgreSQL:

```bash
docker compose up -d
```

Then configure:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/enterprise_rag
```

## Running locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env  # macOS/Linux

python scripts/init_db.py
uvicorn app.main:app --reload
```

Streamlit:

```bash
streamlit run frontend/streamlit_app.py
```

## API highlights

- `POST /documents/upload`
- `POST /documents/process?document_id=...`
- `GET /documents`
- `POST /query`
- `POST /agent/query`
- `GET /agent/trace/{agent_run_id}`
- `GET /review-queue`
- `GET /extractions/{extraction_id}`
- `POST /review/{extraction_id}`
- `GET /health`
- `GET /auth/me`

## LLM Provider

The reference configuration uses **Groq with OpenAI GPT-OSS 120B** through Groq's OpenAI-compatible API.

LLM credentials are loaded through environment variables and are never stored in the repository.

## Evaluation

The project is structured to evaluate the actual system rather than only counting retrieved documents.

### Metrics

- Retrieval: Recall@K, MRR, NDCG
- Grounding: supported-claim rate, unsupported-claim rate
- Citations: citation precision, citation recall, verification rate
- Agent: first-pass success, recovery success, correct refusal, average attempts
- Extraction: field precision/recall/F1, validation accuracy

### Automated validation

**153 / 153 tests passing**

The test suite covers retrieval, reranking, evidence sufficiency, citation verification, agent traces, OCR, document ingestion, invoice extraction and validation, API behavior, authorization, document deletion, and evaluation metrics.
Do not publish benchmark numbers unless they were generated from the repository's evaluation dataset and scripts.

## Project scope

This is a portfolio and learning project, not a finished enterprise SaaS product. It intentionally does **not** implement SSO, multi-tenancy, Kubernetes, distributed queues, streaming, or a large-scale ANN service beyond pgvector. The focus is correctness and depth in RAG/document-AI engineering.

## Repository hygiene

Runtime databases, uploaded documents, `.env`, Python caches, and generated evaluation artifacts are excluded from Git. Use `.env.example` as the configuration template.
