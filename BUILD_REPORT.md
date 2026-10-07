# Final Build Report

## Final target

This build refactors the original project into a coherent portfolio-grade **AI Enterprise RAG & Document Intelligence Platform**.

The three original product modules are preserved:

1. Hybrid RAG + citation verification
2. Multimodal document intelligence
3. Controlled agentic RAG

The refactor focuses on making the architectural claims technically defensible.

## Major fixes implemented

### Retrieval
- Real PostgreSQL/pgvector model path with a `VECTOR(n)` column and HNSW cosine index creation.
- SQLite brute-force fallback retained for offline development.
- Dense and BM25 retrieval share an authorization-aware corpus filter.
- Dependency-free BM25 fallback added so the test suite can run even when `rank-bm25` is unavailable.

### Evidence and grounding
- Evidence is now a first-class object containing document/page/chunk provenance and retrieval scores.
- Agent evidence evaluation receives the actual retrieved passages rather than only a score and chunk count.
- Generated citations are grounded to retrieved evidence before being accepted.
- Claim-level verification is run as a separate verification pass.
- Unsupported generated claims are removed where possible; otherwise the system refuses the answer.

### Agentic RAG
- Bounded state machine with a hard maximum of three attempts.
- Query analysis and bounded multi-hop planning.
- Reformulation occurs only when evidence is insufficient.
- Agent citations are persisted and returned from `/agent/trace/{id}`.

### Document intelligence
- Tables are converted into first-class retrievable chunks instead of remaining only in page JSON.
- Invoice extraction no longer truncates the source to the first 6,000 characters.
- Financial validation uses `Decimal` arithmetic.
- Invoice validation accounts for discount, tax, shipping and fees when present.
- Confidence now exposes field-level signals in addition to overall confidence.
- OCR failure/no-usable-text is surfaced as a failed/partial processing state instead of silently succeeding.

### Security and storage
- PDF extension, magic-byte, size and page-limit validation.
- Content-addressed SHA-256 document fingerprint.
- Optional Fernet encryption for uploaded documents with environment-provided keys.
- Document ownership and explicit read permissions.
- Authorization is applied before retrieval.
- Lightweight local `ADMIN`/`USER` identity model for portfolio demonstration.
- Configurable CORS.
- Audit events for query and agent operations.
- Runtime databases, uploads, caches and `.env` are excluded from the repository.

## Verification performed in this environment

- Python compilation: passed.
- Automated tests: **46/46 passed**.
- SQLite API smoke test: upload -> process -> query -> agent query passed.
- Encrypted storage smoke test: encrypt -> materialize/decrypt -> byte-for-byte validation passed.
- Sample PDF processing smoke test: normal PDF parsed and indexed successfully.

## Not claimed as verified here

The sandbox used for this build does not have network access and does not provide a running PostgreSQL server. Therefore:

- Real pgvector SQL execution was not run here.
- Real sentence-transformer/cross-encoder model downloads were not run here.
- Real external LLM calls were not run here.

The repository contains the corresponding provider/backend paths and setup instructions, but these should be tested locally before claiming live-provider or PostgreSQL E2E validation in a CV/README.

## Final repository hygiene

The deliverable contains:

- source code
- tests
- sample documents
- architecture/evaluation/security documentation
- `.env.example`
- Docker Compose configuration

It intentionally excludes:

- `.env`
- local databases
- uploaded runtime files
- Python caches
- compiled bytecode
- generated evaluation artifacts

## Final engineering principle

The platform is not presented as hallucination-proof or 100% accurate.

Its intended guarantee is narrower and more defensible:

> **Retrieve evidence, evaluate it, generate from it, independently verify claims, preserve provenance, and refuse when the available evidence is not sufficient.**
