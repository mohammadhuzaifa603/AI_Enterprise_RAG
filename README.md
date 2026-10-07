# AI Enterprise RAG & Document Intelligence Platform

A self-hostable enterprise RAG and document intelligence platform combining **hybrid retrieval, evidence-grounded generation, citation verification, multimodal document processing, structured extraction, and controlled agentic retrieval**.

The system is designed as a serious AI-engineering portfolio project focused on **grounding, traceability, verification, measurable retrieval, and safe failure** rather than simply generating answers from documents.

---

## Why This Project Is Different

This is not a basic "upload a PDF and chat with it" application.

The platform is designed around an explicit **evidence → generation → verification** pipeline:

- Dense semantic retrieval is combined with BM25 lexical retrieval.
- Retrieved evidence is evaluated before generation.
- Answers are generated only from retrieved evidence.
- Generated claims are independently verified against source evidence.
- Citations are grounded to actual document chunks and page provenance.
- Unsupported questions can result in a safe refusal.
- Agentic retrieval is bounded to prevent uncontrolled loops.
- Tables are indexed as first-class retrieval units.
- Scanned documents can use OCR fallback.
- Structured invoice extraction is followed by deterministic financial validation.
- Low-confidence or inconsistent extractions can be routed to human review.

---

# Core Capabilities

## 1. Hybrid RAG + Citation Verification

- Dense semantic retrieval
- BM25 keyword retrieval
- Configurable reranking
- Reciprocal Rank Fusion (RRF)
- Authorization-aware retrieval
- Page-level provenance
- Evidence sufficiency evaluation
- Claim-level citation verification
- Citation grounding against retrieved chunks
- Safe refusal when reliable evidence is unavailable

### RAG Flow

```text
User Question
      |
      v
Query Analysis
      |
      v
Hybrid Retrieval
(Dense + BM25)
      |
      v
Reranking
      |
      v
Evidence Layer
      |
      +--> Evidence Sufficiency
      +--> Provenance
      +--> Claims
      +--> Citations
      |
      v
LLM Generation
      |
      v
Independent Claim Verification
      |
      +---- Supported ----> Answer + Citations
      |
      +---- Unsupported --> Refusal
