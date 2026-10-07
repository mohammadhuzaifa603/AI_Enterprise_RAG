# Architecture Decision Record

## Core principle

The system treats **evidence as a first-class object**. Retrieval does not directly become an answer.

```text
retrieve -> evaluate evidence -> generate -> verify claims -> answer/refuse
```

## Evidence object

Each evidence record retains:

- document ID/name
- page number
- chunk ID
- chunk type
- source text
- retrieval score
- reranker score

This makes the answer traceable to the exact source passage.

## Retrieval backend abstraction

`DenseBackend` is separated from orchestration so the same application can use:

- SQLite brute-force search for offline development
- PostgreSQL + pgvector HNSW search for production-style retrieval

BM25 remains a separate lexical signal and both candidate sets are fused before reranking.

## Authorization boundary

Authorization is applied before retrieval. A regular user can retrieve only owned or explicitly permitted documents. The LLM never receives unauthorized chunks as a downstream filtering strategy.

## Agent boundary

The agent is a bounded state machine, not an unconstrained autonomous loop. It has a maximum of three attempts and persists every attempt. Multi-hop questions may be decomposed into a small number of retrieval subqueries.

## Verification boundary

Citation metadata is first grounded to retrieved evidence. Then each answer claim is independently verified against its source excerpt. Unsupported claims are removed when possible; if the answer cannot be made reliably grounded, the system refuses.

## Document intelligence boundary

Tables become retrievable chunks. Invoice extraction is followed by deterministic validation. Monetary arithmetic uses `Decimal`. Low-confidence or inconsistent extractions enter human review.
