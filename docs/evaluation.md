# Evaluation Plan

The evaluation dataset should contain questions with explicit expected evidence chunk IDs and expected answer facts.

## Retrieval

Measure Recall@1/3/5, MRR and NDCG against expected evidence chunks.

## Grounding

Measure the percentage of generated claims that are supported by at least one expected evidence item.

## Citation verification

Measure citation precision, citation recall, and verification agreement against manually labeled claim/evidence pairs.

## Agent

Track first-pass success, recovery after reformulation, correct refusal, number of attempts, and unsupported-answer rate.

## Invoice extraction

Label expected fields and measure precision/recall/F1 per field plus arithmetic validation accuracy.

## Safety cases

Include:

- missing-answer questions
- conflicting documents
- scanned pages
- table-only evidence
- unauthorized documents
- inconsistent invoices
- unsupported LLM citations
