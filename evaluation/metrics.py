"""
Evaluation metric functions.

Every function here is pure: it takes concrete values (sets, lists, strings)
and returns a float.  No database access, no LLM calls, no I/O.
This makes the metrics trivially testable and deterministic.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def recall_at_k(
    relevant_ids: Set[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of relevant items found in the top-K retrieved.

    Returns 0.0 when the relevant set is empty (no ground truth to match).
    """
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return len(relevant_ids & retrieved_k) / len(relevant_ids)


def mean_reciprocal_rank(
    relevant_ids: Set[str],
    retrieved_ids: Sequence[str],
) -> float:
    """Reciprocal rank of the first relevant item.

    Returns 0.0 if no relevant item is retrieved.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    relevant_ids: Set[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if relevant, 0 otherwise).
    Returns 1.0 for perfect ranking, 0.0 when no relevant items exist
    or none are retrieved.
    """
    if not relevant_ids:
        return 0.0

    def _dcg(scores: Sequence[int], cutoff: int) -> float:
        total = 0.0
        for i, s in enumerate(scores[:cutoff], start=1):
            total += (2 ** s - 1) / math.log2(i + 1)
        return total

    relevance = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    dcg = _dcg(relevance, k)

    ideal = sorted(relevance, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------
def citation_precision(
    cited_ids: Set[str],
    relevant_ids: Set[str],
) -> float:
    """Precision = |relevant ∩ cited| / |cited|.

    Returns 0.0 when no citations were made.
    """
    if not cited_ids:
        return 0.0
    return len(relevant_ids & cited_ids) / len(cited_ids)


def citation_recall(
    cited_ids: Set[str],
    relevant_ids: Set[str],
) -> float:
    """Recall = |relevant ∩ cited| / |relevant|.

    Returns 0.0 when the relevant set is empty.
    """
    if not relevant_ids:
        return 0.0
    return len(relevant_ids & cited_ids) / len(relevant_ids)


def citation_support_rate(
    citations: Sequence[Dict],
) -> Optional[float]:
    """Fraction of citations marked as 'supported'.

    Returns None when there are zero citations (undefined metric).
    """
    if not citations:
        return None
    supported = sum(1 for c in citations if c.get("supported", False))
    return supported / len(citations)


# ---------------------------------------------------------------------------
# Answer quality metrics
# ---------------------------------------------------------------------------
def answer_contains_keywords(
    answer: str,
    expected_keywords: Sequence[str],
) -> float:
    """Fraction of expected keywords found in the generated answer.

    Keyword matching is case-insensitive whole-word substring matching.
    Returns 1.0 when no expected keywords are specified (vacuous truth).
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


# ---------------------------------------------------------------------------
# Refusal metrics
# ---------------------------------------------------------------------------
NO_EVIDENCE_ANSWER = (
    "I couldn't find sufficient evidence in the available documents "
    "to answer this reliably."
)


def is_refusal_answer(answer: str) -> bool:
    """True when *answer* is the backend's safe-refusal string."""
    return answer.strip() == NO_EVIDENCE_ANSWER


def refusal_accuracy(
    is_refused: bool,
    should_refuse: bool,
) -> float:
    """1.0 if the system's refusal decision matches the expectation."""
    return 1.0 if is_refused == should_refuse else 0.0


# ---------------------------------------------------------------------------
# Agent metrics
# ---------------------------------------------------------------------------
def agent_first_pass_success(
    succeeded: bool,
    num_attempts: int,
) -> float:
    """1.0 if the agent succeeded on its first (and only) attempt."""
    return 1.0 if (succeeded and num_attempts == 1) else 0.0


def agent_recovery_rate(
    succeeded: bool,
    num_attempts: int,
) -> float:
    """1.0 if the agent succeeded after reformulating (>1 attempt)."""
    return 1.0 if (succeeded and num_attempts > 1) else 0.0


def agent_refusal_accuracy(
    succeeded: bool,
    should_refuse: bool,
) -> float:
    """1.0 if the agent correctly refused / correctly answered."""
    return 1.0 if (succeeded == (not should_refuse)) else 0.0
