"""Tests for evaluation metric functions — pure, deterministic, no DB."""
from evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    citation_precision,
    citation_recall,
    citation_support_rate,
    answer_contains_keywords,
    refusal_accuracy,
    is_refusal_answer,
    agent_first_pass_success,
    agent_recovery_rate,
    agent_refusal_accuracy,
)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------
def test_recall_at_k_perfect_hit():
    relevant = {"a", "b", "c"}
    retrieved = ["a", "b", "c", "d", "e"]
    assert recall_at_k(relevant, retrieved, 5) == 1.0


def test_recall_at_k_partial_hit():
    relevant = {"a", "b", "c"}
    retrieved = ["a", "d", "e"]
    assert recall_at_k(relevant, retrieved, 3) == 1 / 3


def test_recall_at_k_cut_off_at_k():
    relevant = {"a", "b"}
    retrieved = ["a", "b", "c"]
    assert recall_at_k(relevant, retrieved, 1) == 0.5


def test_recall_at_k_empty_relevant():
    assert recall_at_k(set(), ["a", "b"], 3) == 0.0


def test_recall_at_k_empty_retrieved():
    assert recall_at_k({"a"}, [], 3) == 0.0


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------
def test_mrr_first_result_relevant():
    assert mean_reciprocal_rank({"a"}, ["a", "b", "c"]) == 1.0


def test_mrr_third_result_relevant():
    assert mean_reciprocal_rank({"c"}, ["a", "b", "c"]) == 1 / 3


def test_mrr_no_relevant_found():
    assert mean_reciprocal_rank({"x"}, ["a", "b", "c"]) == 0.0


def test_mrr_empty_relevant():
    assert mean_reciprocal_rank(set(), ["a"]) == 0.0


def test_mrr_empty_retrieved():
    assert mean_reciprocal_rank({"a"}, []) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------
def test_ndcg_perfect_ranking():
    relevant = {"a", "b"}
    retrieved = ["a", "b", "c"]
    assert ndcg_at_k(relevant, retrieved, 3) == 1.0


def test_ndcg_all_relevant_at_end():
    """Worst possible ordering: relevant items at the tail."""
    relevant = {"a"}
    retrieved = ["c", "d", "a"]
    assert ndcg_at_k(relevant, retrieved, 3) < 1.0


def test_ndcg_no_relevant_in_retrieved():
    relevant = {"a"}
    retrieved = ["b", "c"]
    assert ndcg_at_k(relevant, retrieved, 2) == 0.0


def test_ndcg_empty_relevant():
    assert ndcg_at_k(set(), ["a"], 3) == 0.0


# ---------------------------------------------------------------------------
# citation_precision / citation_recall
# ---------------------------------------------------------------------------
def test_citation_precision_all_correct():
    assert citation_precision({"a", "b"}, {"a", "b"}) == 1.0


def test_citation_precision_half_correct():
    assert citation_precision({"a", "d"}, {"a", "b", "c"}) == 0.5


def test_citation_precision_no_citations():
    assert citation_precision(set(), {"a"}) == 0.0


def test_citation_recall_full():
    assert citation_recall({"a", "b", "c"}, {"a", "b"}) == 1.0


def test_citation_recall_partial():
    assert citation_recall({"a"}, {"a", "b", "c"}) == 1 / 3


def test_citation_recall_empty_relevant():
    assert citation_recall({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# citation_support_rate
# ---------------------------------------------------------------------------
def test_citation_support_all_supported():
    cites = [{"supported": True}, {"supported": True}]
    assert citation_support_rate(cites) == 1.0


def test_citation_support_mixed():
    cites = [{"supported": True}, {"supported": False}]
    assert citation_support_rate(cites) == 0.5


def test_citation_support_empty_returns_none():
    assert citation_support_rate([]) is None


# ---------------------------------------------------------------------------
# answer_contains_keywords
# ---------------------------------------------------------------------------
def test_answer_contains_all_keywords():
    assert answer_contains_keywords(
        "Revenue was 4.82 million dollars", ["4.82", "million"]
    ) == 1.0


def test_answer_contains_some_keywords():
    assert answer_contains_keywords(
        "Revenue was 4.82 million", ["4.82", "million", "Q1"]
    ) == 2 / 3


def test_answer_contains_no_keywords():
    assert answer_contains_keywords("Hello world", ["revenue", "million"]) == 0.0


def test_answer_contains_no_expected_keywords_returns_one():
    """Vacuous truth: no expected keywords → score 1.0."""
    assert answer_contains_keywords("anything", []) == 1.0


def test_answer_contains_case_insensitive():
    assert answer_contains_keywords("Revenue was high", ["revenue"]) == 1.0


# ---------------------------------------------------------------------------
# refusal metrics
# ---------------------------------------------------------------------------
def test_is_refusal_answer_true():
    assert is_refusal_answer(
        "I couldn't find sufficient evidence in the available documents to answer this reliably."
    ) is True


def test_is_refusal_answer_false():
    assert is_refusal_answer("The answer is 42.") is False


def test_refusal_accuracy_correct_refusal():
    assert refusal_accuracy(True, True) == 1.0


def test_refusal_accuracy_correct_non_refusal():
    assert refusal_accuracy(False, False) == 1.0


def test_refusal_accuracy_false_positive():
    assert refusal_accuracy(True, False) == 0.0


def test_refusal_accuracy_false_negative():
    assert refusal_accuracy(False, True) == 0.0


# ---------------------------------------------------------------------------
# agent metrics
# ---------------------------------------------------------------------------
def test_agent_first_pass_success_yes():
    assert agent_first_pass_success(True, 1) == 1.0


def test_agent_first_pass_success_multiple_attempts():
    assert agent_first_pass_success(True, 3) == 0.0


def test_agent_first_pass_success_failed():
    assert agent_first_pass_success(False, 1) == 0.0


def test_agent_recovery_rate_yes():
    assert agent_recovery_rate(True, 2) == 1.0


def test_agent_recovery_rate_no_recovery():
    assert agent_recovery_rate(True, 1) == 0.0


def test_agent_recovery_rate_failed():
    assert agent_recovery_rate(False, 2) == 0.0


def test_agent_refusal_accuracy_correct_answer():
    assert agent_refusal_accuracy(True, False) == 1.0


def test_agent_refusal_accuracy_correct_refusal():
    assert agent_refusal_accuracy(False, True) == 1.0


def test_agent_refusal_accuracy_false_positive():
    assert agent_refusal_accuracy(False, False) == 0.0


def test_agent_refusal_accuracy_false_negative():
    assert agent_refusal_accuracy(True, True) == 0.0
