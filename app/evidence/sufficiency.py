"""
Evidence sufficiency based on the actual retrieved passages.

Deterministic checks act as a safety gate before any LLM-based evidence
evaluation. Generic lexical overlap is not enough to establish that
retrieved evidence answers the user's question.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from app.config import settings
from app.evidence.models import Evidence, EvidenceDecision
from app.services.llm_client import LLMError, get_llm_client


PROMPT = """TASK: evidence_evaluation

Judge whether the ACTUAL EVIDENCE passages contain enough information
to answer the QUESTION.

The evidence must directly address the important concepts in the question.

Do NOT consider generic words such as "company", "employee", "policy",
"work", or "information" sufficient evidence.

Return JSON:

{
  "sufficient": true|false,
  "confidence": 0-1,
  "missing_information": "...",
  "contradiction": true|false
}
"""


_GENERIC_TERMS = {
    "company",
    "companies",
    "organization",
    "organizations",
    "business",
    "businesses",
    "employee",
    "employees",
    "person",
    "people",
    "policy",
    "policies",
    "information",
    "details",
    "document",
    "documents",
    "section",
    "rule",
    "rules",
    "work",
    "working",
    "way",
    "thing",
    "what",
    "which",
    "how",
    "does",
    "doesn",
    "the",
    "and",
    "for",
    "with",
    "from",
    "about",
    # Question/function words
    "what",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "many",
    "much",
    "few",
    "often",
    "long",
    "can",
    "could",
    "would",
    "should",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "may",
    "might",
    "will",
    "shall",
}


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    normalized = set()

    for token in tokens:

        # Normalize common English morphological variants.
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"

        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]

        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]

        elif token.endswith("ly") and len(token) > 4:
            token = token[:-2]

        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]

        if (
            len(token) > 2
            and token not in _GENERIC_TERMS
        ):
            normalized.add(token)

    return normalized


def _question_content_tokens(question: str) -> set[str]:
    return _content_tokens(question)


def _evidence_tokens(
    evidence: Sequence[Evidence],
) -> set[str]:

    text = " ".join(
        item.text
        for item in evidence
    )

    return _content_tokens(text)


def _lexical_coverage(
    question: str,
    evidence: Sequence[Evidence],
) -> float:

    question_tokens = _question_content_tokens(
        question
    )

    if not question_tokens or not evidence:
        return 0.0

    evidence_tokens = _evidence_tokens(
        evidence
    )

    hits = sum(
        1
        for token in question_tokens
        if token in evidence_tokens
    )

    return hits / len(question_tokens)


def _missing_question_terms(
    question: str,
    evidence: Sequence[Evidence],
) -> set[str]:

    question_tokens = _question_content_tokens(
        question
    )

    evidence_tokens = _evidence_tokens(
        evidence
    )

    return {
        token
        for token in question_tokens
        if token not in evidence_tokens
    }


def evaluate_evidence(
    question: str,
    evidence: Sequence[Evidence],
) -> EvidenceDecision:

    if not evidence:
        return EvidenceDecision(
            False,
            0.0,
            "No retrievable evidence was found.",
        )

    coverage = _lexical_coverage(
        question,
        evidence,
    )

    missing_terms = _missing_question_terms(
        question,
        evidence,
    )

    # ---------------------------------------------------------
    # HARD SAFETY GATE
    # ---------------------------------------------------------
    #
    # If the question contains multiple meaningful concepts
    # and important concepts are completely absent from the
    # retrieved evidence, the evidence cannot be considered
    # sufficient merely because generic words overlap.
    #
    # Example:
    #
    # QUESTION:
    # "maternity leave policy for employees working in Antarctica"
    #
    # EVIDENCE:
    # "Remote Work Policy..."
    #
    # "employee", "policy", and "work" may overlap, but
    # "maternity", "leave", and "Antarctica" are absent.
    #
    # Therefore the evidence must be rejected.
    # ---------------------------------------------------------

    question_tokens = _question_content_tokens(
        question
    )

    if len(question_tokens) >= 2:

        missing_ratio = (
            len(missing_terms)
            / len(question_tokens)
        )

        if missing_ratio >= 0.5:
            return EvidenceDecision(
                False,
                round(coverage, 3),
                (
                    "Important question terms are missing "
                    "from the retrieved evidence: "
                    + ", ".join(
                        sorted(missing_terms)
                    )
                ),
            )

    # If literally none of the meaningful terms occur,
    # reject immediately.
    if coverage == 0.0:
        return EvidenceDecision(
            False,
            0.0,
            (
                "No meaningful terms from the question "
                "were found in the retrieved evidence."
            ),
        )

    evidence_text = "\n\n".join(
        f"[{i + 1}] "
        f"{item.document_name} "
        f"p.{item.page}: "
        f"{item.text}"
        for i, item in enumerate(
            evidence[:6]
        )
    )

    client = get_llm_client()

    try:

        raw = client.complete(
            PROMPT,
            (
                f"QUESTION: {question}\n\n"
                f"EVIDENCE:\n{evidence_text}"
            ),
            json_mode=True,
        )

        if not raw:
            raise LLMError(
                "Evidence evaluation returned an empty response."
            )

        data = json.loads(raw)

        llm_confidence = float(
            data.get(
                "confidence",
                0.0,
            )
        )

        confidence = round(
            0.65 * llm_confidence
            + 0.35 * coverage,
            3,
        )

        sufficient = (
            bool(
                data.get(
                    "sufficient",
                    False,
                )
            )
            and confidence
            >= settings.agent_sufficiency_threshold
        )

        return EvidenceDecision(
            sufficient,
            confidence,
            data.get(
                "missing_information"
            )
            or None,
            bool(
                data.get(
                    "contradiction",
                    False,
                )
            ),
        )

    except (
        LLMError,
        json.JSONDecodeError,
        ValueError,
    ):

        confidence = round(
            coverage,
            3,
        )

        return EvidenceDecision(
            confidence
            >= settings.agent_sufficiency_threshold,
            confidence,
            (
                None
                if confidence
                >= settings.agent_sufficiency_threshold
                else
                "Retrieved passages do not contain "
                "enough direct terms to support the question."
            ),
        )