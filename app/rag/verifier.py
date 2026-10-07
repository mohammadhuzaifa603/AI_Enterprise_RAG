"""Independent claim-level verification."""
from __future__ import annotations
import json, logging, re
from dataclasses import dataclass
from typing import Dict, List
from app.services.llm_client import LLMError, get_llm_client

logger = logging.getLogger(__name__)
VERIFICATION_SYSTEM_PROMPT = """TASK: citation_verification
You are a strict independent verifier. Given a CLAIM and exact SOURCE text,
decide whether the source entails/supports the claim. Do not use outside knowledge.
Return JSON: {\"supported\":true|false,\"confidence\":0-1,\"reason\":\"...\"}
"""
@dataclass
class VerificationResult:
    supported: bool
    confidence: float
    reason: str

def verify_claim(claim_text: str, source_text: str) -> VerificationResult:
    if not claim_text.strip() or not source_text.strip():
        return VerificationResult(
            False,
            0.0,
            "Missing claim or source text."
        )

    client = get_llm_client()

    try:
        raw = client.complete(
            VERIFICATION_SYSTEM_PROMPT,
            f"CLAIM: {claim_text}\n\nSOURCE: {source_text}",
            json_mode=True,
        )

        # Some LLM/provider responses can be empty.
        # Do not try to parse None as JSON.
        if not raw:
            raise LLMError(
                "Claim verification returned an empty response."
            )

        data = json.loads(raw)

        return VerificationResult(
            bool(data.get("supported", False)),
            max(
                0.0,
                min(
                    1.0,
                    float(data.get("confidence", 0.0))
                )
            ),
            str(data.get("reason", "")),
        )

    except (LLMError, json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "Claim verification failed: %s",
            exc,
        )

        return VerificationResult(
            False,
            0.0,
            f"Verification error: {exc}",
        )

def verify_citations(citations: List[Dict]) -> List[Dict]:
    return [
        {**c, "supported": (r := verify_claim(c.get("claim_text", ""), c.get("excerpt", ""))).supported,
         "confidence": r.confidence, "reason": r.reason}
        for c in citations
    ]

def _sentences(answer: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]


def _claim_matches_sentence(sentence: str, claim: str, threshold: float = 0.5) -> bool:
    """Check whether a citation's claim_text supports a given answer sentence.

    Uses token overlap ratio: the fraction of the claim's key tokens that
    appear in the sentence. This avoids false positives where any single
    supported citation validates every sentence in the answer.
    """
    if not claim.strip():
        return False
    sentence_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    claim_tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
    if not claim_tokens:
        return False
    overlap = len(sentence_tokens & claim_tokens) / len(claim_tokens)
    return overlap >= threshold

def verify_answer_claims(
    answer: str,
    citations: List[Dict]
) -> tuple[str, List[Dict], bool]:
    """Require every substantive answer sentence to have supported evidence.

    A sentence is considered supported when it has at least one verified
    citation. The citation itself is independently checked against the source.
    """

    verified = verify_citations(citations)
    sentences = _sentences(answer)

    if not sentences:
        return "", verified, False
    

    supported = []

    for sentence in sentences:
        matching = [
            citation
            for citation in verified
            if citation.get("supported")
            and _claim_matches_sentence(
                sentence,
                citation.get("claim_text", ""),
            )
        ]

        if matching:
            supported.append(sentence)
    if len(supported) == len(sentences):
        return answer, verified, True

    if supported:
        return " ".join(supported), verified, True

    return (
        "I couldn't verify the generated claims against the "
        "retrieved source evidence, so I won't present them as "
        "reliable facts.",
        verified,
        False,
    )