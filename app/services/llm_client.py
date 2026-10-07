"""
LLM client.

`RealLLMClient` calls any OpenAI-compatible `/chat/completions` endpoint
(OpenAI, Groq, local Ollama, etc.) using the API key / base URL / model
from `.env`.

`MockLLMClient` is a deterministic, template-based fallback that is used
automatically whenever `LLM_API_KEY` is not set. It exists so the whole
platform - upload, retrieval, chat, extraction, the agent loop - is
genuinely runnable end-to-end with zero external accounts or cost. It is
clearly surfaced to the user everywhere ("Mock LLM mode") rather than
silently pretending to be a real model; swapping in a real key is a
one-line `.env` change (see README).

Every call site in the app goes through `get_llm_client()` so the
real/mock choice is made in exactly one place.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from app.config import llm_is_configured, settings

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "did", "do", "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "so",
    "to", "of", "in", "on", "at", "for", "with", "about", "as", "by",
    "this", "that", "these", "those", "it", "its", "compare", "versus",
    "more", "most", "specific", "directly", "topic", "found", "not",
    "information", "available", "corpus", "passages", "please", "tell",
}


def _keywords_only(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9']+", text)
    return [w for w in words if w.lower() not in _STOPWORDS and len(w) > 2]


class LLMError(Exception):
    """Raised when a real LLM call fails (network, auth, timeout, etc.)."""


class LLMClient:
    mode: str = "unknown"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        raise NotImplementedError


class RealLLMClient(LLMClient):
    mode = "live"

    def __init__(self):
        self._client = httpx.Client(
            timeout=settings.llm_timeout_seconds
        )

    def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
    ) -> str:

        url = (
            f"{settings.llm_base_url.rstrip('/')}"
            "/chat/completions"
        )

        headers = {
            "Authorization": (
                f"Bearer {settings.llm_api_key}"
            ),
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        # Use `response_format: json_object` (when json_mode is requested)
        # so the model is structurally constrained to emit parseable JSON,
        # and set a generous max_tokens.
        
        if json_mode:
            payload["response_format"] = {
                "type": "json_object"
            }

        try:
            resp = self._client.post(
                url,
                headers=headers,
                json=payload,
            )

            resp.raise_for_status()

            data = resp.json()

            # Validate the response structure instead of assuming
            # choices/message/content always exist.
            choices = data.get("choices")

            if not isinstance(choices, list) or not choices:
                raise LLMError(
                    "LLM response contained no choices."
                )

            message = choices[0].get("message")

            if not isinstance(message, dict):
                raise LLMError(
                    "LLM response contained no valid message."
                )

            content = message.get("content")

            # Normal OpenAI-compatible response.
            if isinstance(content, str) and content.strip():
                return content.strip()

            # Some providers return content as structured parts.
            if isinstance(content, list):
                text_parts = []

                for part in content:
                    if not isinstance(part, dict):
                        continue

                    text = part.get("text")

                    if isinstance(text, str) and text.strip():
                        text_parts.append(text)

                combined = "".join(text_parts).strip()

                if combined:
                    return combined

            # Do NOT return None. A successful HTTP response with
            # no usable answer is an LLM-level failure.
            finish_reason = choices[0].get(
                "finish_reason"
            )

            raise LLMError(
                "LLM returned an empty message content "
                f"(finish_reason={finish_reason!r})."
            )

        except httpx.TimeoutException as exc:
            raise LLMError(
                f"LLM request timed out: {exc}"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise LLMError(
                "LLM API returned an error: "
                f"{exc.response.status_code} "
                f"{exc.response.text[:300]}"
            ) from exc

        except LLMError:
            raise

        except (ValueError, KeyError) as exc:
            raise LLMError(
                f"Invalid LLM response format: {exc}"
            ) from exc

        except Exception as exc:
            raise LLMError(
                f"LLM call failed: {exc}"
            ) from exc


class MockLLMClient(LLMClient):
    """
    Deterministic, rule-based stand-in for an LLM. Each "task" is
    recognised by a marker embedded in the prompt by the caller
    (see the `TASK:` prefix convention below) so this single mock can
    serve generation, query analysis, evidence evaluation, query
    reformulation, and invoice extraction without a real model.
    """

    mode = "mock"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        task = self._detect_task(system)
        if task == "generation":
            return self._mock_generation(user)
        if task == "query_analysis":
            return self._mock_query_analysis(user)
        if task == "evidence_evaluation":
            return self._mock_evidence_evaluation(user)
        if task == "reformulation":
            return self._mock_reformulation(user)
        if task == "extraction":
            return self._mock_extraction(user)
        if task == "citation_verification":
            return self._mock_citation_verification(user)
        # Generic fallback: extractive summary of the provided context.
        return json.dumps({"answer": "Mock LLM: no specific handler for this task.", "citations": []})

    @staticmethod
    def _detect_task(system: str) -> str:
        m = re.search(r"TASK:\s*(\w+)", system)
        return m.group(1) if m else ""

    # -- Module 1: answer generation -----------------------------------
    def _mock_generation(self, user: str) -> str:
        evidence_blocks = re.findall(
            r"\[EVIDENCE (\d+)\|doc=(.*?)\|page=(\d+)"
            r"(?:\|type=.*?)?\|chunk=(.*?)\]\n"
            r"(.*?)(?=\n\[EVIDENCE|\n\nQUESTION:|\Z)",
            user,
            re.S,
        )
        question_match = re.search(r"QUESTION:\s*(.*)", user)
        question = question_match.group(1).strip() if question_match else ""

        if not evidence_blocks:
            return json.dumps({"answer": "I couldn't find sufficient evidence in the available documents to answer this reliably.", "citations": []})

        # Extractive strategy: pick the sentence in the top evidence chunk with
        # highest lexical overlap with the question, cite it plainly.
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        best_sentence = None
        best_overlap = -1
        citations = []
        for idx, doc, page, chunk_id, text in evidence_blocks[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            for sent in sentences:
                s_tokens = set(re.findall(r"[a-z0-9]+", sent.lower()))
                overlap = len(q_tokens & s_tokens)
                if overlap > best_overlap and len(sent.strip()) > 15:
                    best_overlap = overlap
                    best_sentence = sent.strip()
                    best_doc, best_page, best_chunk = doc.strip(), int(page), chunk_id.strip()

        if best_sentence is None:
            # fall back to first evidence block
            doc, page, chunk_id, text = evidence_blocks[0][1], evidence_blocks[0][2], evidence_blocks[0][3], evidence_blocks[0][4]
            best_sentence = text.strip()[:300]
            best_doc, best_page, best_chunk = doc.strip(), int(page), chunk_id.strip()

        answer = best_sentence
        citations.append({"document": best_doc, "page": best_page, "chunk_id": best_chunk, "claim_text": answer})
        return json.dumps({"answer": answer, "citations": citations})

    # -- Module 3: query analysis --------------------------------------
    def _mock_query_analysis(self, user: str) -> str:
        q_match = re.search(r"QUESTION:\s*(.*)", user)
        question = q_match.group(1).strip() if q_match else user
        lower = question.lower()
        words = re.findall(r"[a-zA-Z0-9']+", question)

        query_type = "simple"
        if any(w in lower for w in [" and ", "compare", "difference between", "versus", " vs "]):
            query_type = "multi-hop"
        elif "?" not in question and len(words) < 4:
            query_type = "ambiguous"
        elif any(w in lower for w in ["this document", "the attached", "this file", "this pdf"]):
            query_type = "document-specific"

        dates = re.findall(r"\b(19|20)\d{2}\b", question)
        keywords = [w.lower() for w in _keywords_only(question)][:8]
        entities = [w for w in words if w[:1].isupper() and w.lower() not in _STOPWORDS][:8]

        return json.dumps(
            {
                "query_type": query_type,
                "entities": entities,
                "keywords": keywords,
                "dates": dates,
                "important_concepts": keywords[:5],
            }
        )

    # -- Module 3: evidence evaluation ----------------------------------
    def _mock_evidence_evaluation(self, user: str) -> str:
        q_match = re.search(r"QUESTION:\s*(.*?)(?:\n\nEVIDENCE:|\Z)", user, re.S)
        question = q_match.group(1).strip() if q_match else user
        evidence_match = re.search(r"EVIDENCE:\s*(.*)", user, re.S)
        evidence = evidence_match.group(1) if evidence_match else ""
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        e_tokens = set(re.findall(r"[a-z0-9]+", evidence.lower()))
        coverage = (len(q_tokens & e_tokens) / len(q_tokens)) if q_tokens else 0.0
        sufficient = coverage >= settings.agent_sufficiency_threshold
        missing = "" if sufficient else "Direct evidence for one or more important question terms was not found."
        return json.dumps({"sufficient": sufficient, "confidence": round(coverage, 3), "missing_information": missing, "contradiction": False})

    # -- Module 3: query reformulation ----------------------------------
    def _mock_reformulation(self, user: str) -> str:
        q_match = re.search(r"ORIGINAL_QUESTION:\s*(.*)", user)
        missing_match = re.search(r"MISSING_INFORMATION:\s*(.*)", user)
        question = q_match.group(1).strip() if q_match else ""
        missing = missing_match.group(1).strip() if missing_match else ""
        keywords = _keywords_only(question)
        extra = _keywords_only(missing)[:3]
        new_query = " ".join(dict.fromkeys(keywords + extra))
        return json.dumps({"reformulated_query": new_query or question})

    # -- Module 2: invoice extraction ------------------------------------
    def _mock_extraction(self, user: str) -> str:
        text = user
        fields: Dict[str, Any] = {}

        def find(pattern, default=None, flags=re.I):
            m = re.search(pattern, text, flags)
            return m.group(1).strip() if m else default

        fields["vendor_name"] = find(r"(?:vendor|from|bill\s*from)\s*[:\-]\s*(.+)")
        fields["invoice_number"] = find(r"invoice\s*(?:number|#|no\.?)\s*[:\-]?\s*([A-Za-z0-9\-]+)")
        fields["invoice_date"] = find(r"(?:invoice\s*date|date)\s*[:\-]\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})")
        fields["subtotal"] = find(r"sub\s*-?total\s*[:\-]?\s*\$?([0-9,]+\.?[0-9]*)")
        fields["tax"] = find(r"tax\s*[:\-]?\s*\$?([0-9,]+\.?[0-9]*)")
        fields["total"] = find(r"(?<!sub)total\s*[:\-]?\s*\$?([0-9,]+\.?[0-9]*)")
        fields["currency"] = find(r"\b(USD|EUR|GBP|PKR)\b") or ("USD" if "$" in text else None)

        for k in ("subtotal", "tax", "total"):
            if fields.get(k):
                try:
                    fields[k] = float(str(fields[k]).replace(",", ""))
                except ValueError:
                    fields[k] = None
        return json.dumps(fields)

    # -- Module 1: citation verification ---------------------------------
    def _mock_citation_verification(self, user: str) -> str:
        claim_match = re.search(r"CLAIM:\s*(.*)", user)
        source_match = re.search(r"SOURCE:\s*(.*)", user, re.S)
        claim = claim_match.group(1).strip() if claim_match else ""
        source = source_match.group(1).strip() if source_match else ""

        claim_tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
        source_tokens = set(re.findall(r"[a-z0-9]+", source.lower()))
        if not claim_tokens:
            return json.dumps({"supported": False, "confidence": 0.0, "reason": "Empty claim."})
        overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
        supported = overlap >= 0.6
        reason = (
            f"{int(overlap * 100)}% of the claim's key terms appear in the cited source chunk."
            if supported
            else f"Only {int(overlap * 100)}% of the claim's key terms appear in the cited source chunk."
        )
        return json.dumps({"supported": supported, "confidence": round(overlap, 3), "reason": reason})


_client_singleton: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    if llm_is_configured():
        _client_singleton = RealLLMClient()
    else:
        logger.info("LLM_API_KEY not set - using MockLLMClient (deterministic, offline).")
        _client_singleton = MockLLMClient()
    return _client_singleton


def reset_llm_client_cache() -> None:
    """Used by tests to force re-evaluation of settings.llm_api_key."""
    global _client_singleton
    _client_singleton = None
