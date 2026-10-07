"""Invoice extraction with Decimal-safe financial fields."""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ValidationError
from app.services.llm_client import LLMError, get_llm_client

logger = logging.getLogger(__name__)
EXTRACTION_SYSTEM_PROMPT = """TASK: extraction
Extract invoice fields from the provided document text. Only use values that literally appear in the text.
Never invent a number, date or name. If a field is not present, use null.
Respond ONLY with JSON keys: vendor_name, invoice_number, invoice_date, subtotal, discount, tax, shipping,
fees, total, currency, line_items. Monetary values must be plain decimal numbers.
"""

class InvoiceFields(BaseModel):
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    subtotal: Optional[Decimal] = None
    discount: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    shipping: Optional[Decimal] = None
    fees: Optional[Decimal] = None
    total: Optional[Decimal] = None
    currency: Optional[str] = None
    line_items: list[dict[str, Any]] = []

def _decimalize(value):
    if value is None or isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None

def _extract_json_object(raw: str) -> Any:
    """Extract a valid JSON object from a raw LLM response.

    Handles markdown-wrapped JSON, reasoning text before the JSON,
    and a JSON object nested inside the ``answer`` field.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise json.JSONDecodeError("No JSON object found in response", raw, 0)


def extract_invoice_fields(document_text: str) -> Dict[str, Any]:
    client = get_llm_client()
    try:
        raw = client.complete(EXTRACTION_SYSTEM_PROMPT, document_text, json_mode=True)
        if not raw or not raw.strip():
            raise LLMError("Invoice extraction returned an empty response.")
        data = _extract_json_object(raw)
        for key in ("subtotal", "discount", "tax", "shipping", "fees", "total"):
            data[key] = _decimalize(data.get(key))
        fields = InvoiceFields(**data)
        return fields.model_dump()
    except (LLMError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        logger.warning("Invoice extraction failed: %s", exc)
        return InvoiceFields().model_dump()


# ---------------------------------------------------------------------------
# Field-level evidence grounding
# ---------------------------------------------------------------------------

@dataclass
class FieldEvidenceData:
    """Grounded evidence for a single extracted invoice field."""

    field_name: str
    page_number: Optional[int]
    source_text: str
    confidence: float


SEARCHABLE_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "currency",
    "subtotal", "discount", "tax", "shipping", "fees", "total",
]


def _decimal_search_variants(value: Decimal) -> List[str]:
    """Build string variants for searching a Decimal value in page text."""
    variants = [str(value)]
    try:
        comma_fmt = format(value, ",.2f")
        if comma_fmt not in variants:
            variants.append(comma_fmt)
    except (ValueError, TypeError):
        pass
    stripped = str(value).rstrip("0").rstrip(".")
    if stripped and stripped not in variants:
        variants.append(stripped)
    return variants


def _extract_line(text: str, idx: int) -> str:
    """Return the line of *text* that contains character position *idx*."""
    line_start = text.rfind("\n", 0, idx) + 1
    line_end = text.find("\n", idx)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()


def _search_value_in_pages(
    value: Any,
    pages: List[Tuple[int, str]],
) -> List[Tuple[int, str]]:
    """Search for *value* across page texts.

    Returns a list of ``(page_number, source_line)`` matches.
    """
    if value is None or value == "":
        return []

    if isinstance(value, Decimal):
        search_strings = _decimal_search_variants(value)
    else:
        search_strings = [str(value).strip()]

    matches: List[Tuple[int, str]] = []
    for page_num, page_text in pages:
        page_lower = page_text.lower()
        for needle in search_strings:
            idx = page_lower.find(needle.lower())
            if idx >= 0:
                matches.append((page_num, _extract_line(page_text, idx)))
                break
    return matches


def ground_field_evidence(
    fields: Dict[str, Any],
    pages: Optional[List[Tuple[int, str]]] = None,
) -> Dict[str, Optional[FieldEvidenceData]]:
    """Ground extracted invoice fields to their source page text.

    Only fields that were actually extracted (non-None, non-empty) are
    considered.  A field that cannot be located in any page text receives
    ``None`` evidence rather than a fabricated page reference.

    Returns a mapping of ``field_name -> FieldEvidenceData | None``.
    """
    if not pages:
        return {f: None for f in SEARCHABLE_FIELDS}

    result: Dict[str, Optional[FieldEvidenceData]] = {}
    for field_name in SEARCHABLE_FIELDS:
        value = fields.get(field_name)
        if value is None or value == "":
            result[field_name] = None
            continue

        matches = _search_value_in_pages(value, pages)
        if not matches:
            result[field_name] = None
        elif len(matches) == 1:
            page_num, source = matches[0]
            result[field_name] = FieldEvidenceData(
                field_name=field_name,
                page_number=page_num,
                source_text=source,
                confidence=1.0,
            )
        else:
            page_num, source = matches[0]
            result[field_name] = FieldEvidenceData(
                field_name=field_name,
                page_number=page_num,
                source_text=source,
                confidence=0.5,
            )
    return result
