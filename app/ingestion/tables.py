"""
Table extraction via pdfplumber.

Tables are represented as structured {"columns": [...], "rows": [[...]]}
dicts. Extraction failures for a given page never abort the pipeline -
they are reported back as `extraction_failed` so the caller can flag
the page for review instead of inventing table content.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pdfplumber

logger = logging.getLogger(__name__)


def extract_tables_from_pdf(file_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """Return {page_number (1-indexed): [table_dict, ...]} for the whole PDF."""
    results: Dict[int, List[Dict[str, Any]]] = {}
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                tables = _extract_tables_from_page(page, page_number)
                if tables:
                    results[page_number] = tables
    except Exception as exc:  # pragma: no cover - depends on pdfplumber internals
        logger.warning("Table extraction failed for %s: %s", file_path, exc)
    return results


def _extract_tables_from_page(page, page_number: int) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    try:
        raw_tables = page.extract_tables()
    except Exception as exc:  # pragma: no cover
        logger.warning("Table extraction failed on page %s: %s", page_number, exc)
        return [{"columns": [], "rows": [], "extraction_failed": True, "error": str(exc)}]

    for raw in raw_tables or []:
        if not raw or len(raw) < 1:
            continue
        header = [str(c).strip() if c is not None else "" for c in raw[0]]
        rows = []
        for row in raw[1:]:
            cleaned = [c.strip() if isinstance(c, str) else c for c in row]
            # Skip fully-empty rows rather than inventing placeholder values.
            if any(v not in (None, "") for v in cleaned):
                rows.append(cleaned)
        tables.append({"columns": header, "rows": rows, "extraction_failed": False})
    return tables
