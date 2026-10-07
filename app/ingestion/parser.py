"""
PDF parsing.

Responsible for opening a PDF, iterating pages, and extracting native
text via PyMuPDF. Decides per-page whether the extracted text is
"usable" (see `needs_ocr`) so the ingestion pipeline only invokes OCR
where it is actually necessary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF

from app.config import settings


class InvalidPDFError(Exception):
    """Raised when a file cannot be opened / parsed as a PDF."""


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed
    text: str
    char_count: int
    needs_ocr: bool
    width: float
    height: float


@dataclass
class ParsedDocument:
    num_pages: int
    pages: List[ParsedPage] = field(default_factory=list)


def parse_pdf(file_path: str) -> ParsedDocument:
    """Open a PDF and extract native text per page.

    Raises InvalidPDFError for corrupt/empty/unsupported files instead of
    letting a raw exception escape to the caller.
    """
    try:
        doc = fitz.open(file_path)
    except Exception as exc:  # pragma: no cover - depends on PyMuPDF internals
        raise InvalidPDFError(f"Could not open PDF: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise InvalidPDFError("PDF has zero pages.")

    pages: List[ParsedPage] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        text = text.strip()
        char_count = len(text)
        needs_ocr = char_count < settings.ocr_min_chars_per_page
        pages.append(
            ParsedPage(
                page_number=i + 1,
                text=text,
                char_count=char_count,
                needs_ocr=needs_ocr,
                width=page.rect.width,
                height=page.rect.height,
            )
        )
    num_pages = doc.page_count
    doc.close()
    return ParsedDocument(num_pages=num_pages, pages=pages)


def render_page_to_png_bytes(file_path: str, page_number_zero_indexed: int, zoom: float = 2.0) -> bytes:
    """Render a single page to PNG bytes for OCR consumption."""
    doc = fitz.open(file_path)
    try:
        page = doc.load_page(page_number_zero_indexed)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")
    finally:
        doc.close()
