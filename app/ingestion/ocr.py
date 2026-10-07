"""
OCR fallback.

Only invoked for pages that `parser.parse_pdf` flagged as `needs_ocr`.
Uses Tesseract via pytesseract after rendering the PDF page to an image.
"""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass

from PIL import Image

from app.config import settings
from app.ingestion.parser import render_page_to_png_bytes

logger = logging.getLogger(__name__)


class OCRUnavailableError(Exception):
    """Raised when the Tesseract binary / language data is not available."""


@dataclass
class OCRResult:
    text: str
    confidence: float | None
    failed: bool = False
    error: str | None = None


def ocr_page(
    file_path: str,
    page_number_zero_indexed: int,
) -> OCRResult:
    """Render a page and run Tesseract OCR.

    Returns OCR text together with an average confidence score.
    OCR failures are represented explicitly instead of being confused
    with a successful OCR result containing no text.
    """
    try:
        import pytesseract

        # Prefer an explicit executable path so Windows/Linux installations
        # do not depend on PATH configuration.
        tesseract_cmd = settings.tesseract_cmd or os.getenv("TESSERACT_CMD")

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    except ImportError:
        logger.warning("pytesseract not installed; OCR unavailable.")
        return OCRResult(
            text="",
            confidence=None,
            failed=True,
            error="pytesseract is not installed.",
        )

    try:
        png_bytes = render_page_to_png_bytes(
            file_path,
            page_number_zero_indexed,
        )

        image = Image.open(io.BytesIO(png_bytes))

        text = pytesseract.image_to_string(image).strip()

        # Get word-level confidence scores from Tesseract.
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
        )

        confidences = []

        for value in data.get("conf", []):
            try:
                confidence = float(value)

                # Tesseract uses negative values for invalid/unavailable
                # confidence entries.
                if confidence >= 0:
                    confidences.append(confidence)
            except (TypeError, ValueError):
                continue

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else None
        )

        return OCRResult(
            text=text,
            confidence=average_confidence,
            failed=False,
        )

    except Exception as exc:
        logger.warning(
            "OCR failed for page %s of %s: %s",
            page_number_zero_indexed,
            file_path,
            exc,
        )

        return OCRResult(
            text="",
            confidence=None,
            failed=True,
            error=str(exc),
        )