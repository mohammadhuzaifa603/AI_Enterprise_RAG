"""Document ingestion: validate -> parse/OCR -> tables -> chunks -> embeddings.

Tables are first-class chunks, so structured PDF content participates in
retrieval instead of being stored only as page metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.document_ai.confidence import compute_confidence
from app.document_ai.extraction import extract_invoice_fields, ground_field_evidence
from app.document_ai.validation import validate_invoice_fields
from app.ingestion.chunker import chunk_page_text
from app.ingestion.embeddings import embed_texts
from app.ingestion.ocr import ocr_page
from app.ingestion.parser import InvalidPDFError, parse_pdf
from app.ingestion.tables import extract_tables_from_pdf
from app.models import Chunk, Document, Extraction, FieldEvidence, Page
from app.storage import materialize

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    document_id: str
    status: str
    num_pages: int
    num_chunks: int
    ocr_pages: int
    tables_found: int
    ocr_failed_pages: int = 0
    low_confidence_ocr_pages: int = 0
    extraction_id: str | None = None
    error_message: str | None = None


def _table_to_text(table: dict) -> str:
    cols = table.get("columns") or []
    rows = table.get("rows") or []

    lines = [
        "TABLE: " + " | ".join(str(x or "") for x in cols)
    ]

    for row in rows:
        lines.append(
            " | ".join(str(x or "") for x in row)
        )

    return "\n".join(lines)

def _determine_document_status(
    usable_pages: int,
    total_pages: int,
    ocr_failed_pages: int,
    low_confidence_ocr_pages: int,
) -> str:
    if low_confidence_ocr_pages > 0:
        return "needs_review"

    if (
        ocr_failed_pages > 0
        or usable_pages < total_pages
    ):
        return "partial"

    return "processed"

def process_document(
    db: Session,
    document: Document,
) -> IngestionResult:

    document.status = "processing"
    document.error_message = None
    db.commit()

    try:
        with materialize(
            document.file_path,
            document.storage_encrypted,
        ) as pdf_path:

            parsed = parse_pdf(pdf_path)

            if parsed.num_pages > 100:
                raise InvalidPDFError(
                    "PDF exceeds the configured page limit of 100 pages."
                )

            tables_by_page = extract_tables_from_pdf(
                pdf_path
            )

            all_chunks = []
            full_text_parts = []
            page_texts = []

            ocr_pages_count = 0
            usable_pages = 0
            ocr_failed_pages = 0
            low_confidence_ocr_pages = 0

            # ---------------------------------------------------------
            # Process every page
            # ---------------------------------------------------------

            for parsed_page in parsed.pages:

                text = parsed_page.text or ""
                used_ocr = False
                ocr_confidence = None

                # -----------------------------------------------------
                # OCR fallback
                # -----------------------------------------------------

                if parsed_page.needs_ocr:

                    ocr_result = ocr_page(
                        pdf_path,
                        parsed_page.page_number - 1,
                    )

                    ocr_confidence = ocr_result.confidence

                    if ocr_result.failed:
                        ocr_failed_pages += 1

                    if (
                        ocr_result.confidence is not None
                        and ocr_result.confidence < 50
                    ):
                        low_confidence_ocr_pages += 1

                    if ocr_result.text:
                        text = ocr_result.text
                        used_ocr = True
                        ocr_pages_count += 1

                # -----------------------------------------------------
                # Determine whether page has usable content
                # -----------------------------------------------------

                if text.strip():
                    usable_pages += 1

                # -----------------------------------------------------
                # Tables for this page
                # -----------------------------------------------------

                page_tables = tables_by_page.get(
                    parsed_page.page_number,
                    [],
                )

                # -----------------------------------------------------
                # Persist page
                # -----------------------------------------------------

                page_row = Page(
                    document_id=document.id,
                    page_number=parsed_page.page_number,
                    text=text,
                    used_ocr=used_ocr,
                    char_count=len(text),
                    tables_json=page_tables or None,
                    ocr_confidence=ocr_confidence,
                )

                db.add(page_row)
                db.flush()

                full_text_parts.append(text)
                page_texts.append((parsed_page.page_number, text))

                # -----------------------------------------------------
                # Text chunks
                # -----------------------------------------------------

                for c in chunk_page_text(
                    parsed_page.page_number,
                    text,
                ):

                    all_chunks.append(
                        (
                            page_row.id,
                            c.page_number,
                            "text",
                            c.text,
                            {
                                "source": (
                                    "ocr"
                                    if used_ocr
                                    else "native"
                                )
                            },
                        )
                    )

                # -----------------------------------------------------
                # Table chunks
                # -----------------------------------------------------

                for table_index, table in enumerate(
                    page_tables
                ):

                    table_text = _table_to_text(
                        table
                    )

                    if (
                        table_text.strip()
                        and not table.get(
                            "extraction_failed"
                        )
                    ):

                        all_chunks.append(
                            (
                                page_row.id,
                                parsed_page.page_number,
                                "table",
                                table_text,
                                {
                                    "source": "pdf_table",
                                    "table_index": table_index,
                                },
                            )
                        )

            # ---------------------------------------------------------
            # Fail if the entire document has no usable content
            # ---------------------------------------------------------

            if not usable_pages:

                document.status = "failed"

                document.error_message = (
                    "No usable text could be extracted from the "
                    "document; OCR may be unavailable."
                )

                db.commit()

                return IngestionResult(
                document_id=document.id,
                status="failed",
                num_pages=parsed.num_pages,
                num_chunks=0,
                ocr_pages=ocr_pages_count,
                tables_found=sum(
                    len(v)
                    for v in tables_by_page.values()
                ),
                ocr_failed_pages=ocr_failed_pages,
                low_confidence_ocr_pages=low_confidence_ocr_pages,
                error_message=document.error_message,
            )
            # ---------------------------------------------------------
            # Generate embeddings
            # ---------------------------------------------------------

            vectors = (
                embed_texts(
                    [x[3] for x in all_chunks]
                )
                if all_chunks
                else []
            )

            # ---------------------------------------------------------
            # Persist chunks
            # ---------------------------------------------------------

            for idx, (
                (page_id, page_number, chunk_type, text, metadata),
                vector,
            ) in enumerate(
                zip(all_chunks, vectors)
            ):

                db.add(
                    Chunk(
                        document_id=document.id,
                        page_id=page_id,
                        page_number=page_number,
                        chunk_index=idx,
                        chunk_type=chunk_type,
                        text=text,
                        embedding=vector,
                        chunk_metadata=metadata,
                    )
                )

            # ---------------------------------------------------------
            # Final document status
            # ---------------------------------------------------------

            tables_found = sum(
                len(v)
                for v in tables_by_page.values()
            )

            document.num_pages = parsed.num_pages

            document.status = _determine_document_status(
                usable_pages=usable_pages,
                total_pages=parsed.num_pages,
                ocr_failed_pages=ocr_failed_pages,
                low_confidence_ocr_pages=low_confidence_ocr_pages,
            )

            document.processed_at = (
                datetime.now(timezone.utc)
            )

            db.commit()

            # ---------------------------------------------------------
            # Invoice extraction
            # ---------------------------------------------------------

            extraction_id = None

            if document.doc_type == "invoice":

                extraction_id = run_invoice_extraction(
                    db,
                    document,
                    "\n".join(full_text_parts),
                    ocr_pages_count
                    / max(1, parsed.num_pages),
                    pages=page_texts,
                )

            return IngestionResult(
                document_id=document.id,
                status=document.status,
                num_pages=parsed.num_pages,
                num_chunks=len(all_chunks),
                ocr_pages=ocr_pages_count,
                tables_found=tables_found,
                ocr_failed_pages=ocr_failed_pages,
                low_confidence_ocr_pages=low_confidence_ocr_pages,
                extraction_id=extraction_id,
            )

    # -------------------------------------------------------------
    # Invalid PDF
    # -------------------------------------------------------------

    except InvalidPDFError as exc:

        document.status = "failed"
        document.error_message = str(exc)

        db.commit()

        return IngestionResult(
            document_id=document.id,
            status="failed",
            num_pages=0,
            num_chunks=0,
            ocr_pages=0,
            tables_found=0,
            error_message=str(exc),
        )

    # -------------------------------------------------------------
    # Unexpected failure
    # -------------------------------------------------------------

    except Exception as exc:

        logger.exception(
            "Document processing failed for %s",
            document.id,
        )

        document.status = "failed"
        document.error_message = str(exc)

        db.commit()

        return IngestionResult(
            document_id=document.id,
            status="failed",
            num_pages=document.num_pages,
            num_chunks=0,
            ocr_pages=0,
            tables_found=0,
            error_message=str(exc),
        )


def run_invoice_extraction(
    db: Session,
    document: Document,
    full_text: str,
    ocr_ratio: float,
    pages: list[tuple[int, str]] | None = None,
) -> str:

    fields = extract_invoice_fields(
        full_text
    )

    valid, errors, warnings = (
        validate_invoice_fields(fields)
    )

    score, breakdown, status = (
        compute_confidence(
            fields,
            valid,
            errors,
            warnings,
            ocr_ratio,
        )
    )

    extraction = Extraction(
        document_id=document.id,
        doc_type="invoice",
        fields=fields,
        valid=valid,
        validation_errors=errors,
        validation_warnings=warnings,
        confidence_score=score,
        confidence_breakdown=breakdown,
        status=status,
    )

    db.add(extraction)
    db.flush()

    if pages:
        field_evidence = ground_field_evidence(fields, pages)
        for field_name, evidence in field_evidence.items():
            if evidence is not None:
                db.add(FieldEvidence(
                    extraction_id=extraction.id,
                    field_name=evidence.field_name,
                    page_number=evidence.page_number,
                    source_text=evidence.source_text,
                    confidence=evidence.confidence,
                ))

    db.commit()
    db.refresh(extraction)

    return extraction.id