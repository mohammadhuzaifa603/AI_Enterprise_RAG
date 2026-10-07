"""
Bulk-ingest every PDF in sample_documents/ (or a given directory) directly
against the database - useful for demos/tests without going through the
HTTP API.

Usage:
    python scripts/ingest.py [directory] [--doc-type generic|invoice]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.database import init_db, session_scope
from app.models import Document
from app.services.ingestion_pipeline import IngestionResult, process_document
from app.storage import save_document

INVOICE_KEYWORDS = ("invoice",)


def guess_doc_type(filename: str) -> str:
    lower = filename.lower()
    return "invoice" if any(k in lower for k in INVOICE_KEYWORDS) else "generic"


def ingest_pdf(db, pdf_path: Path, doc_type: str) -> tuple[Document, IngestionResult]:
    """Save *pdf_path* via the storage abstraction and process it.

    Returns ``(Document, IngestionResult)``.  Reuses ``save_document()``
    so that encryption configuration is respected and ``storage_encrypted``
    / ``file_sha256`` are set correctly — never hard-coded.
    """
    raw_bytes = pdf_path.read_bytes()
    file_path, file_digest, encrypted = save_document(raw_bytes, pdf_path.name)

    document = Document(
        filename=Path(file_path).name,
        original_filename=pdf_path.name,
        file_path=file_path,
        file_sha256=file_digest,
        storage_encrypted=encrypted,
        doc_type=doc_type,
        status="uploaded",
    )
    db.add(document)
    db.flush()
    result = process_document(db, document)
    db.refresh(document)
    return document, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="sample_documents")
    parser.add_argument("--doc-type", choices=["generic", "invoice", "auto"], default="auto")
    args = parser.parse_args()

    init_db()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        sys.exit(1)

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in {directory}. Ingesting...")
    for pdf_path in pdf_files:
        doc_type = args.doc_type if args.doc_type != "auto" else guess_doc_type(pdf_path.name)

        with session_scope() as db:
            document, result = ingest_pdf(db, pdf_path, doc_type)
            print(
                f"  {pdf_path.name} ({doc_type}) -> status={result.status} "
                f"pages={result.num_pages} chunks={result.num_chunks} "
                f"ocr_pages={result.ocr_pages} tables={result.tables_found}"
                + (f" extraction={result.extraction_id}" if result.extraction_id else "")
            )

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
