from __future__ import annotations
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.config import settings
from app.database import get_db
from app.models import Document, DocumentPermission, User
from app.retrieval.filters import get_or_create_local_user
from app.schemas import DocumentDeleteResponse, DocumentOut, DocumentProcessResult
from app.services.ingestion_pipeline import process_document
from app.storage import save_document

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentOut)
def upload_document(file: UploadFile = File(...), doc_type: str = Form("generic"), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    data = file.file.read(settings.max_upload_bytes + 1)
    file.file.close()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_bytes // (1024*1024)} MB limit.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF payload.")

    try:
        path, digest, encrypted = save_document(data, file.filename or "document.pdf")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    user = get_or_create_local_user(db)
    document = Document(
        filename=Path(path).name,
        original_filename=file.filename or Path(path).name,
        file_path=path,
        file_sha256=digest,
        storage_encrypted=encrypted,
        owner_id=user.id,
        doc_type=doc_type,
        status="uploaded",
    )
    db.add(document)
    db.flush()
    db.add(DocumentPermission(document_id=document.id, user_id=user.id, permission="READ"))
    db.commit()
    db.refresh(document)
    return document

@router.post("/process", response_model=DocumentProcessResult)
def process_document_endpoint(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if document.status == "processing":
        raise HTTPException(status_code=409, detail="Document is already being processed.")
    return process_document(db, document)

@router.get("", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).order_by(Document.created_at.desc()).all()

@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    user = get_or_create_local_user(db)

    if user.role != "ADMIN" and document.owner_id != user.id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to delete this document.",
        )

    original_filename = document.original_filename

    try:
        file_path = Path(document.file_path)
        file_path.unlink(missing_ok=True)
    except OSError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove stored file: {exc}",
        ) from exc

    # Force-load child relationships so SQLAlchemy's cascade
    # delete-orphan can issue DELETEs for pages, chunks,
    # extractions, and permissions in the correct order.
    document.permissions
    document.chunks
    document.pages
    document.extractions

    db.delete(document)
    db.commit()

    return DocumentDeleteResponse(
        id=document_id,
        original_filename=original_filename,
        deleted=True,
        detail=f"Document '{original_filename}' and all associated records "
        f"(pages, chunks, embeddings, extractions, field evidence, "
        f"reviews, and permissions) were removed.",
    )
