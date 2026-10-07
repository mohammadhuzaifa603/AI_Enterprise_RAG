from __future__ import annotations
from sqlalchemy.orm import Session
from app.models import AuditLog

def log_event(db: Session, action: str, user_id: str | None = None, document_id: str | None = None, query_id: str | None = None, metadata: dict | None = None):
    db.add(AuditLog(user_id=user_id, action=action, document_id=document_id, query_id=query_id, metadata_json=metadata or {}))
