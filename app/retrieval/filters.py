"""Authorization-aware retrieval filters."""
from __future__ import annotations

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentPermission, User

DEFAULT_LOCAL_USER = "local-user"


def get_or_create_local_user(db: Session) -> User:
    user = db.query(User).filter(User.username == DEFAULT_LOCAL_USER).first()
    if user:
        return user
    user = User(username=DEFAULT_LOCAL_USER, role="ADMIN")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def accessible_chunk_query(db: Session, user_id: str | None = None):
    """Return chunks restricted to documents the user may read.

    Admins can read all documents. Regular users can read owned documents
    or documents explicitly granted through document_permissions.
    """
    user = db.get(User, user_id) if user_id else get_or_create_local_user(db)
    if user is None:
        user = get_or_create_local_user(db)

    if user.role == "ADMIN":
        return db.query(Chunk)

    allowed = (
        db.query(DocumentPermission.document_id)
        .filter(DocumentPermission.user_id == user.id, DocumentPermission.permission == "READ")
    )
    return db.query(Chunk).join(Document, Chunk.document_id == Document.id).filter(
        or_(Document.owner_id == user.id, Chunk.document_id.in_(allowed))
    )
