"""Lightweight local identity layer for the portfolio build.

This is intentionally not enterprise SSO. It establishes a stable user/role
identity so authorization-aware retrieval can be demonstrated without adding
an unrelated IAM product to the project.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.retrieval.filters import get_or_create_local_user

router = APIRouter(prefix="/auth", tags=["auth"])

@router.get("/me")
def current_user(db: Session = Depends(get_db)):
    user = get_or_create_local_user(db)
    return {"id": user.id, "username": user.username, "role": user.role}

@router.get("/users")
def list_users(db: Session = Depends(get_db)):
    return [{"id": u.id, "username": u.username, "role": u.role, "active": u.is_active} for u in db.query(User).all()]
