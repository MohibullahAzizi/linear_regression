"""Audit event router."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.AuditEventOut])
def list_audit_events(limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.AuditEvent).order_by(models.AuditEvent.created_at.desc()).limit(limit).all()
