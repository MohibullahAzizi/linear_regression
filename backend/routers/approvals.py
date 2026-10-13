"""Approval request router."""
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.ApprovalOut])
def list_approvals(status: str = None, db: Session = Depends(get_db)):
    q = db.query(models.ApprovalRequest)
    if status:
        q = q.filter_by(status=status)
    return q.order_by(models.ApprovalRequest.created_at.desc()).all()

@router.post("", response_model=schemas.ApprovalOut)
def create_approval(data: schemas.ApprovalCreate, db: Session = Depends(get_db)):
    ap = models.ApprovalRequest(**data.model_dump())
    db.add(ap)
    db.commit()
    db.refresh(ap)
    return ap

@router.post("/{approval_id}/approve", response_model=schemas.ApprovalOut)
def approve_request(approval_id: str, db: Session = Depends(get_db)):
    ap = db.query(models.ApprovalRequest).filter_by(id=approval_id).first()
    if not ap:
        raise HTTPException(404, "Approval request not found")
    ap.status = "approved"
    ap.decided_at = datetime.datetime.utcnow()
    db.commit()
    db.add(models.AuditEvent(
        event_type="approval_granted",
        description=f"Approval granted for: {ap.operation}",
        entity_type="approval",
        entity_id=ap.id,
    ))
    db.commit()
    db.refresh(ap)
    return ap

@router.post("/{approval_id}/reject", response_model=schemas.ApprovalOut)
def reject_request(approval_id: str, db: Session = Depends(get_db)):
    ap = db.query(models.ApprovalRequest).filter_by(id=approval_id).first()
    if not ap:
        raise HTTPException(404, "Approval request not found")
    ap.status = "rejected"
    ap.decided_at = datetime.datetime.utcnow()
    db.commit()
    db.add(models.AuditEvent(
        event_type="approval_rejected",
        description=f"Approval rejected for: {ap.operation}",
        entity_type="approval",
        entity_id=ap.id,
    ))
    db.commit()
    db.refresh(ap)
    return ap
