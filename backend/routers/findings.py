"""Findings router."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.FindingOut])
def list_findings(
    assessment_id: str = None,
    severity: str = None,
    status: str = None,
    db: Session = Depends(get_db),
):
    q = db.query(models.Finding)
    if assessment_id:
        q = q.filter_by(assessment_id=assessment_id)
    if severity:
        q = q.filter(models.Finding.severity == severity)
    if status:
        q = q.filter(models.Finding.validation_status == status)
    return q.order_by(models.Finding.created_at.desc()).all()

@router.get("/{finding_id}", response_model=schemas.FindingOut)
def get_finding(finding_id: str, db: Session = Depends(get_db)):
    f = db.query(models.Finding).filter_by(id=finding_id).first()
    if not f:
        raise HTTPException(404, "Finding not found")
    return f

@router.post("", response_model=schemas.FindingOut)
def create_finding(data: schemas.FindingCreate, db: Session = Depends(get_db)):
    f = models.Finding(**data.model_dump())
    db.add(f)
    db.commit()
    db.refresh(f)
    return f

@router.put("/{finding_id}", response_model=schemas.FindingOut)
def update_finding(finding_id: str, data: schemas.FindingUpdate, db: Session = Depends(get_db)):
    f = db.query(models.Finding).filter_by(id=finding_id).first()
    if not f:
        raise HTTPException(404, "Finding not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(f, k, v)
    db.commit()
    db.refresh(f)
    return f

@router.delete("/{finding_id}")
def delete_finding(finding_id: str, db: Session = Depends(get_db)):
    f = db.query(models.Finding).filter_by(id=finding_id).first()
    if not f:
        raise HTTPException(404, "Finding not found")
    db.delete(f)
    db.commit()
    return {"ok": True}
