"""Assessment management router."""
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.AssessmentOut])
def list_assessments(db: Session = Depends(get_db)):
    return db.query(models.Assessment).order_by(models.Assessment.created_at.desc()).all()

@router.get("/{assessment_id}", response_model=schemas.AssessmentOut)
def get_assessment(assessment_id: str, db: Session = Depends(get_db)):
    a = db.query(models.Assessment).filter_by(id=assessment_id).first()
    if not a:
        raise HTTPException(404, "Assessment not found")
    return a

@router.post("", response_model=schemas.AssessmentOut)
def create_assessment(data: schemas.AssessmentCreate, db: Session = Depends(get_db)):
    a = models.Assessment(**data.model_dump(), status="draft" if not data.authorization_confirmed else "active")
    db.add(a)
    db.commit()
    db.refresh(a)
    # Audit
    db.add(models.AuditEvent(
        event_type="assessment_created",
        description=f"Assessment '{a.name}' created",
        entity_type="assessment",
        entity_id=a.id,
    ))
    db.commit()
    return a

@router.put("/{assessment_id}", response_model=schemas.AssessmentOut)
def update_assessment(assessment_id: str, data: schemas.AssessmentUpdate, db: Session = Depends(get_db)):
    a = db.query(models.Assessment).filter_by(id=assessment_id).first()
    if not a:
        raise HTTPException(404, "Assessment not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(a, k, v)
    db.commit()
    db.refresh(a)
    return a

@router.delete("/{assessment_id}")
def delete_assessment(assessment_id: str, db: Session = Depends(get_db)):
    a = db.query(models.Assessment).filter_by(id=assessment_id).first()
    if not a:
        raise HTTPException(404, "Assessment not found")
    db.delete(a)
    db.commit()
    return {"ok": True}

@router.post("/{assessment_id}/validate-scope")
def validate_scope(assessment_id: str, target: str, db: Session = Depends(get_db)):
    a = db.query(models.Assessment).filter_by(id=assessment_id).first()
    if not a:
        raise HTTPException(404, "Assessment not found")

    scope = a.scope or []
    excluded = a.excluded_targets or []

    # Check if target is in excluded list
    for ex in excluded:
        if ex in target or target in ex:
            return {"valid": False, "reason": f"Target '{target}' is in the excluded targets list: {ex}"}

    # Check if target is within scope
    in_scope = False
    for s in scope:
        if s in target or target in s or s == "*" or target.startswith(s):
            in_scope = True
            break

    if not in_scope and scope:
        return {"valid": False, "reason": f"Target '{target}' is not within the authorized scope: {scope}"}

    return {"valid": True, "reason": "Target is within the authorized scope."}
