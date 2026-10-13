"""Workflow router."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.WorkflowOut])
def list_workflows(db: Session = Depends(get_db)):
    return db.query(models.Workflow).order_by(models.Workflow.created_at.desc()).all()

@router.get("/{workflow_id}", response_model=schemas.WorkflowOut)
def get_workflow(workflow_id: str, db: Session = Depends(get_db)):
    w = db.query(models.Workflow).filter_by(id=workflow_id).first()
    if not w:
        raise HTTPException(404, "Workflow not found")
    return w

@router.post("", response_model=schemas.WorkflowOut)
def create_workflow(data: schemas.WorkflowCreate, db: Session = Depends(get_db)):
    w = models.Workflow(**data.model_dump())
    db.add(w)
    db.commit()
    db.refresh(w)
    return w

@router.put("/{workflow_id}", response_model=schemas.WorkflowOut)
def update_workflow(workflow_id: str, data: schemas.WorkflowUpdate, db: Session = Depends(get_db)):
    w = db.query(models.Workflow).filter_by(id=workflow_id).first()
    if not w:
        raise HTTPException(404, "Workflow not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(w, k, v)
    db.commit()
    db.refresh(w)
    return w

@router.delete("/{workflow_id}")
def delete_workflow(workflow_id: str, db: Session = Depends(get_db)):
    w = db.query(models.Workflow).filter_by(id=workflow_id).first()
    if not w:
        raise HTTPException(404, "Workflow not found")
    db.delete(w)
    db.commit()
    return {"ok": True}

@router.post("/{workflow_id}/duplicate", response_model=schemas.WorkflowOut)
def duplicate_workflow(workflow_id: str, db: Session = Depends(get_db)):
    w = db.query(models.Workflow).filter_by(id=workflow_id).first()
    if not w:
        raise HTTPException(404, "Workflow not found")
    new_w = models.Workflow(
        name=f"{w.name} (Copy)",
        description=w.description,
        steps=w.steps,
        enabled=True,
    )
    db.add(new_w)
    db.commit()
    db.refresh(new_w)
    return new_w
