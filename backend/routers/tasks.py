"""Task execution router."""
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.TaskOut])
def list_tasks(assessment_id: str = None, db: Session = Depends(get_db)):
    q = db.query(models.TaskExecution)
    if assessment_id:
        q = q.filter_by(assessment_id=assessment_id)
    return q.order_by(models.TaskExecution.created_at.desc()).all()

@router.get("/{task_id}", response_model=schemas.TaskOut)
def get_task(task_id: str, db: Session = Depends(get_db)):
    t = db.query(models.TaskExecution).filter_by(id=task_id).first()
    if not t:
        raise HTTPException(404, "Task not found")
    return t

@router.post("", response_model=schemas.TaskOut)
def create_task(data: schemas.TaskCreate, db: Session = Depends(get_db)):
    # Validate scope if assessment is linked
    if data.assessment_id:
        a = db.query(models.Assessment).filter_by(id=data.assessment_id).first()
        if not a:
            raise HTTPException(404, "Assessment not found")
        if not a.authorization_confirmed:
            raise HTTPException(400, "Assessment authorization has not been confirmed. Cannot create tasks.")

    t = models.TaskExecution(
        **data.model_dump(),
        status="queued",
        current_phase="Initialization",
    )
    db.add(t)
    db.commit()
    db.refresh(t)

    # Audit
    db.add(models.AuditEvent(
        event_type="task_created",
        description=f"Task created: {data.description}",
        entity_type="task",
        entity_id=t.id,
    ))
    db.commit()
    return t

@router.put("/{task_id}", response_model=schemas.TaskOut)
def update_task(task_id: str, data: schemas.TaskUpdate, db: Session = Depends(get_db)):
    t = db.query(models.TaskExecution).filter_by(id=task_id).first()
    if not t:
        raise HTTPException(404, "Task not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(t, k, v)
    if data.status == "running" and not t.started_at:
        t.started_at = datetime.datetime.utcnow()
    if data.status in ("completed", "failed", "cancelled"):
        t.ended_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(t)
    return t

@router.post("/{task_id}/cancel")
def cancel_task(task_id: str, db: Session = Depends(get_db)):
    t = db.query(models.TaskExecution).filter_by(id=task_id).first()
    if not t:
        raise HTTPException(404, "Task not found")
    if t.status in ("completed", "failed", "cancelled"):
        return {"status": t.status, "message": f"Task is already {t.status}. Cannot cancel."}
    t.cancellation_status = "requested"
    t.status = "cancelled"
    t.ended_at = datetime.datetime.utcnow()
    db.commit()
    db.add(models.AuditEvent(
        event_type="task_cancelled",
        description=f"Task {task_id} cancellation requested",
        entity_type="task",
        entity_id=task_id,
    ))
    db.commit()
    return {"status": "cancelled", "message": "Cancellation request submitted. Task has been marked as cancelled."}
