"""Tool management router."""
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.ToolOut])
def list_tools(db: Session = Depends(get_db)):
    return db.query(models.ToolConfiguration).order_by(models.ToolConfiguration.created_at).all()

@router.get("/{tool_id}", response_model=schemas.ToolOut)
def get_tool(tool_id: str, db: Session = Depends(get_db)):
    t = db.query(models.ToolConfiguration).filter_by(id=tool_id).first()
    if not t:
        raise HTTPException(404, "Tool not found")
    return t

@router.post("", response_model=schemas.ToolOut)
def create_tool(data: schemas.ToolCreate, db: Session = Depends(get_db)):
    t = models.ToolConfiguration(**data.model_dump(), status="configured")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t

@router.put("/{tool_id}", response_model=schemas.ToolOut)
def update_tool(tool_id: str, data: schemas.ToolUpdate, db: Session = Depends(get_db)):
    t = db.query(models.ToolConfiguration).filter_by(id=tool_id).first()
    if not t:
        raise HTTPException(404, "Tool not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(t, k, v)
    db.commit()
    db.refresh(t)
    return t

@router.delete("/{tool_id}")
def delete_tool(tool_id: str, db: Session = Depends(get_db)):
    t = db.query(models.ToolConfiguration).filter_by(id=tool_id).first()
    if not t:
        raise HTTPException(404, "Tool not found")
    db.delete(t)
    db.commit()
    return {"ok": True}

@router.post("/{tool_id}/check")
def check_availability(tool_id: str, db: Session = Depends(get_db)):
    t = db.query(models.ToolConfiguration).filter_by(id=tool_id).first()
    if not t:
        raise HTTPException(404, "Tool not found")
    # Tools require an external runner — we cannot execute them directly
    t.last_error = "External runner not configured. Tools require a separate execution service."
    t.status = "unavailable"
    db.commit()
    return {
        "status": "unavailable",
        "message": "This tool requires an external execution runner. Configure the runner in Integrations to enable tool execution.",
    }
