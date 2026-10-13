"""Model management router."""
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import httpx
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.ModelOut])
def list_models(db: Session = Depends(get_db)):
    return db.query(models.ModelConfiguration).order_by(models.ModelConfiguration.created_at).all()

@router.get("/{model_id}", response_model=schemas.ModelOut)
def get_model(model_id: str, db: Session = Depends(get_db)):
    m = db.query(models.ModelConfiguration).filter_by(id=model_id).first()
    if not m:
        raise HTTPException(404, "Model not found")
    return m

@router.post("", response_model=schemas.ModelOut)
def create_model(data: schemas.ModelCreate, db: Session = Depends(get_db)):
    m = models.ModelConfiguration(**data.model_dump(), status="configured")
    db.add(m)
    db.commit()
    db.refresh(m)
    return m

@router.put("/{model_id}", response_model=schemas.ModelOut)
def update_model(model_id: str, data: schemas.ModelUpdate, db: Session = Depends(get_db)):
    m = db.query(models.ModelConfiguration).filter_by(id=model_id).first()
    if not m:
        raise HTTPException(404, "Model not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(m, k, v)
    db.commit()
    db.refresh(m)
    return m

@router.delete("/{model_id}")
def delete_model(model_id: str, db: Session = Depends(get_db)):
    m = db.query(models.ModelConfiguration).filter_by(id=model_id).first()
    if not m:
        raise HTTPException(404, "Model not found")
    db.delete(m)
    db.commit()
    return {"ok": True}

@router.post("/{model_id}/test")
def test_model(model_id: str, db: Session = Depends(get_db)):
    m = db.query(models.ModelConfiguration).filter_by(id=model_id).first()
    if not m:
        raise HTTPException(404, "Model not found")

    if m.provider == "ollama":
        endpoint = m.endpoint or "http://localhost:11434"
        try:
            r = httpx.get(f"{endpoint}/api/tags", timeout=10)
            if r.status_code == 200:
                tags = r.json().get("models", [])
                model_names = [t.get("name", "") for t in tags]
                m.status = "connected"
                m.last_health_check = datetime.datetime.utcnow()
                m.last_error = ""
                db.commit()
                return {
                    "status": "connected",
                    "message": f"Ollama is reachable. Available models: {', '.join(model_names) or 'none'}",
                    "available_models": model_names,
                }
            else:
                m.status = "disconnected"
                m.last_error = f"Ollama returned status {r.status_code}"
                m.last_health_check = datetime.datetime.utcnow()
                db.commit()
                return {"status": "disconnected", "message": m.last_error}
        except Exception as e:
            m.status = "disconnected"
            m.last_error = str(e)
            m.last_health_check = datetime.datetime.utcnow()
            db.commit()
            return {
                "status": "disconnected",
                "message": f"Cannot reach Ollama at {endpoint}. Ensure Ollama is running and accessible from the backend container. Error: {e}",
            }

    # OpenAI-compatible
    if m.provider in ("openai", "custom") and m.endpoint:
        try:
            r = httpx.get(f"{m.endpoint}/v1/models", timeout=10)
            m.status = "connected"
            m.last_health_check = datetime.datetime.utcnow()
            m.last_error = ""
            db.commit()
            return {"status": "connected", "message": f"Endpoint responded with status {r.status_code}"}
        except Exception as e:
            m.status = "disconnected"
            m.last_error = str(e)
            m.last_health_check = datetime.datetime.utcnow()
            db.commit()
            return {"status": "disconnected", "message": f"Connection failed: {e}"}

    m.status = "not_configured"
    m.last_error = "No endpoint configured"
    m.last_health_check = datetime.datetime.utcnow()
    db.commit()
    return {"status": "not_configured", "message": "No endpoint configured."}

@router.post("/{model_id}/activate")
def activate_model(model_id: str, db: Session = Depends(get_db)):
    m = db.query(models.ModelConfiguration).filter_by(id=model_id).first()
    if not m:
        raise HTTPException(404, "Model not found")
    # Deactivate all others
    for other in db.query(models.ModelConfiguration).all():
        other.is_active = False
    m.is_active = True
    db.commit()
    return {"ok": True, "active_model": m.display_name}
