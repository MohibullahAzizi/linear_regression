"""Agent management router."""
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import httpx
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.AgentOut])
def list_agents(db: Session = Depends(get_db)):
    return db.query(models.AgentConfiguration).order_by(models.AgentConfiguration.created_at).all()

@router.get("/{agent_id}", response_model=schemas.AgentOut)
def get_agent(agent_id: str, db: Session = Depends(get_db)):
    agent = db.query(models.AgentConfiguration).filter_by(id=agent_id).first()
    if not agent:
        raise HTTPException(404, "Agent not found")
    return agent

@router.post("", response_model=schemas.AgentOut)
def create_agent(data: schemas.AgentCreate, db: Session = Depends(get_db)):
    agent = models.AgentConfiguration(**data.model_dump(), status="configured")
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent

@router.put("/{agent_id}", response_model=schemas.AgentOut)
def update_agent(agent_id: str, data: schemas.AgentUpdate, db: Session = Depends(get_db)):
    agent = db.query(models.AgentConfiguration).filter_by(id=agent_id).first()
    if not agent:
        raise HTTPException(404, "Agent not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(agent, k, v)
    db.commit()
    db.refresh(agent)
    return agent

@router.delete("/{agent_id}")
def delete_agent(agent_id: str, db: Session = Depends(get_db)):
    agent = db.query(models.AgentConfiguration).filter_by(id=agent_id).first()
    if not agent:
        raise HTTPException(404, "Agent not found")
    db.delete(agent)
    db.commit()
    return {"ok": True}

@router.post("/{agent_id}/test")
def test_connection(agent_id: str, db: Session = Depends(get_db)):
    agent = db.query(models.AgentConfiguration).filter_by(id=agent_id).first()
    if not agent:
        raise HTTPException(404, "Agent not found")

    # Orchestrator is always "configured"
    if agent.adapter_type == "orchestrator":
        agent.status = "connected"
        agent.last_health_check = datetime.datetime.utcnow()
        agent.last_error = ""
        db.commit()
        return {"status": "connected", "message": "Master Security Agent is operational (internal)."}

    # For API-based agents, try to reach the endpoint
    if agent.endpoint and agent.endpoint.startswith("http"):
        try:
            r = httpx.get(agent.endpoint, timeout=10)
            agent.status = "connected"
            agent.last_health_check = datetime.datetime.utcnow()
            agent.last_error = ""
            db.commit()
            return {"status": "connected", "message": f"Endpoint responded with status {r.status_code}"}
        except Exception as e:
            agent.status = "disconnected"
            agent.last_health_check = datetime.datetime.utcnow()
            agent.last_error = str(e)
            db.commit()
            return {"status": "disconnected", "message": f"Connection failed: {e}"}

    # No endpoint configured
    agent.status = "not_configured"
    agent.last_health_check = datetime.datetime.utcnow()
    agent.last_error = "No endpoint configured"
    db.commit()
    return {"status": "not_configured", "message": "No endpoint configured. Please set the API endpoint in the agent configuration."}
