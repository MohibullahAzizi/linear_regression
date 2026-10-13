"""SENTINEL NEXUS AI — FastAPI backend entry point."""
import os
import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import engine, Base, get_db
from seed import seed_database
from routers import agents, models, tools, assessments, tasks, conversations, workflows, findings, reports, health, audit, approvals

# Create tables and seed
Base.metadata.create_all(bind=engine)
seed_database()

app = FastAPI(
    title="SENTINEL NEXUS AI API",
    description="AI-Powered Penetration Testing Tool Integration Platform",
    version="1.0.0",
)

# CORS
cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(tools.router, prefix="/api/tools", tags=["tools"])
app.include_router(assessments.router, prefix="/api/assessments", tags=["assessments"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])
app.include_router(workflows.router, prefix="/api/workflows", tags=["workflows"])
app.include_router(findings.router, prefix="/api/findings", tags=["findings"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(audit.router, prefix="/api/audit", tags=["audit"])
app.include_router(approvals.router, prefix="/api/approvals", tags=["approvals"])


@app.get("/")
def root():
    return {"name": "SENTINEL NEXUS AI", "version": "1.0.0", "status": "online"}
