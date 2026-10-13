"""Reports router."""
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models, schemas

router = APIRouter()

@router.get("", response_model=list[schemas.ReportOut])
def list_reports(assessment_id: str = None, db: Session = Depends(get_db)):
    q = db.query(models.Report)
    if assessment_id:
        q = q.filter_by(assessment_id=assessment_id)
    return q.order_by(models.Report.created_at.desc()).all()

@router.get("/{report_id}", response_model=schemas.ReportOut)
def get_report(report_id: str, db: Session = Depends(get_db)):
    r = db.query(models.Report).filter_by(id=report_id).first()
    if not r:
        raise HTTPException(404, "Report not found")
    return r

@router.post("", response_model=schemas.ReportOut)
def create_report(data: schemas.ReportCreate, db: Session = Depends(get_db)):
    r = models.Report(**data.model_dump(), status="draft")
    db.add(r)
    db.commit()
    db.refresh(r)
    return r

@router.post("/generate/{assessment_id}", response_model=schemas.ReportOut)
def generate_report(assessment_id: str, db: Session = Depends(get_db)):
    a = db.query(models.Assessment).filter_by(id=assessment_id).first()
    if not a:
        raise HTTPException(404, "Assessment not found")

    findings = db.query(models.Finding).filter_by(assessment_id=assessment_id).all()
    severity_counts = {}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    content = {
        "report_id": f"RPT-{assessment_id[:8].upper()}",
        "executive_summary": f"This report covers the security assessment '{a.name}' conducted on {a.created_at.strftime('%Y-%m-%d')}.",
        "assessment_objectives": a.objective,
        "scope": a.scope or [],
        "scope_limitations": a.excluded_targets or [],
        "methodology": "The assessment followed a structured approach: reconnaissance, vulnerability scanning, analysis, and verification.",
        "tools_used": a.selected_tools or [],
        "agents_used": a.selected_agents or [],
        "assessment_date": a.created_at.isoformat(),
        "findings_summary": {
            "total": len(findings),
            "by_severity": severity_counts,
        },
        "detailed_findings": [
            {
                "title": f.title,
                "severity": f.severity,
                "confidence": f.confidence,
                "description": f.description,
                "evidence": f.evidence,
                "remediation": f.remediation,
                "validation_status": f.validation_status,
                "target": f.target,
            }
            for f in findings
        ],
        "remediation_recommendations": "Prioritize critical and high severity findings. Follow the remediation guidance for each finding.",
        "limitations": "This assessment was conducted within the defined scope. Findings outside the scope were not evaluated.",
        "conclusion": f"Assessment '{a.name}' is {a.status}. {len(findings)} findings were identified.",
        "is_demo": a.is_demo,
    }

    r = models.Report(
        assessment_id=assessment_id,
        title=f"Report: {a.name}",
        content=content,
        status="generated",
        is_demo=a.is_demo,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r

@router.put("/{report_id}", response_model=schemas.ReportOut)
def update_report(report_id: str, data: dict, db: Session = Depends(get_db)):
    r = db.query(models.Report).filter_by(id=report_id).first()
    if not r:
        raise HTTPException(404, "Report not found")
    r.content = data.get("content", r.content)
    r.title = data.get("title", r.title)
    db.commit()
    db.refresh(r)
    return r

@router.delete("/{report_id}")
def delete_report(report_id: str, db: Session = Depends(get_db)):
    r = db.query(models.Report).filter_by(id=report_id).first()
    if not r:
        raise HTTPException(404, "Report not found")
    db.delete(r)
    db.commit()
    return {"ok": True}
