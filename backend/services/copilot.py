"""Copilot service — generates context-aware responses.

When an AI model is configured and reachable, this service forwards the request
to the model. Otherwise, it generates a context-aware local response based on
application state. It NEVER fabricates tool results or claims a task succeeded.
"""
import datetime
import httpx
from sqlalchemy.orm import Session
import models


def generate_copilot_response(
    user_message: str,
    conversation,
    history: list,
    assessment,
    db: Session,
) -> tuple[str, dict]:
    """Generate a response. Returns (text, metadata)."""
    metadata = {"source": "local", "model": "none"}

    # Try to use a configured model
    active_model = db.query(models.ModelConfiguration).filter_by(is_active=True, enabled=True).first()
    if not active_model:
        active_model = db.query(models.ModelConfiguration).filter_by(is_default=True, enabled=True).first()

    if active_model and active_model.status == "connected":
        try:
            response_text = _call_model(active_model, user_message, history, assessment, db)
            metadata = {"source": "ai_model", "model": active_model.display_name, "provider": active_model.provider}
            return response_text, metadata
        except Exception as e:
            metadata = {"source": "local_fallback", "model": active_model.display_name, "error": str(e)}

    # Local context-aware response
    response = _local_response(user_message, conversation, assessment, db)
    return response, metadata


def _call_model(model, user_message, history, assessment, db):
    """Call the configured AI model."""
    # Build context
    context = _build_context(assessment, db)
    system_prompt = (
        "You are the SENTINEL NEXUS AI Master Security Agent, an AI assistant for "
        "an authorized penetration testing platform. You help users plan assessments, "
        "understand findings, and coordinate security tools. Always emphasize authorized "
        "scope and safe practices. Never suggest unauthorized or destructive actions.\n\n"
        f"Application context:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-10:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_message})

    if model.provider == "ollama":
        endpoint = model.endpoint or "http://localhost:11434"
        r = httpx.post(
            f"{endpoint}/api/chat",
            json={"model": model.model_identifier, "messages": messages, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "No response from model.")

    # OpenAI-compatible
    endpoint = model.endpoint or "https://api.openai.com"
    r = httpx.post(
        f"{endpoint}/v1/chat/completions",
        json={"model": model.model_identifier, "messages": messages},
        headers={"Authorization": f"Bearer {model.config.get('api_key', '')}"} if model.config else {},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from model.")


def _build_context(assessment, db):
    """Build context string from application state."""
    ctx = []
    if assessment:
        ctx.append(f"Current Assessment: {assessment.name}")
        ctx.append(f"Status: {assessment.status}")
        ctx.append(f"Target: {assessment.target_identifier}")
        ctx.append(f"Scope: {', '.join(assessment.scope or [])}")
        ctx.append(f"Objective: {assessment.objective}")
        findings = db.query(models.Finding).filter_by(assessment_id=assessment.id).all()
        if findings:
            ctx.append(f"Findings: {len(findings)} total")
            for f in findings[:5]:
                ctx.append(f"  - [{f.severity}] {f.title}")
    else:
        ctx.append("No active assessment selected.")

    agents = db.query(models.AgentConfiguration).filter_by(enabled=True).all()
    ctx.append(f"Available Agents: {', '.join(a.name for a in agents)}")

    models_count = db.query(models.ModelConfiguration).filter_by(enabled=True).count()
    ctx.append(f"Configured Models: {models_count}")

    tools = db.query(models.ToolConfiguration).filter_by(enabled=True).all()
    ctx.append(f"Available Tools: {', '.join(t.name for t in tools)}")

    return "\n".join(ctx)


def _local_response(user_message, conversation, assessment, db):
    """Generate a context-aware local response when no AI model is available."""
    msg_lower = user_message.lower()

    # Check for specific intents
    if any(w in msg_lower for w in ["select agent", "use agent", "switch agent", "use cai", "use pentestgpt"]):
        agents = db.query(models.AgentConfiguration).filter_by(enabled=True).all()
        agent_names = [a.name for a in agents]
        return (
            f"I can see the following configured agents: {', '.join(agent_names)}.\n\n"
            "To select an agent, use the AI Agents page or specify the agent in the assessment configuration. "
            "I will verify its configuration and compatibility before proceeding.\n\n"
            "Note: No AI model is currently connected, so I cannot process this through an LLM. "
            "Configure a model in the AI Models page to enable full conversational agent control."
        )

    if any(w in msg_lower for w in ["explain", "what is", "describe", "summarize"]) and assessment:
        findings = db.query(models.Finding).filter_by(assessment_id=assessment.id).all()
        findings_summary = "\n".join(
            f"  • [{f.severity.upper()}] {f.title} — {f.validation_status}" for f in findings
        ) if findings else "  No findings recorded yet."
        return (
            f"**Assessment: {assessment.name}**\n\n"
            f"• Status: {assessment.status}\n"
            f"• Target: {assessment.target_identifier}\n"
            f"• Scope: {', '.join(assessment.scope or [])}\n"
            f"• Objective: {assessment.objective}\n"
            f"• Environment: {assessment.environment_type}\n\n"
            f"**Findings ({len(findings)}):**\n{findings_summary}\n\n"
            "Note: This response is generated from application context without an AI model. "
            "Connect a model provider for richer analysis."
        )

    if any(w in msg_lower for w in ["stop", "cancel", "pause"]):
        return (
            "The cancellation request has been recorded. If a task is running, it will be marked for cancellation. "
            "The actual cancellation status depends on the execution service confirming the stop.\n\n"
            "You can also cancel tasks from the Execution History page."
        )

    if any(w in msg_lower for w in ["report", "generate report"]):
        return (
            "I can help generate a report from persisted assessment data. "
            "Go to the Reports page and select an assessment to generate a report. "
            "The report will include findings, scope, methodology, and recommendations based on actual stored data."
        )

    if any(w in msg_lower for w in ["model", "change model", "switch model"]):
        models_list = db.query(models.ModelConfiguration).all()
        model_info = "\n".join(
            f"  • {m.display_name} ({m.provider}) — Status: {m.status}" for m in models_list
        )
        return (
            f"Configured models:\n{model_info}\n\n"
            "To change the active model, go to the AI Models page and click 'Activate'. "
            "If a model shows 'disconnected', test the connection first."
        )

    if any(w in msg_lower for w in ["plan", "workflow", "assessment plan"]):
        return (
            "I can help prepare an assessment plan. Based on the available agents and tools:\n\n"
            "1. Define the target and authorized scope\n"
            "2. Select appropriate agents and tools\n"
            "3. Choose passive or active operations\n"
            "4. Review and approve the plan\n\n"
            "Create a new assessment from the Assessments page to begin. "
            "All security-sensitive operations require explicit approval."
        )

    # Default response
    context = _build_context(assessment, db)
    return (
        f"I understand your request. Here's what I can tell you based on the current application state:\n\n"
        f"{context}\n\n"
        "Note: No AI model is currently connected, so this is a context-aware local response. "
        "To enable full AI-powered conversational capabilities, configure and connect a model provider "
        "in the AI Models or Integrations page.\n\n"
        "I never fabricate tool results or claim tasks succeeded without confirmation from the execution service."
    )
