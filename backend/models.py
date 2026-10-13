"""SQLAlchemy models for SENTINEL NEXUS AI."""
import uuid
import datetime
from sqlalchemy import (
    Column, String, Text, Integer, Boolean, DateTime, ForeignKey, JSON, Float
)
from sqlalchemy.orm import relationship
from database import Base


def _uid():
    return str(uuid.uuid4())


class AgentConfiguration(Base):
    __tablename__ = "agents"
    id = Column(String, primary_key=True, default=_uid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    adapter_type = Column(String, default="custom")
    endpoint = Column(String, default="")
    status = Column(String, default="not_configured")  # not_configured, configured, connected, disconnected, running, waiting_approval, completed, failed, disabled
    capabilities = Column(JSON, default=list)
    supported_models = Column(JSON, default=list)
    runtime_requirements = Column(Text, default="")
    config = Column(JSON, default=dict)
    last_health_check = Column(DateTime, nullable=True)
    last_error = Column(Text, default="")
    enabled = Column(Boolean, default=True)
    is_custom = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class ModelConfiguration(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True, default=_uid)
    display_name = Column(String, nullable=False)
    provider = Column(String, default="ollama")  # ollama, openai, custom
    model_identifier = Column(String, default="")
    endpoint = Column(String, default="")
    status = Column(String, default="not_configured")  # not_configured, configured, connected, disconnected, disabled
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=False)
    context_config = Column(JSON, default=dict)
    capabilities = Column(JSON, default=list)
    last_health_check = Column(DateTime, nullable=True)
    last_error = Column(Text, default="")
    notes = Column(Text, default="")
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class ToolConfiguration(Base):
    __tablename__ = "tools"
    id = Column(String, primary_key=True, default=_uid)
    name = Column(String, nullable=False)
    category = Column(String, default="recon")
    description = Column(Text, default="")
    integration_method = Column(String, default="runner")
    status = Column(String, default="not_configured")  # not_configured, configured, available, unavailable, disabled
    version = Column(String, default="")
    task_types = Column(JSON, default=list)
    runtime_requirements = Column(Text, default="")
    config = Column(JSON, default=dict)
    last_execution_time = Column(DateTime, nullable=True)
    last_error = Column(Text, default="")
    required_permissions = Column(JSON, default=list)
    enabled = Column(Boolean, default=True)
    is_custom = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class Assessment(Base):
    __tablename__ = "assessments"
    id = Column(String, primary_key=True, default=_uid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    target_type = Column(String, default="host")
    target_identifier = Column(String, default="")
    scope = Column(JSON, default=list)
    excluded_targets = Column(JSON, default=list)
    environment_type = Column(String, default="lab")
    selected_agents = Column(JSON, default=list)
    selected_model_id = Column(String, ForeignKey("models.id"), nullable=True)
    selected_tools = Column(JSON, default=list)
    objective = Column(Text, default="")
    allowed_operations = Column(JSON, default=list)
    time_limit = Column(Integer, nullable=True)
    requires_approval = Column(Boolean, default=True)
    authorization_confirmed = Column(Boolean, default=False)
    notes = Column(Text, default="")
    status = Column(String, default="draft")  # draft, active, running, paused, completed, cancelled, failed
    progress = Column(Float, default=0.0)
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    tasks = relationship("TaskExecution", back_populates="assessment", cascade="all, delete-orphan")
    findings = relationship("Finding", back_populates="assessment", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="assessment")


class TaskExecution(Base):
    __tablename__ = "task_executions"
    id = Column(String, primary_key=True, default=_uid)
    assessment_id = Column(String, ForeignKey("assessments.id"), nullable=True)
    description = Column(Text, default="")
    selected_agent = Column(String, default="")
    selected_model = Column(String, default="")
    selected_tools = Column(JSON, default=list)
    scope_reference = Column(JSON, default=list)
    status = Column(String, default="queued")  # queued, running, completed, failed, cancelled, waiting_approval
    current_phase = Column(String, default="")
    progress = Column(Float, default=0.0)
    result_summary = Column(Text, default="")
    errors = Column(Text, default="")
    cancellation_status = Column(String, default="")
    approval_history = Column(JSON, default=list)
    logs = Column(JSON, default=list)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    assessment = relationship("Assessment", back_populates="tasks")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=_uid)
    title = Column(String, default="New Conversation")
    assessment_id = Column(String, ForeignKey("assessments.id"), nullable=True)
    selected_agent = Column(String, default="")
    selected_model = Column(String, default="")
    context = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")
    assessment = relationship("Assessment", back_populates="conversations")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, default=_uid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, default="")
    meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")


class Workflow(Base):
    __tablename__ = "workflows"
    id = Column(String, primary_key=True, default=_uid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    steps = Column(JSON, default=list)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    id = Column(String, primary_key=True, default=_uid)
    workflow_id = Column(String, ForeignKey("workflows.id"), nullable=False)
    assessment_id = Column(String, ForeignKey("assessments.id"), nullable=True)
    status = Column(String, default="pending")
    step_statuses = Column(JSON, default=list)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    workflow = relationship("Workflow", back_populates="executions")


class Finding(Base):
    __tablename__ = "findings"
    id = Column(String, primary_key=True, default=_uid)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    target = Column(String, default="")
    affected_component = Column(String, default="")
    severity = Column(String, default="informational")  # critical, high, medium, low, informational, unknown
    confidence = Column(String, default="low")  # low, medium, high
    evidence = Column(Text, default="")
    source_tool = Column(String, default="")
    source_agent = Column(String, default="")
    detection_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    validation_status = Column(String, default="new")  # new, under_review, confirmed, false_positive, accepted_risk, resolved
    remediation = Column(Text, default="")
    assessment_id = Column(String, ForeignKey("assessments.id"), nullable=True)
    review_status = Column(String, default="new")
    notes = Column(Text, default="")
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    assessment = relationship("Assessment", back_populates="findings")


class Report(Base):
    __tablename__ = "reports"
    id = Column(String, primary_key=True, default=_uid)
    assessment_id = Column(String, ForeignKey("assessments.id"), nullable=True)
    title = Column(String, default="Assessment Report")
    content = Column(JSON, default=dict)
    status = Column(String, default="draft")  # draft, generated, exported
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class IntegrationHealth(Base):
    __tablename__ = "integration_health"
    id = Column(String, primary_key=True, default=_uid)
    integration_name = Column(String, nullable=False)
    integration_type = Column(String, default="agent")
    status = Column(String, default="unknown")
    last_check = Column(DateTime, default=datetime.datetime.utcnow)
    error_message = Column(Text, default="")
    details = Column(JSON, default=dict)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(String, primary_key=True, default=_uid)
    event_type = Column(String, nullable=False)
    description = Column(Text, default="")
    entity_type = Column(String, default="")
    entity_id = Column(String, default="")
    meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"
    id = Column(String, primary_key=True, default=_uid)
    assessment_id = Column(String, ForeignKey("assessments.id"), nullable=True)
    task_id = Column(String, ForeignKey("task_executions.id"), nullable=True)
    operation = Column(Text, default="")
    scope = Column(JSON, default=list)
    status = Column(String, default="pending")  # pending, approved, rejected, cancelled
    decision_by = Column(String, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    decided_at = Column(DateTime, nullable=True)
