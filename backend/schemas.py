"""Pydantic schemas for API validation."""
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


# --- Agent ---
class AgentBase(BaseModel):
    name: str
    description: str = ""
    adapter_type: str = "custom"
    endpoint: str = ""
    capabilities: list[str] = []
    supported_models: list[str] = []
    runtime_requirements: str = ""
    config: dict[str, Any] = {}
    enabled: bool = True

class AgentCreate(AgentBase):
    is_custom: bool = True

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    adapter_type: Optional[str] = None
    endpoint: Optional[str] = None
    status: Optional[str] = None
    capabilities: Optional[list[str]] = None
    supported_models: Optional[list[str]] = None
    runtime_requirements: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    enabled: Optional[bool] = None

class AgentOut(AgentBase):
    id: str
    status: str
    last_health_check: Optional[datetime] = None
    last_error: str = ""
    is_custom: bool
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- Model ---
class ModelBase(BaseModel):
    display_name: str
    provider: str = "ollama"
    model_identifier: str = ""
    endpoint: str = ""
    context_config: dict[str, Any] = {}
    capabilities: list[str] = []
    notes: str = ""
    enabled: bool = True

class ModelCreate(ModelBase):
    pass

class ModelUpdate(BaseModel):
    display_name: Optional[str] = None
    provider: Optional[str] = None
    model_identifier: Optional[str] = None
    endpoint: Optional[str] = None
    status: Optional[str] = None
    is_default: Optional[bool] = None
    is_active: Optional[bool] = None
    context_config: Optional[dict[str, Any]] = None
    capabilities: Optional[list[str]] = None
    notes: Optional[str] = None
    enabled: Optional[bool] = None

class ModelOut(ModelBase):
    id: str
    status: str
    is_default: bool
    is_active: bool
    last_health_check: Optional[datetime] = None
    last_error: str = ""
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- Tool ---
class ToolBase(BaseModel):
    name: str
    category: str = "recon"
    description: str = ""
    integration_method: str = "runner"
    version: str = ""
    task_types: list[str] = []
    runtime_requirements: str = ""
    config: dict[str, Any] = {}
    required_permissions: list[str] = []
    enabled: bool = True

class ToolCreate(ToolBase):
    is_custom: bool = True

class ToolUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    integration_method: Optional[str] = None
    status: Optional[str] = None
    version: Optional[str] = None
    task_types: Optional[list[str]] = None
    runtime_requirements: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    enabled: Optional[bool] = None

class ToolOut(ToolBase):
    id: str
    status: str
    last_execution_time: Optional[datetime] = None
    last_error: str = ""
    is_custom: bool
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- Assessment ---
class AssessmentBase(BaseModel):
    name: str
    description: str = ""
    target_type: str = "host"
    target_identifier: str = ""
    scope: list[str] = []
    excluded_targets: list[str] = []
    environment_type: str = "lab"
    selected_agents: list[str] = []
    selected_model_id: Optional[str] = None
    selected_tools: list[str] = []
    objective: str = ""
    allowed_operations: list[str] = []
    time_limit: Optional[int] = None
    requires_approval: bool = True
    authorization_confirmed: bool = False
    notes: str = ""
    is_demo: bool = False

class AssessmentCreate(AssessmentBase):
    pass

class AssessmentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    target_type: Optional[str] = None
    target_identifier: Optional[str] = None
    scope: Optional[list[str]] = None
    excluded_targets: Optional[list[str]] = None
    environment_type: Optional[str] = None
    selected_agents: Optional[list[str]] = None
    selected_model_id: Optional[str] = None
    selected_tools: Optional[list[str]] = None
    objective: Optional[str] = None
    allowed_operations: Optional[list[str]] = None
    time_limit: Optional[int] = None
    requires_approval: Optional[bool] = None
    authorization_confirmed: Optional[bool] = None
    notes: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[float] = None

class AssessmentOut(AssessmentBase):
    id: str
    status: str
    progress: float
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- Task ---
class TaskBase(BaseModel):
    assessment_id: Optional[str] = None
    description: str = ""
    selected_agent: str = ""
    selected_model: str = ""
    selected_tools: list[str] = []
    scope_reference: list[str] = []

class TaskCreate(TaskBase):
    pass

class TaskUpdate(BaseModel):
    status: Optional[str] = None
    current_phase: Optional[str] = None
    progress: Optional[float] = None
    result_summary: Optional[str] = None
    errors: Optional[str] = None
    cancellation_status: Optional[str] = None

class TaskOut(TaskBase):
    id: str
    status: str
    current_phase: str
    progress: float
    result_summary: str
    errors: str
    cancellation_status: str
    approval_history: list[Any] = []
    logs: list[Any] = []
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- Conversation ---
class ConversationBase(BaseModel):
    title: str = "New Conversation"
    assessment_id: Optional[str] = None
    selected_agent: str = ""
    selected_model: str = ""

class ConversationCreate(ConversationBase):
    pass

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    selected_agent: Optional[str] = None
    selected_model: Optional[str] = None
    context: Optional[dict[str, Any]] = None

class ConversationOut(ConversationBase):
    id: str
    context: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- ChatMessage ---
class ChatMessageCreate(BaseModel):
    conversation_id: str
    role: str = "user"
    content: str
    metadata: dict[str, Any] = {}

class ChatMessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: dict[str, Any] = {}
    created_at: datetime
    class Config:
        from_attributes = True


# --- Workflow ---
class WorkflowBase(BaseModel):
    name: str
    description: str = ""
    steps: list[Any] = []
    enabled: bool = True

class WorkflowCreate(WorkflowBase):
    pass

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[list[Any]] = None
    enabled: Optional[bool] = None

class WorkflowOut(WorkflowBase):
    id: str
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- Finding ---
class FindingBase(BaseModel):
    title: str
    description: str = ""
    target: str = ""
    affected_component: str = ""
    severity: str = "informational"
    confidence: str = "low"
    evidence: str = ""
    source_tool: str = ""
    source_agent: str = ""
    remediation: str = ""
    assessment_id: Optional[str] = None
    is_demo: bool = False

class FindingCreate(FindingBase):
    pass

class FindingUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[str] = None
    validation_status: Optional[str] = None
    review_status: Optional[str] = None
    remediation: Optional[str] = None
    notes: Optional[str] = None

class FindingOut(FindingBase):
    id: str
    detection_timestamp: datetime
    validation_status: str
    review_status: str
    notes: str
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- Report ---
class ReportBase(BaseModel):
    assessment_id: Optional[str] = None
    title: str = "Assessment Report"
    content: dict[str, Any] = {}
    is_demo: bool = False

class ReportCreate(ReportBase):
    pass

class ReportOut(ReportBase):
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- Approval ---
class ApprovalCreate(BaseModel):
    assessment_id: Optional[str] = None
    task_id: Optional[str] = None
    operation: str = ""
    scope: list[str] = []

class ApprovalOut(ApprovalCreate):
    id: str
    status: str
    decision_by: str = ""
    created_at: datetime
    decided_at: Optional[datetime] = None
    class Config:
        from_attributes = True


# --- Audit ---
class AuditEventOut(BaseModel):
    id: str
    event_type: str
    description: str
    entity_type: str
    entity_id: str
    meta: dict[str, Any] = {}
    created_at: datetime
    class Config:
        from_attributes = True


# --- Health ---
class HealthOut(BaseModel):
    status: str = "ok"
    backend: str = "online"
    database: str = "connected"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
