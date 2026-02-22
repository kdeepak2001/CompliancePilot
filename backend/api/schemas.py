"""
CompliancePilot - API Schemas
==============================
Pydantic models for request validation and response formatting.
Every API endpoint uses these schemas to ensure data integrity.

Industry Standard: Input validation at API boundary
prevents bad data from ever reaching the database.

Author: CompliancePilot
Version: 1.0.0
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ================================================================
# ENUMS — Fixed value sets for regulatory fields
# ================================================================

class RegulatoryDomain(str, Enum):
    EU_AI_ACT = "EU_AI_ACT"
    HIPAA = "HIPAA"
    SOC2 = "SOC2"
    GENERAL = "General"


class RiskTier(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ReviewStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ================================================================
# REQUEST SCHEMAS — What the API receives
# ================================================================

class DecisionLogRequest(BaseModel):
    """
    Schema for incoming decision log from Agent Wrapper SDK.
    Every field is validated before touching the database.
    """
    session_id: str = Field(..., min_length=1, max_length=100)
    agent_id: str = Field(..., min_length=1, max_length=100)
    agent_version: str = Field(default="1.0.0", max_length=50)
    timestamp: Optional[str] = None
    input_prompt: str = Field(..., min_length=1)
    agent_output: str = Field(..., min_length=1)
    tool_calls: str = Field(default="[]")
    decision_context: str = Field(default="{}")
    user_id_hash: str = Field(default="")
    processing_time_ms: int = Field(default=0, ge=0)
    is_demo: bool = Field(default=False)

    class Config:
        str_strip_whitespace = True


class HumanReviewRequest(BaseModel):
    """
    Schema for human reviewer approving or rejecting a decision.
    Justification is mandatory — regulators require written reasoning.
    """
    reviewer_id: str = Field(..., min_length=1, max_length=100)
    review_status: str = Field(..., pattern="^(approved|rejected)$")
    justification: str = Field(..., min_length=10, max_length=2000)


class AgentRegistrationRequest(BaseModel):
    """
    Schema for registering a new AI agent in the system.
    """
    agent_id: str = Field(..., min_length=1, max_length=100)
    agent_name: str = Field(..., min_length=1, max_length=200)
    agent_description: str = Field(default="")
    agent_version: str = Field(default="1.0.0")
    industry: str = Field(default="General")
    use_case: str = Field(default="")
    risk_category: str = Field(default="Limited")
    regulatory_frameworks: str = Field(default="[]")
    owner_id: str = Field(default="")


# ================================================================
# RESPONSE SCHEMAS — What the API returns
# ================================================================

class DecisionLogResponse(BaseModel):
    """Standard response for a single decision log."""
    id: int
    session_id: str
    agent_id: str
    agent_version: str
    timestamp: Optional[str]
    input_prompt: str
    agent_output: str
    tool_calls: str
    regulatory_domain: str
    risk_tier: str
    risk_score: int
    article_triggered: str
    recommended_action: str
    classification_reasoning: str
    requires_human_review: bool
    review_status: str
    reviewer_id: str
    reviewer_justification: str
    reviewed_at: Optional[str]
    processing_time_ms: int
    is_demo: bool

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    """Real time statistics for the dashboard."""
    total_decisions: int
    decisions_today: int
    pending_reviews: int
    critical_decisions: int
    high_decisions: int
    medium_decisions: int
    low_decisions: int
    approved_reviews: int
    rejected_reviews: int
    agents_active: int
    avg_processing_time_ms: float
    risk_distribution: dict
    decisions_by_agent: dict
    recent_decisions: List[dict]


class APIResponse(BaseModel):
    """
    Standard wrapper for all API responses.
    Consistent format across every endpoint.
    Makes frontend integration predictable.
    """
    success: bool
    message: str
    data: Optional[dict] = None
    errors: Optional[List[str]] = None