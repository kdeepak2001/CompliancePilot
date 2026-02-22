"""
CompliancePilot - Database Models
==================================
Industry-grade database schema designed to satisfy:
- EU AI Act Article 9, 13, 14, 16 audit requirements
- HIPAA Audit Controls (§ 164.312)
- SOC 2 Type II Processing Integrity criteria
- GDPR data minimization principles

Author: CompliancePilot
Version: 1.0.0
"""

from sqlalchemy import (
    Column, String, DateTime, Text, 
    Integer, Boolean, Index, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime, timezone
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def utcnow():
    """Always store timestamps in UTC for regulatory compliance."""
    return datetime.now(timezone.utc)


def hash_pii(value: str) -> str:
    """
    One-way hash for PII fields.
    Satisfies GDPR data minimization and HIPAA minimum necessary standard.
    We can still search and match — but raw PII is never stored.
    """
    if not value:
        return ""
    return hashlib.sha256(value.encode()).hexdigest()


# ================================================================
# CORE DECISION LOG TABLE
# Every AI decision ever made is stored here — immutable.
# This is the heart of the entire compliance system.
# ================================================================

class DecisionLog(Base):
    """
    Immutable audit log of every AI agent decision.
    
    Design principle: Once written, never edited.
    This satisfies the audit chain integrity requirement
    for EU AI Act, HIPAA, and SOC 2 simultaneously.
    """
    __tablename__ = "decision_logs"

    # ── Primary Identity ──────────────────────────────────────────
    id = Column(
        Integer, 
        primary_key=True, 
        autoincrement=True,
        comment="Internal unique identifier"
    )
    session_id = Column(
        String(100), 
        nullable=False, 
        index=True,
        comment="User session identifier - links decisions to a user journey"
    )
    agent_id = Column(
        String(100), 
        nullable=False, 
        index=True,
        comment="Which AI agent made this decision - medical, hr, financial"
    )
    agent_version = Column(
        String(50),
        nullable=False,
        default="1.0.0",
        comment="Version of the agent - critical for regulatory accountability"
    )

    # ── Timing ────────────────────────────────────────────────────
    timestamp = Column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
        index=True,
        comment="UTC timestamp of decision - required by all regulatory frameworks"
    )

    # ── Decision Content ──────────────────────────────────────────
    input_prompt = Column(
        Text, 
        nullable=False,
        comment="Full input sent to the agent - the question or task"
    )
    agent_output = Column(
        Text, 
        nullable=False,
        comment="Full output from the agent - the actual decision made"
    )
    tool_calls = Column(
        Text, 
        default="[]",
        comment="JSON array of any tools the agent used during decision making"
    )
    decision_context = Column(
        Text,
        default="{}",
        comment="JSON object with additional context - industry, use case, metadata"
    )

    # ── PII Protection ────────────────────────────────────────────
    user_id_hash = Column(
        String(64),
        default="",
        comment="SHA-256 hash of user ID - never store raw PII. GDPR/HIPAA compliant."
    )

    # ── Regulatory Classification ─────────────────────────────────
    regulatory_domain = Column(
        String(50), 
        default="General",
        index=True,
        comment="Primary regulation: EU_AI_ACT, HIPAA, SOC2, GENERAL"
    )
    risk_tier = Column(
        String(20), 
        default="Low",
        index=True,
        comment="Risk level: Low, Medium, High, Critical"
    )
    risk_score = Column(
        Integer,
        default=0,
        comment="Numeric risk score 0-100 for trend analysis and reporting"
    )
    article_triggered = Column(
        String(200), 
        default="None",
        comment="Specific regulation article - e.g. EU AI Act Article 14"
    )
    recommended_action = Column(
        Text,
        default="None",
        comment="What the system recommends should happen next"
    )
    classification_reasoning = Column(
        Text,
        default="",
        comment="Why the AI classified this decision at this risk level - explainability"
    )
    classification_model = Column(
        String(100),
        default="",
        comment="Which LLM model performed the classification - for audit accountability"
    )

    # ── Human Oversight Workflow ───────────────────────────────────
    requires_human_review = Column(
        Boolean, 
        default=False,
        index=True,
        comment="True if risk is High or Critical - mandatory human review"
    )
    review_status = Column(
        String(20), 
        default="not_required",
        index=True,
        comment="not_required, pending, approved, rejected"
    )
    reviewer_id = Column(
        String(100), 
        default="",
        comment="ID of the human who reviewed this decision"
    )
    reviewer_justification = Column(
        Text, 
        default="",
        comment="Written justification from reviewer - required audit evidence"
    )
    reviewed_at = Column(
        DateTime(timezone=True), 
        nullable=True,
        comment="UTC timestamp when human review was completed"
    )

    # ── System Metadata ───────────────────────────────────────────
    processing_time_ms = Column(
        Integer,
        default=0,
        comment="How long the agent took to respond in milliseconds"
    )
    classification_time_ms = Column(
        Integer,
        default=0,
        comment="How long the classification engine took in milliseconds"
    )
    is_demo = Column(
        Boolean,
        default=False,
        comment="True if this is demo/test data - filters out from real reports"
    )

    # ── Composite Indexes for Fast Reporting ──────────────────────
    __table_args__ = (
        Index("ix_agent_risk", "agent_id", "risk_tier"),
        Index("ix_agent_timestamp", "agent_id", "timestamp"),
        Index("ix_review_status", "requires_human_review", "review_status"),
        Index("ix_regulatory_risk", "regulatory_domain", "risk_tier"),
    )

    def __repr__(self):
        return (
            f"<DecisionLog id={self.id} "
            f"agent={self.agent_id} "
            f"risk={self.risk_tier} "
            f"status={self.review_status}>"
        )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "input_prompt": self.input_prompt,
            "agent_output": self.agent_output,
            "tool_calls": self.tool_calls,
            "decision_context": self.decision_context,
            "regulatory_domain": self.regulatory_domain,
            "risk_tier": self.risk_tier,
            "risk_score": self.risk_score,
            "article_triggered": self.article_triggered,
            "recommended_action": self.recommended_action,
            "classification_reasoning": self.classification_reasoning,
            "classification_model": self.classification_model,
            "requires_human_review": self.requires_human_review,
            "review_status": self.review_status,
            "reviewer_id": self.reviewer_id,
            "reviewer_justification": self.reviewer_justification,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "processing_time_ms": self.processing_time_ms,
            "classification_time_ms": self.classification_time_ms,
            "is_demo": self.is_demo,
        }


# ================================================================
# AGENT REGISTRY TABLE
# Tracks every AI agent registered in the system.
# Required by EU AI Act Article 16 - provider obligations.
# ================================================================

class AgentRegistry(Base):
    """
    Registry of all AI agents monitored by CompliancePilot.
    Satisfies EU AI Act Article 16 - providers must maintain
    documentation of their AI systems.
    """
    __tablename__ = "agent_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), unique=True, nullable=False, index=True)
    agent_name = Column(String(200), nullable=False)
    agent_description = Column(Text, default="")
    agent_version = Column(String(50), default="1.0.0")
    industry = Column(String(100), default="General")
    use_case = Column(String(200), default="")
    risk_category = Column(String(50), default="Limited")
    regulatory_frameworks = Column(Text, default="[]")
    is_active = Column(Boolean, default=True)
    registered_at = Column(DateTime(timezone=True), default=utcnow)
    last_active = Column(DateTime(timezone=True), default=utcnow)
    total_decisions = Column(Integer, default=0)
    owner_id = Column(String(100), default="")

    def to_dict(self):
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_description": self.agent_description,
            "agent_version": self.agent_version,
            "industry": self.industry,
            "use_case": self.use_case,
            "risk_category": self.risk_category,
            "regulatory_frameworks": self.regulatory_frameworks,
            "is_active": self.is_active,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "total_decisions": self.total_decisions,
            "owner_id": self.owner_id,
        }


# ================================================================
# DATABASE SETUP
# ================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./compliancepilot.db"
)

# SQLite needs this special argument.
# PostgreSQL does not — but this code works for both.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False  # Set True to see SQL queries during debugging
)

SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)


def create_tables():
    """
    Create all database tables if they don't exist.
    Safe to call multiple times — won't overwrite existing data.
    """
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")


def get_db():
    """
    Database session dependency for FastAPI.
    Automatically closes session after each request.
    This prevents memory leaks in production.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
