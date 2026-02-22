"""
CompliancePilot - FastAPI Backend
===================================
Central nervous system of the entire compliance system.
Receives decisions, triggers classification, serves dashboard.

Endpoints:
    POST /api/decisions/log      - Receive decision from wrapper
    GET  /api/decisions          - Get all decisions
    GET  /api/decisions/{id}     - Get single decision
    PUT  /api/decisions/{id}/review - Human review action
    GET  /api/agents             - Get all agents
    GET  /api/dashboard/stats    - Live dashboard statistics
    POST /api/reports/generate   - Generate PDF report

Author: CompliancePilot
Version: 1.0.0
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timezone, date
from typing import Optional, List
import json
import logging
import os

from backend.database.models import (
    DecisionLog,
    AgentRegistry,
    get_db,
    create_tables
)
from backend.api.schemas import (
    DecisionLogRequest,
    HumanReviewRequest,
    AgentRegistrationRequest,
    DashboardStats,
    APIResponse
)

# ── Logging Setup ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CompliancePilot.API")

# ── FastAPI App Initialization ─────────────────────────────────────
app = FastAPI(
    title="CompliancePilot API",
    description="AI Regulatory Governance Layer - Audit every AI decision automatically",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# ── CORS Middleware ────────────────────────────────────────────────
# Allows the frontend dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup Event ──────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Runs when server starts.
    Creates database tables if they don't exist.
    """
    create_tables()
    logger.info(" CompliancePilot API started successfully")


# ================================================================
# HEALTH CHECK
# ================================================================

@app.get("/health")
async def health_check():
    """Simple health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "CompliancePilot API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ================================================================
# DECISION LOG ENDPOINTS
# ================================================================

@app.post("/api/decisions/log")
async def log_decision(
    request: DecisionLogRequest,
    db: Session = Depends(get_db)
):
    """
    Receive and store a new AI agent decision.
    Called automatically by the Agent Wrapper SDK.
    Triggers classification engine after storing.
    """
    try:
        # ── Store decision in database ─────────────────────────────
        decision = DecisionLog(
            session_id=request.session_id,
            agent_id=request.agent_id,
            agent_version=request.agent_version,
            input_prompt=request.input_prompt,
            agent_output=request.agent_output,
            tool_calls=request.tool_calls,
            decision_context=request.decision_context,
            user_id_hash=request.user_id_hash,
            processing_time_ms=request.processing_time_ms,
            is_demo=request.is_demo,
            review_status="not_required"
        )

        db.add(decision)
        db.commit()
        db.refresh(decision)

        logger.info(
            f" Decision logged | "
            f"ID: {decision.id} | "
            f"Agent: {request.agent_id}"
        )

        # ── Trigger classification asynchronously ──────────────────
        # Import here to avoid circular imports
        from backend.classification.engine import classify_decision
        import asyncio

        # Run classification in background
        asyncio.create_task(
            classify_decision(decision.id, request.input_prompt, 
                            request.agent_output, request.agent_id,
                            request.decision_context)
        )

        # ── Update agent last active time ──────────────────────────
        agent = db.query(AgentRegistry).filter(
            AgentRegistry.agent_id == request.agent_id
        ).first()

        if agent:
            agent.last_active = datetime.now(timezone.utc)
            agent.total_decisions += 1
            db.commit()

        return {
            "success": True,
            "message": "Decision logged successfully",
            "decision_id": decision.id
        }

    except Exception as e:
        logger.error(f" Failed to log decision: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions")
async def get_decisions(
    agent_id: Optional[str] = None,
    risk_tier: Optional[str] = None,
    review_status: Optional[str] = None,
    is_demo: Optional[bool] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get all decisions with optional filters.
    Supports filtering by agent, risk level, review status.
    """
    query = db.query(DecisionLog)

    if agent_id:
        query = query.filter(DecisionLog.agent_id == agent_id)
    if risk_tier:
        query = query.filter(DecisionLog.risk_tier == risk_tier)
    if review_status:
        query = query.filter(DecisionLog.review_status == review_status)
    if is_demo is not None:
        query = query.filter(DecisionLog.is_demo == is_demo)

    total = query.count()
    decisions = query.order_by(
        desc(DecisionLog.timestamp)
    ).offset(offset).limit(limit).all()

    return {
        "success": True,
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": [d.to_dict() for d in decisions]
    }


@app.get("/api/decisions/{decision_id}")
async def get_decision(
    decision_id: int,
    db: Session = Depends(get_db)
):
    """Get a single decision by ID."""
    decision = db.query(DecisionLog).filter(
        DecisionLog.id == decision_id
    ).first()

    if not decision:
        raise HTTPException(
            status_code=404,
            detail=f"Decision {decision_id} not found"
        )

    return {
        "success": True,
        "data": decision.to_dict()
    }


@app.put("/api/decisions/{decision_id}/review")
async def review_decision(
    decision_id: int,
    request: HumanReviewRequest,
    db: Session = Depends(get_db)
):
    """
    Human reviewer approves or rejects a flagged decision.
    This action is permanently logged as audit evidence.
    Satisfies EU AI Act Article 14 human oversight requirement.
    """
    decision = db.query(DecisionLog).filter(
        DecisionLog.id == decision_id
    ).first()

    if not decision:
        raise HTTPException(
            status_code=404,
            detail=f"Decision {decision_id} not found"
        )

    if not decision.requires_human_review:
        raise HTTPException(
            status_code=400,
            detail="This decision does not require human review"
        )

    # ── Record the review permanently ─────────────────────────────
    decision.review_status = request.review_status
    decision.reviewer_id = request.reviewer_id
    decision.reviewer_justification = request.justification
    decision.reviewed_at = datetime.now(timezone.utc)

    db.commit()

    logger.info(
        f" Decision {decision_id} reviewed | "
        f"Status: {request.review_status} | "
        f"Reviewer: {request.reviewer_id}"
    )

    return {
        "success": True,
        "message": f"Decision {request.review_status} successfully",
        "decision_id": decision_id,
        "reviewed_at": decision.reviewed_at.isoformat()
    }


# ================================================================
# AGENT ENDPOINTS
# ================================================================

@app.post("/api/agents/register")
async def register_agent(
    request: AgentRegistrationRequest,
    db: Session = Depends(get_db)
):
    """Register a new AI agent in the system."""
    existing = db.query(AgentRegistry).filter(
        AgentRegistry.agent_id == request.agent_id
    ).first()

    if existing:
        return {
            "success": True,
            "message": "Agent already registered",
            "agent_id": request.agent_id
        }

    agent = AgentRegistry(
        agent_id=request.agent_id,
        agent_name=request.agent_name,
        agent_description=request.agent_description,
        agent_version=request.agent_version,
        industry=request.industry,
        use_case=request.use_case,
        risk_category=request.risk_category,
        regulatory_frameworks=request.regulatory_frameworks,
        owner_id=request.owner_id
    )

    db.add(agent)
    db.commit()

    logger.info(f" Agent registered: {request.agent_id}")

    return {
        "success": True,
        "message": "Agent registered successfully",
        "agent_id": request.agent_id
    }


@app.get("/api/agents")
async def get_agents(db: Session = Depends(get_db)):
    """Get all registered agents."""
    agents = db.query(AgentRegistry).all()
    return {
        "success": True,
        "total": len(agents),
        "data": [a.to_dict() for a in agents]
    }


# ================================================================
# DASHBOARD STATISTICS ENDPOINT
# ================================================================

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(
    agent_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Real time statistics for the dashboard.
    Aggregates all decision data into summary metrics.
    """
    query = db.query(DecisionLog)
    if agent_id:
        query = query.filter(DecisionLog.agent_id == agent_id)

    total = query.count()

    # Today's decisions
    today = date.today()
    today_count = query.filter(
        func.date(DecisionLog.timestamp) == today
    ).count()

    # Risk distribution
    risk_counts = {}
    for tier in ["Low", "Medium", "High", "Critical"]:
        risk_counts[tier] = query.filter(
            DecisionLog.risk_tier == tier
        ).count()

    # Review statistics
    pending = query.filter(
        DecisionLog.review_status == "pending"
    ).count()
    approved = query.filter(
        DecisionLog.review_status == "approved"
    ).count()
    rejected = query.filter(
        DecisionLog.review_status == "rejected"
    ).count()

    # Active agents
    agents_active = db.query(
        func.count(func.distinct(DecisionLog.agent_id))
    ).scalar()

    # Average processing time
    avg_time = db.query(
        func.avg(DecisionLog.processing_time_ms)
    ).scalar() or 0

    # Decisions by agent
    by_agent = {}
    agent_counts = db.query(
        DecisionLog.agent_id,
        func.count(DecisionLog.id)
    ).group_by(DecisionLog.agent_id).all()
    for agent_id_val, count in agent_counts:
        by_agent[agent_id_val] = count

    # Recent decisions
    recent = query.order_by(
        desc(DecisionLog.timestamp)
    ).limit(10).all()

    return {
        "success": True,
        "data": {
            "total_decisions": total,
            "decisions_today": today_count,
            "pending_reviews": pending,
            "critical_decisions": risk_counts.get("Critical", 0),
            "high_decisions": risk_counts.get("High", 0),
            "medium_decisions": risk_counts.get("Medium", 0),
            "low_decisions": risk_counts.get("Low", 0),
            "approved_reviews": approved,
            "rejected_reviews": rejected,
            "agents_active": agents_active,
            "avg_processing_time_ms": round(avg_time, 2),
            "risk_distribution": risk_counts,
            "decisions_by_agent": by_agent,
            "recent_decisions": [d.to_dict() for d in recent]
        }
    }


# ================================================================
# REPORT GENERATION ENDPOINT
# ================================================================

@app.post("/api/reports/generate")
async def generate_report(
    agent_id: Optional[str] = None,
    report_type: str = "full",
    db: Session = Depends(get_db)
):
    """
    Trigger PDF compliance report generation.
    Returns path to generated report file.
    """
    try:
        from backend.reports.generator import generate_compliance_report

        report_path = await generate_compliance_report(
            db=db,
            agent_id=agent_id,
            report_type=report_type
        )

        return {
            "success": True,
            "message": "Report generated successfully",
            "report_path": report_path
        }

    except Exception as e:
        logger.error(f" Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/download/{filename}")
async def download_report(filename: str):
    """Download a generated PDF report."""
    report_path = f"reports/{filename}"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=filename
    )