"""
CompliancePilot - PDF Compliance Report Generator
===================================================
Generates professional audit-ready PDF reports.

Report Types:
    1. AI System Card - EU AI Act Article 13
    2. Decision Audit Trail - HIPAA Section 164.312
    3. Human Oversight Record - EU AI Act Article 14
    4. Risk Assessment Summary - EU AI Act Article 9
    5. Full Compliance Report - All four combined

Author: CompliancePilot
Version: 1.0.0
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from backend.database.models import DecisionLog, AgentRegistry

os.makedirs("reports", exist_ok=True)

COLORS = {
    "primary": colors.HexColor("#1a3c5e"),
    "secondary": colors.HexColor("#2d6a9f"),
    "critical": colors.HexColor("#c0392b"),
    "high": colors.HexColor("#e67e22"),
    "medium": colors.HexColor("#f1c40f"),
    "low": colors.HexColor("#27ae60"),
    "light_gray": colors.HexColor("#f5f5f5"),
    "border": colors.HexColor("#dddddd"),
    "white": colors.white,
    "black": colors.black,
    "text": colors.HexColor("#2c3e50")
}

RISK_COLORS = {
    "Critical": COLORS["critical"],
    "High": COLORS["high"],
    "Medium": COLORS["medium"],
    "Low": COLORS["low"]
}


def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        fontSize=24,
        fontName="Helvetica-Bold",
        textColor=COLORS["primary"],
        alignment=TA_CENTER,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        fontSize=12,
        fontName="Helvetica",
        textColor=COLORS["secondary"],
        alignment=TA_CENTER,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=14,
        fontName="Helvetica-Bold",
        textColor=COLORS["primary"],
        spaceBefore=16,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="SubHeader",
        fontSize=11,
        fontName="Helvetica-Bold",
        textColor=COLORS["secondary"],
        spaceBefore=10,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        name="SmallText",
        fontSize=8,
        fontName="Helvetica",
        textColor=COLORS["text"],
        spaceAfter=2
    ))

    styles.add(ParagraphStyle(
        name="DisclaimerText",
        fontSize=7,
        fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#888888"),
        spaceAfter=2,
        leading=10
    ))

    styles.add(ParagraphStyle(
        name="CenterText",
        fontSize=9,
        fontName="Helvetica",
        textColor=COLORS["text"],
        alignment=TA_CENTER
    ))

    styles["BodyText"].fontSize = 9
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].textColor = COLORS["text"]
    styles["BodyText"].spaceAfter = 4
    styles["BodyText"].leading = 14

    return styles


def build_header(styles, report_type, agent_id):
    elements = []
    elements.append(Spacer(1, 20))
    # Added a second ")" below to close the .append() call
    elements.append(Paragraph("CompliancePilot", styles["ReportTitle"])) 
    elements.append(Spacer(1, 16))
    # Added "))" below to close both Paragraph and .append()
    elements.append(Paragraph(
        "AI Regulatory Governance System",
        styles["ReportSubtitle"]

    ))
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=2, color=COLORS["primary"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(report_type, styles["SectionHeader"]))

    meta_data = [
        ["Generated:", datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")],
        ["Agent ID:", agent_id or "All Agents"],
        ["Report Version:", "1.0.0"],
        ["Classification:", "CONFIDENTIAL - COMPLIANCE DOCUMENT"]
    ]

    meta_table = Table(meta_data, colWidths=[1.5*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), COLORS["text"]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    elements.append(meta_table)
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=COLORS["border"]))
    elements.append(Spacer(1, 12))
    return elements


def build_disclaimer(styles):
    elements = []
    elements.append(HRFlowable(width="100%", thickness=0.5, color=COLORS["border"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("LEGAL DISCLAIMER", styles["SubHeader"]))
    disclaimer_text = (
        "This report is generated by an AI-assisted compliance system and "
        "constitutes a compliance evidence aid only. It does not constitute "
        "legal, medical, or financial advice. Classifications are "
        "AI-assisted recommendations requiring human validation for High "
        "and Critical risk decisions. For Indian operations: DPDP Act 2023 "
        "Rules notified 14 November 2025. Data Protection Board operational. "
        "Maximum penalty Rs 250 crores per breach. Phase 2 enforcement "
        "November 2026. Phase 3 full enforcement May 2027. Organizations "
        "must consult qualified legal counsel for final compliance "
        "determinations in their jurisdiction."
    )
    elements.append(Paragraph(disclaimer_text, styles["DisclaimerText"]))
    elements.append(Spacer(1, 12))
    return elements


def risk_color_cell(risk_tier: str) -> str:
    """Return colored risk tier text for tables."""
    return risk_tier


# ================================================================
# REPORT 1 - AI SYSTEM CARD
# Required by EU AI Act Article 13
# ================================================================

async def generate_system_card(
    db: Session,
    agent_id: Optional[str],
    styles: dict
) -> list:
    """Generate AI System Card section."""
    elements = []
    elements.append(Paragraph("AI System Card", styles["SectionHeader"]))
    elements.append(Paragraph(
        "Required by EU AI Act Article 13 - Transparency and Information Provision",
        styles["SmallText"]
    ))
    elements.append(Spacer(1, 8))

    # Get agent info
    agent_query = db.query(AgentRegistry)
    if agent_id:
        agent_query = agent_query.filter(
            AgentRegistry.agent_id == agent_id
        )
    agents = agent_query.all()

    if not agents:
        elements.append(Paragraph(
            "No agents registered in system.",
            styles["BodyText"]
        ))
        return elements

    for agent in agents:
        agent_data = [
            ["Field", "Value"],
            ["Agent ID", agent.agent_id],
            ["Agent Name", agent.agent_name],
            ["Version", agent.agent_version],
            ["Industry", agent.industry],
            ["Use Case", agent.use_case or "Not specified"],
            ["Risk Category", agent.risk_category],
            ["Status", "Active" if agent.is_active else "Inactive"],
            ["Registered", agent.registered_at.strftime(
                "%Y-%m-%d") if agent.registered_at else "Unknown"],
            ["Total Decisions", str(agent.total_decisions)],
        ]

        try:
            frameworks = json.loads(agent.regulatory_frameworks or "[]")
            agent_data.append([
                "Regulatory Frameworks",
                ", ".join(frameworks) if frameworks else "Not specified"
            ])
        except Exception:
            pass

        table = Table(agent_data, colWidths=[2*inch, 4.5*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 1), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [COLORS["white"], COLORS["light_gray"]]),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

    return elements


# ================================================================
# REPORT 2 - DECISION AUDIT TRAIL
# Required by HIPAA Section 164.312
# ================================================================

async def generate_system_card(db, agent_id, styles):
    elements = []
    elements.append(Paragraph("AI System Card", styles["SectionHeader"]))
    elements.append(Paragraph(
        "Required by EU AI Act Article 13 - Transparency and Information Provision",
        styles["SmallText"]
    ))
    elements.append(Spacer(1, 8))

    agent_query = db.query(AgentRegistry)
    if agent_id:
        agent_query = agent_query.filter(AgentRegistry.agent_id == agent_id)
    agents = agent_query.all()

    if not agents:
        elements.append(Paragraph("No agents registered in system.", styles["BodyText"]))
        return elements

    for agent in agents:
        agent_data = [
            ["Field", "Value"],
            ["Agent ID", agent.agent_id],
            ["Agent Name", agent.agent_name],
            ["Version", agent.agent_version],
            ["Industry", agent.industry],
            ["Use Case", agent.use_case or "Not specified"],
            ["Risk Category", agent.risk_category],
            ["Status", "Active" if agent.is_active else "Inactive"],
            ["Registered", agent.registered_at.strftime("%Y-%m-%d") if agent.registered_at else "Unknown"],
            ["Total Decisions", str(agent.total_decisions)],
        ]

        try:
            frameworks = json.loads(agent.regulatory_frameworks or "[]")
            agent_data.append(["Regulatory Frameworks", ", ".join(frameworks) if frameworks else "Not specified"])
        except Exception:
            pass

        table = Table(agent_data, colWidths=[2*inch, 4.5*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 1), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLORS["white"], COLORS["light_gray"]]),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

    return elements


async def generate_audit_trail(db, agent_id, styles):
    elements = []
    elements.append(Paragraph("Decision Audit Trail", styles["SectionHeader"]))
    elements.append(Paragraph(
        "Required by HIPAA Section 164.312 - Audit Controls",
        styles["SmallText"]
    ))
    elements.append(Spacer(1, 8))

    query = db.query(DecisionLog)
    if agent_id:
        query = query.filter(DecisionLog.agent_id == agent_id)
    decisions = query.order_by(desc(DecisionLog.timestamp)).limit(100).all()

    if not decisions:
        elements.append(Paragraph("No decisions recorded in system.", styles["BodyText"]))
        return elements

    elements.append(Paragraph(f"Total decisions in audit trail: {len(decisions)}", styles["BodyText"]))
    elements.append(Spacer(1, 8))

    table_data = [["ID", "Timestamp", "Agent", "Risk", "Domain", "Review Status"]]
    for d in decisions:
        table_data.append([
            str(d.id),
            d.timestamp.strftime("%Y-%m-%d %H:%M") if d.timestamp else "N/A",
            d.agent_id[:20],
            d.risk_tier,
            d.regulatory_domain[:12],
            d.review_status
        ])

    col_widths = [0.5*inch, 1.3*inch, 1.3*inch, 0.8*inch, 1.1*inch, 1.5*inch]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLORS["white"], COLORS["light_gray"]]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]

    for i, d in enumerate(decisions, start=1):
        risk_color = RISK_COLORS.get(d.risk_tier, COLORS["text"])
        table_style.append(("TEXTCOLOR", (3, i), (3, i), risk_color))
        table_style.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))

    table.setStyle(TableStyle(table_style))
    elements.append(table)
    elements.append(Spacer(1, 12))
    return elements


async def generate_oversight_record(db, agent_id, styles):
    elements = []
    elements.append(Paragraph("Human Oversight Record", styles["SectionHeader"]))
    elements.append(Paragraph(
        "Required by EU AI Act Article 14 - Human Oversight",
        styles["SmallText"]
    ))
    elements.append(Spacer(1, 8))

    query = db.query(DecisionLog).filter(DecisionLog.requires_human_review == True)
    if agent_id:
        query = query.filter(DecisionLog.agent_id == agent_id)
    flagged = query.order_by(desc(DecisionLog.timestamp)).all()

    total = len(flagged)
    approved = sum(1 for d in flagged if d.review_status == "approved")
    rejected = sum(1 for d in flagged if d.review_status == "rejected")
    pending = sum(1 for d in flagged if d.review_status == "pending")

    summary_data = [
        ["Total Flagged", "Approved", "Rejected", "Pending"],
        [str(total), str(approved), str(rejected), str(pending)]
    ]

    summary_table = Table(summary_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
        ("BACKGROUND", (1, 1), (1, 1), COLORS["low"]),
        ("BACKGROUND", (2, 1), (2, 1), COLORS["critical"]),
        ("BACKGROUND", (3, 1), (3, 1), COLORS["medium"]),
        ("TEXTCOLOR", (1, 1), (3, 1), COLORS["white"]),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    if flagged:
        table_data = [["ID", "Risk", "Status", "Reviewer", "Reviewed At", "Justification"]]
        for d in flagged:
            justification = d.reviewer_justification or "Pending"
            if len(justification) > 40:
                justification = justification[:40] + "..."
            table_data.append([
                str(d.id),
                d.risk_tier,
                d.review_status,
                d.reviewer_id[:15] or "Pending",
                d.reviewed_at.strftime("%Y-%m-%d") if d.reviewed_at else "Pending",
                justification
            ])

        col_widths = [0.4*inch, 0.7*inch, 0.8*inch, 1.1*inch, 1*inch, 2.5*inch]
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLORS["white"], COLORS["light_gray"]]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)

    elements.append(Spacer(1, 12))
    return elements


async def generate_risk_summary(db, agent_id, styles):
    elements = []
    elements.append(Paragraph("Risk Assessment Summary", styles["SectionHeader"]))
    elements.append(Paragraph(
        "Required by EU AI Act Article 9 - Risk Management System",
        styles["SmallText"]
    ))
    elements.append(Spacer(1, 8))

    query = db.query(DecisionLog)
    if agent_id:
        query = query.filter(DecisionLog.agent_id == agent_id)
    total = query.count()

    risk_data = [["Risk Tier", "Count", "Percentage", "Regulatory Implication"]]
    implications = {
        "Critical": "Immediate human review mandatory. Potential EU AI Act Annex III violation.",
        "High": "Human review required. EU AI Act Article 14 oversight triggered.",
        "Medium": "Monitor closely. Document decisions for audit trail.",
        "Low": "Standard logging. No immediate action required."
    }

    for tier in ["Critical", "High", "Medium", "Low"]:
        count = query.filter(DecisionLog.risk_tier == tier).count()
        pct = f"{(count/total*100):.1f}%" if total > 0 else "0%"
        risk_data.append([tier, str(count), pct, implications[tier]])

    risk_data.append(["TOTAL", str(total), "100%", ""])

    table = Table(risk_data, colWidths=[0.9*inch, 0.6*inch, 0.9*inch, 4.1*inch])
    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), COLORS["light_gray"]),
        ("FONTNAME", (0, 1), (-1, -2), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]

    for i, tier in enumerate(["Critical", "High", "Medium", "Low"], start=1):
        table_style.append(("TEXTCOLOR", (0, i), (0, i), RISK_COLORS[tier]))
        table_style.append(("FONTNAME", (0, i), (0, i), "Helvetica-Bold"))

    table.setStyle(TableStyle(table_style))
    elements.append(table)
    elements.append(Spacer(1, 12))

    citations = [
        ["Regulation", "Requirement", "Status"],
        ["EU AI Act Article 9", "Risk management system", "Covered"],
        ["EU AI Act Article 13", "Transparency obligations", "Covered"],
        ["EU AI Act Article 14", "Human oversight", "Covered"],
        ["EU AI Act Article 16", "Provider obligations", "Covered"],
        ["HIPAA Section 164.312", "Audit controls", "Covered"],
        ["SOC 2 PI1", "Processing integrity", "Covered"],
        ["DPDP Act 2023 Section 8", "Data fiduciary obligations", "Covered"],
        ["DPDP Act 2023 Section 9", "Children data protection", "Covered"],
        ["DPDP Rules 2025 Rule 6", "Security safeguards", "Covered"],
        ["IT Act 2000 Section 43A", "Data protection compensation", "Covered"],
        ["RBI AI Guidelines 2024", "Banking AI oversight", "Covered"],
        ["SEBI AI Circular 2023", "Financial AI audit trail", "Covered"],
    ]

    elements.append(Paragraph("Regulatory Citations", styles["SubHeader"]))
    citations_table = Table(citations, colWidths=[2.2*inch, 2.8*inch, 1.5*inch])
    citations_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLORS["white"], COLORS["light_gray"]]),
        ("TEXTCOLOR", (2, 1), (2, -1), COLORS["low"]),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(citations_table)
    elements.append(Spacer(1, 12))
    return elements


async def generate_compliance_report(
    db: Session,
    agent_id: Optional[str] = None,
    report_type: str = "full"
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    agent_suffix = f"_{agent_id}" if agent_id else "_all_agents"
    filename = f"compliance_report{agent_suffix}_{timestamp}.pdf"
    filepath = f"reports/{filename}"

    styles = get_styles()
    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    elements = []
    elements.extend(build_header(
        styles,
        "Full Compliance Report" if report_type == "full" else report_type.replace("_", " ").title(),
        agent_id or "All Agents"
    ))

    if report_type in ["full", "system_card"]:
        elements.extend(await generate_system_card(db, agent_id, styles))
        if report_type == "full":
            elements.append(PageBreak())

    if report_type in ["full", "audit_trail"]:
        elements.extend(await generate_audit_trail(db, agent_id, styles))
        if report_type == "full":
            elements.append(PageBreak())

    if report_type in ["full", "oversight_record"]:
        elements.extend(await generate_oversight_record(db, agent_id, styles))
        if report_type == "full":
            elements.append(PageBreak())

    if report_type in ["full", "risk_summary"]:
        elements.extend(await generate_risk_summary(db, agent_id, styles))

    elements.extend(build_disclaimer(styles))
    doc.build(elements)
    return filepath


# ================================================================
# REPORT 3 - HUMAN OVERSIGHT RECORD
# Required by EU AI Act Article 14
# ================================================================

async def generate_oversight_record(
    db: Session,
    agent_id: Optional[str],
    styles: dict
) -> list:
    """Generate Human Oversight Record section."""
    elements = []
    elements.append(Paragraph(
        "Human Oversight Record",
        styles["SectionHeader"]
    ))
    elements.append(Paragraph(
        "Required by EU AI Act Article 14 - Human Oversight",
        styles["SmallText"]
    ))
    elements.append(Spacer(1, 8))

    query = db.query(DecisionLog).filter(
        DecisionLog.requires_human_review == True
    )
    if agent_id:
        query = query.filter(DecisionLog.agent_id == agent_id)
    flagged = query.order_by(desc(DecisionLog.timestamp)).all()

    total = len(flagged)
    approved = sum(1 for d in flagged if d.review_status == "approved")
    rejected = sum(1 for d in flagged if d.review_status == "rejected")
    pending = sum(1 for d in flagged if d.review_status == "pending")

    summary_data = [
        ["Total Flagged", "Approved", "Rejected", "Pending"],
        [str(total), str(approved), str(rejected), str(pending)]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch]
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
        ("BACKGROUND", (0, 1), (0, 1), COLORS["light_gray"]),
        ("BACKGROUND", (1, 1), (1, 1), COLORS["low"]),
        ("BACKGROUND", (2, 1), (2, 1), COLORS["critical"]),
        ("BACKGROUND", (3, 1), (3, 1), COLORS["medium"]),
        ("TEXTCOLOR", (1, 1), (3, 1), COLORS["white"]),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    if flagged:
        table_data = [["ID", "Risk", "Status", "Reviewer", "Reviewed At", "Justification"]]
        for d in flagged:
            table_data.append([
                str(d.id),
                d.risk_tier,
                d.review_status,
                d.reviewer_id[:15] or "Pending",
                d.reviewed_at.strftime("%Y-%m-%d") if d.reviewed_at else "Pending",
                (d.reviewer_justification[:40] + "...") if d.reviewer_justification and len(d.reviewer_justification) > 40 else (d.reviewer_justification or "Pending")
            ])

        col_widths = [0.4*inch, 0.7*inch, 0.8*inch, 1.1*inch, 1*inch, 2.5*inch]
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [COLORS["white"], COLORS["light_gray"]]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)

    elements.append(Spacer(1, 12))
    return elements


# ================================================================
# REPORT 4 - RISK ASSESSMENT SUMMARY
# Required by EU AI Act Article 9
# ================================================================

async def generate_risk_summary(
    db: Session,
    agent_id: Optional[str],
    styles: dict
) -> list:
    """Generate Risk Assessment Summary section."""
    elements = []
    elements.append(Paragraph(
        "Risk Assessment Summary",
        styles["SectionHeader"]
    ))
    elements.append(Paragraph(
        "Required by EU AI Act Article 9 - Risk Management System",
        styles["SmallText"]
    ))
    elements.append(Spacer(1, 8))

    query = db.query(DecisionLog)
    if agent_id:
        query = query.filter(DecisionLog.agent_id == agent_id)

    total = query.count()

    risk_data = [["Risk Tier", "Count", "Percentage", "Regulatory Implication"]]
    implications = {
        "Critical": "Immediate human review mandatory. Potential EU AI Act Annex III violation.",
        "High": "Human review required. EU AI Act Article 14 oversight triggered.",
        "Medium": "Monitor closely. Document decisions for audit trail.",
        "Low": "Standard logging. No immediate action required."
    }

    for tier in ["Critical", "High", "Medium", "Low"]:
        count = query.filter(DecisionLog.risk_tier == tier).count()
        pct = f"{(count/total*100):.1f}%" if total > 0 else "0%"
        risk_data.append([
            tier,
            str(count),
            pct,
            implications[tier]
        ])

    risk_data.append(["TOTAL", str(total), "100%", ""])

    table = Table(
        risk_data,
        colWidths=[0.9*inch, 0.6*inch, 0.9*inch, 4.1*inch]
    )

    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), COLORS["light_gray"]),
        ("FONTNAME", (0, 1), (-1, -2), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]

    for i, tier in enumerate(["Critical", "High", "Medium", "Low"], start=1):
        table_style.append(
            ("TEXTCOLOR", (0, i), (0, i), RISK_COLORS[tier])
        )
        table_style.append(
            ("FONTNAME", (0, i), (0, i), "Helvetica-Bold")
        )

    table.setStyle(TableStyle(table_style))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Regulatory citations
    elements.append(Paragraph(
        "Regulatory Citations",
        styles["SubHeader"]
    ))

    citations = [
        ["Regulation", "Requirement", "Status"],
        ["EU AI Act Article 9", "Risk management system", "Covered"],
        ["EU AI Act Article 13", "Transparency obligations", "Covered"],
        ["EU AI Act Article 14", "Human oversight", "Covered"],
        ["EU AI Act Article 16", "Provider obligations", "Covered"],
        ["HIPAA Section 164.312", "Audit controls", "Covered"],
        ["SOC 2 PI1", "Processing integrity", "Covered"],
        ["DPDP Act 2023 Section 8", "Data fiduciary obligations", "Covered"],
        ["DPDP Act 2023 Section 9", "Children data protection", "Covered"],
        ["DPDP Rules 2025 Rule 6", "Security safeguards", "Covered"],
        ["IT Act 2000 Section 43A", "Data protection compensation", "Covered"],
        ["RBI AI Guidelines 2024", "Banking AI oversight", "Covered"],
        ["SEBI AI Circular 2023", "Financial AI audit trail", "Covered"],
    ]

    citations_table = Table(
        citations,
        colWidths=[2.2*inch, 2.8*inch, 1.5*inch]
    )
    citations_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [COLORS["white"], COLORS["light_gray"]]),
        ("TEXTCOLOR", (2, 1), (2, -1), COLORS["low"]),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))

    elements.append(citations_table)
    elements.append(Spacer(1, 12))
    return elements


# ================================================================
# MAIN REPORT GENERATOR
# ================================================================

async def generate_compliance_report(
    db: Session,
    agent_id: Optional[str] = None,
    report_type: str = "full"
) -> str:
    """
    Generate a complete PDF compliance report.

    Args:
        db: Database session
        agent_id: Filter by specific agent or None for all
        report_type: full, system_card, audit_trail,
                     oversight_record, risk_summary

    Returns:
        Path to generated PDF file
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    agent_suffix = f"_{agent_id}" if agent_id else "_all_agents"
    filename = f"compliance_report{agent_suffix}_{timestamp}.pdf"
    filepath = f"reports/{filename}"

    styles = get_styles()

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    elements = []

    # Header
    elements.extend(build_header(
        styles,
        "Full Compliance Report" if report_type == "full" else report_type.replace("_", " ").title(),
        agent_id or "All Agents"
    ))

    # Generate requested sections
    if report_type in ["full", "system_card"]:
        elements.extend(await generate_system_card(db, agent_id, styles))
        if report_type == "full":
            elements.append(PageBreak())

    if report_type in ["full", "audit_trail"]:
        elements.extend(await generate_audit_trail(db, agent_id, styles))
        if report_type == "full":
            elements.append(PageBreak())

    if report_type in ["full", "oversight_record"]:
        elements.extend(await generate_oversight_record(db, agent_id, styles))
        if report_type == "full":
            elements.append(PageBreak())

    if report_type in ["full", "risk_summary"]:
        elements.extend(await generate_risk_summary(db, agent_id, styles))

    # Legal disclaimer at end
    elements.extend(build_disclaimer(styles))

    # Build PDF
    doc.build(elements)

    return filepath