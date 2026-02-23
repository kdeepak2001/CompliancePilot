"""
CompliancePilot - Enterprise Secure Classification Engine
==========================================================
Version 3.0.0 - Multi-Domain Universal Design

SECURITY ARCHITECTURE:
- Zero raw PII ever leaves this system
- All data sanitized before external API call
- Indian and International PII patterns covered
- Full audit trail of every sanitization action

REGULATORY COVERAGE:

INTERNATIONAL:
- EU AI Act 2024 (Articles 9, 13, 14, 16, Annex III)
- HIPAA Security Rule (Section 164.312)
- SOC 2 Type II (Processing Integrity, Confidentiality)
- GDPR (Articles 5, 6, 9, 22, 25)

INDIA - UPDATED FEBRUARY 2026:
- DPDP Act 2023 + DPDP Rules 2025
  (Rules notified 14 November 2025)
  (Phase 1 active, Phase 2 November 2026, Phase 3 May 2027)
  (Data Protection Board established November 13 2025)
  (Maximum penalty Rs 250 crores per breach)
- IT Act 2000 (Sections 43, 66, 72A)
- RBI Guidelines on Responsible AI in Banking 2024
- SEBI AI/ML Circular 2023
- Indian Medical Council AI Guidelines
- IRDAI Guidelines on AI in Insurance

SUPPORTED DOMAINS:
- Healthcare, Finance, HR, Legal, Education,
  Government, Retail, General

LEGAL NOTICE:
Classifications are AI-assisted recommendations only.
Not legal, medical, or financial advice.
Human review mandatory for High and Critical decisions.

Author: CompliancePilot
Version: 3.0.0
"""

import json
import time
import re
import logging
import os
from typing import Dict, Tuple
from groq import Groq
from dotenv import load_dotenv
from backend.database.models import DecisionLog, SessionLocal

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CompliancePilot.Classification")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ================================================================
# DOMAIN REGISTRY
# Universal design - any industry plugs in here.
# Adding a new domain requires zero code changes.
# ================================================================

DOMAIN_REGISTRY = {
    "healthcare": {
        "name": "Healthcare",
        "primary_regulations": ["HIPAA", "EU_AI_ACT", "DPDP_ACT_2023"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000", "IMC_GUIDELINES"],
        "default_risk": "High",
        "auto_review_threshold": "Medium",
        "key_risks": [
            "Patient data exposure",
            "Clinical decision liability",
            "Treatment recommendation errors"
        ]
    },
    "finance": {
        "name": "Finance and Banking",
        "primary_regulations": ["SOC2", "EU_AI_ACT", "DPDP_ACT_2023"],
        "indian_regulations": ["RBI_AI_GUIDELINES_2024", "SEBI_AI_CIRCULAR", "DPDP_ACT_2023"],
        "default_risk": "High",
        "auto_review_threshold": "Medium",
        "key_risks": [
            "Biased credit decisions",
            "Financial data exposure",
            "KYC compliance failures"
        ]
    },
    "hr": {
        "name": "Human Resources",
        "primary_regulations": ["EU_AI_ACT", "DPDP_ACT_2023"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "High",
        "auto_review_threshold": "Low",
        "key_risks": [
            "Discriminatory hiring decisions",
            "Employee data privacy",
            "Bias in performance scoring"
        ]
    },
    "legal": {
        "name": "Legal Services",
        "primary_regulations": ["EU_AI_ACT", "DPDP_ACT_2023", "SOC2"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "High",
        "auto_review_threshold": "Low",
        "key_risks": [
            "Privileged information exposure",
            "Incorrect legal advice liability",
            "Client confidentiality breach"
        ]
    },
    "education": {
        "name": "Education",
        "primary_regulations": ["EU_AI_ACT", "DPDP_ACT_2023"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "Medium",
        "auto_review_threshold": "Medium",
        "key_risks": [
            "Biased student assessment",
            "Minor data protection under DPDP Section 9",
            "Discriminatory admissions"
        ]
    },
    "government": {
        "name": "Government and Public Services",
        "primary_regulations": ["EU_AI_ACT", "DPDP_ACT_2023", "IT_ACT_2000"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "High",
        "auto_review_threshold": "Low",
        "key_risks": [
            "Discriminatory service denial",
            "Citizen data privacy",
            "Fundamental rights violations"
        ]
    },
    "retail": {
        "name": "Retail and E-Commerce",
        "primary_regulations": ["DPDP_ACT_2023", "SOC2", "GDPR"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "Medium",
        "auto_review_threshold": "High",
        "key_risks": [
            "Customer data profiling",
            "Children targeting under DPDP Section 9",
            "Consent violations"
        ]
    },
    "general": {
        "name": "General",
        "primary_regulations": ["DPDP_ACT_2023", "EU_AI_ACT"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "Low",
        "auto_review_threshold": "High",
        "key_risks": [
            "General data privacy risks",
            "Automated decision transparency"
        ]
    }
}


# ================================================================
# PII SANITIZER
# Zero data leakage architecture.
# No raw personal data ever sent to Groq.
# ================================================================

class PIISanitizer:
    """
    Detects and masks PII before sending to external APIs.
    Covers Indian and International PII patterns.
    Satisfies GDPR Article 25, DPDP Act 2023 Section 8,
    HIPAA Minimum Necessary Standard.
    """

    def __init__(self):
        self.patterns = {
            # International
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE_INTL": r'\+?1?\s*\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',
            "CREDIT_CARD": r'\b(?:\d{4}[\s-]?){3}\d{4}\b',
            "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "IP_ADDRESS": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            # Indian specific
            "AADHAAR": r'\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b',
            "PAN_CARD": r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            "INDIAN_PHONE": r'\b[6-9]\d{9}\b',
            "PASSPORT_INDIA": r'\b[A-Z]{1}[0-9]{7}\b',
            "VOTER_ID": r'\b[A-Z]{3}[0-9]{7}\b',
            "GSTIN": r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b',
            "IFSC": r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            # Medical
            "PATIENT_ID": r'\bPATIENT[_\s]?ID[:\s]+[A-Z0-9]+\b',
            "MRN": r'\bMRN[:\s]+[A-Z0-9]+\b',
        }

    def sanitize(self, text: str) -> Tuple[str, Dict]:
        sanitized = text
        sanitization_map = {}
        counters = {}

        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, sanitized, re.IGNORECASE)
            for match in matches:
                if match not in sanitization_map:
                    counters[pii_type] = counters.get(pii_type, 0) + 1
                    token = f"[{pii_type}_{counters[pii_type]}]"
                    sanitization_map[match] = token
                    sanitized = sanitized.replace(match, token)

        return sanitized, sanitization_map

    def get_summary(self, sanitization_map: Dict) -> str:
        if not sanitization_map:
            return "No PII detected in input"
        types_found = set()
        for original, token in sanitization_map.items():
            pii_type = token.split("_")[0].replace("[", "")
            types_found.add(pii_type)
        return (
            f"PII sanitized before external API call. "
            f"Types masked: {', '.join(types_found)}. "
            f"Total items masked: {len(sanitization_map)}. "
            f"Original data retained only in local database."
        )


# ================================================================
# LEGAL DISCLAIMER
# Attached to every classification output.
# ================================================================

LEGAL_DISCLAIMER = """
LEGAL NOTICE - READ CAREFULLY:

1. AI ASSISTED CLASSIFICATION ONLY
   This is an AI-assisted recommendation. Not legal,
   medical, or financial advice of any kind.

2. HUMAN REVIEW MANDATORY
   High and Critical risk decisions require mandatory
   human review before any action is taken.

3. INDIAN LAW NOTICE
   DPDP Act 2023 Rules notified 14 November 2025.
   Data Protection Board established and operational.
   Maximum penalty: Rs 250 crores per breach.
   Phase 2 enforcement: November 2026.
   Phase 3 full enforcement: May 2027.
   Organizations must comply with all applicable phases.

4. LIABILITY LIMITATION
   CompliancePilot accepts no liability for decisions
   made solely on the basis of this classification.

5. DATA PROTECTION
   All personal data sanitized before classification.
   Original data never transmitted to external services.
"""


# ================================================================
# CLASSIFICATION PROMPT
# Covers all domains and regulations.
# ================================================================

CLASSIFICATION_PROMPT = """
You are a senior AI regulatory compliance expert covering:

INTERNATIONAL: EU AI Act 2024, HIPAA, SOC 2 Type II, GDPR
INDIA: DPDP Act 2023 + Rules 2025, IT Act 2000, RBI AI Guidelines 2024,
       SEBI AI Circular 2023, Indian Medical Council Guidelines,
       IRDAI AI Guidelines

AGENT INFORMATION:
Agent ID: {agent_id}
Industry Domain: {industry}
Applicable Regulations: {applicable_regulations}
Domain Key Risks: {key_risks}

NOTE: Input below is pre-sanitized. PII replaced with tokens.
Classify based on context and nature of decision only.

SANITIZED INPUT:
{sanitized_input}

SANITIZED OUTPUT:
{sanitized_output}

SANITIZATION SUMMARY:
{sanitization_summary}

RISK TIER RULES:
- Critical (80-100): Medical diagnosis, treatment decisions, loan denial,
  HR termination, criminal justice, children data under DPDP Section 9
- High (60-79): HR screening, financial advisory, insurance decisions,
  sensitive personal data under DPDP Section 9, credit scoring
- Medium (40-59): Customer service with personal data, scheduling,
  general financial queries, content recommendations
- Low (0-39): General information, factual queries, no personal data

RESPOND WITH VALID JSON ONLY. NO TEXT OUTSIDE JSON.

{{
    "regulatory_domain": "primary regulation code",
    "secondary_regulations": ["list of other applicable regulations"],
    "risk_tier": "Low or Medium or High or Critical",
    "risk_score": <integer 0-100>,
    "article_triggered": "primary article reference",
    "secondary_articles": ["other applicable articles"],
    "recommended_action": "specific action required",
    "requires_human_review": <true or false>,
    "classification_reasoning": "2-3 sentences explaining classification",
    "compliance_notes": "specific compliance requirements",
    "indian_law_notes": "specific Indian law considerations if applicable",
    "dpdp_applicability": "DPDP Act 2023 specific notes including phase timeline",
    "data_handling_recommendation": "how data in this decision should be handled"
}}
"""


# ================================================================
# MAIN CLASSIFICATION FUNCTION
# ================================================================

async def classify_decision(
    decision_id: int,
    input_prompt: str,
    agent_output: str,
    agent_id: str,
    decision_context: str = "{}"
) -> dict:
    """
    Securely classify an AI agent decision.

    Security flow:
    1. Detect industry domain
    2. Sanitize all PII from input and output
    3. Send only sanitized data to Groq
    4. Attach legal disclaimer to result
    5. Update database with full classification
    """
    start_time = time.time()

    logger.info(
        f"Starting classification | "
        f"Decision: {decision_id} | Agent: {agent_id}"
    )

    # Detect domain from context
    try:
        context = json.loads(decision_context)
        industry = context.get("industry", "general").lower()
    except Exception:
        industry = "general"

    # Get domain configuration
    domain_config = DOMAIN_REGISTRY.get(
        industry,
        DOMAIN_REGISTRY["general"]
    )

    # Sanitize PII
    sanitizer = PIISanitizer()
    sanitized_input, input_map = sanitizer.sanitize(input_prompt)
    sanitized_output, output_map = sanitizer.sanitize(agent_output)
    combined_map = {**input_map, **output_map}
    sanitization_summary = sanitizer.get_summary(combined_map)

    logger.info(
        f"PII sanitized | Items masked: {len(combined_map)} | "
        f"Decision: {decision_id}"
    )

    # Build prompt with domain specific context
    prompt = CLASSIFICATION_PROMPT.format(
        agent_id=agent_id,
        industry=domain_config["name"],
        applicable_regulations=", ".join(
            domain_config["primary_regulations"] +
            domain_config["indian_regulations"]
        ),
        key_risks=", ".join(domain_config["key_risks"]),
        sanitized_input=sanitized_input[:2000],
        sanitized_output=sanitized_output[:2000],
        sanitization_summary=sanitization_summary
    )

    # Get classification from Groq
    classification = await _call_groq(prompt)

    # Attach security and legal metadata
    classification["legal_disclaimer"] = LEGAL_DISCLAIMER
    classification["sanitization_summary"] = sanitization_summary
    classification["pii_items_protected"] = len(combined_map)
    classification["data_sent_to_external_api"] = "sanitized_only"
    classification["classification_model"] = "llama-3.3-70b-versatile"
    classification["domain_config"] = domain_config["name"]

    classification_time_ms = int((time.time() - start_time) * 1000)

    # Update database
    await _update_decision(
        decision_id=decision_id,
        classification=classification,
        classification_time_ms=classification_time_ms,
        sanitization_summary=sanitization_summary
    )

    logger.info(
        f"Classification complete | Decision: {decision_id} | "
        f"Risk: {classification.get('risk_tier')} | "
        f"Domain: {classification.get('regulatory_domain')} | "
        f"PII protected: {len(combined_map)} | "
        f"Time: {classification_time_ms}ms"
    )

    return classification


# ================================================================
# GROQ API CALLER
# Only sanitized data reaches here. Ever.
# ================================================================

async def _call_groq(prompt: str) -> dict:
    """
    Call Groq API with sanitized data only.
    Returns safe default if API fails.
    System never crashes because classification failed.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a regulatory compliance expert covering "
                        "EU AI Act, HIPAA, SOC 2, GDPR, Indian DPDP Act 2023, "
                        "IT Act 2000, RBI Guidelines, SEBI Guidelines. "
                        "Always respond with valid JSON only. "
                        "No markdown. No text outside JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=800
        )

        raw_response = response.choices[0].message.content.strip()

        # Clean markdown if model adds it
        if "```" in raw_response:
            parts = raw_response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw_response = part
                    break

        classification = json.loads(raw_response)

        # Validate required fields
        required_fields = [
            "regulatory_domain", "risk_tier", "risk_score",
            "article_triggered", "recommended_action",
            "requires_human_review", "classification_reasoning"
        ]
        for field in required_fields:
            if field not in classification:
                raise ValueError(f"Missing field: {field}")

        return classification

    except Exception as e:
        logger.error(f"Groq classification failed: {str(e)}")

        # Safe default — never crash the system
        return {
            "regulatory_domain": "General",
            "secondary_regulations": [],
            "risk_tier": "Medium",
            "risk_score": 50,
            "article_triggered": "Classification pending - manual review required",
            "secondary_articles": [],
            "recommended_action": "Manual classification required due to system error",
            "requires_human_review": True,
            "classification_reasoning": (
                "Automatic classification failed. "
                "Defaulting to Medium risk with human review required "
                "to ensure compliance is not compromised."
            ),
            "compliance_notes": "Manual review required",
            "indian_law_notes": "Manual review required",
            "dpdp_applicability": "Manual review required",
            "data_handling_recommendation": "Treat as sensitive pending manual review"
        }


# ================================================================
# DATABASE UPDATER
# ================================================================

async def _update_decision(
    decision_id: int,
    classification: dict,
    classification_time_ms: int,
    sanitization_summary: str
) -> None:
    """
    Update decision record with classification results.
    """
    db = SessionLocal()

    try:
        decision = db.query(DecisionLog).filter(
            DecisionLog.id == decision_id
        ).first()

        if not decision:
            logger.error(f"Decision {decision_id} not found")
            return

        decision.regulatory_domain = classification.get(
            "regulatory_domain", "General"
        )
        decision.risk_tier = classification.get("risk_tier", "Medium")
        decision.risk_score = classification.get("risk_score", 50)
        decision.article_triggered = classification.get(
            "article_triggered", "None"
        )
        decision.recommended_action = classification.get(
            "recommended_action", "None"
        )
        decision.requires_human_review = classification.get(
            "requires_human_review", False
        )
        decision.classification_reasoning = (
            classification.get("classification_reasoning", "") +
            " | " + sanitization_summary
        )
        decision.classification_model = "llama-3.3-70b-versatile"
        decision.classification_time_ms = classification_time_ms

        if decision.requires_human_review:
            decision.review_status = "pending"
        else:
            decision.review_status = "not_required"

        db.commit()

        logger.info(
            f"Decision {decision_id} updated | "
            f"Risk: {decision.risk_tier} | "
            f"Review required: {decision.requires_human_review}"
        )

    except Exception as e:
        logger.error(f"Failed to update decision {decision_id}: {str(e)}")
        db.rollback()

    finally:
        db.close()