"""
CompliancePilot - Classification Engine
Version 3.0.0 - Gemini Powered
"""

import json
import time
import re
import logging
import os
from typing import Dict, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv
from backend.database.models import DecisionLog, SessionLocal

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CompliancePilot.Classification")


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


DOMAIN_REGISTRY = {
    "healthcare": {
        "name": "Healthcare",
        "primary_regulations": ["HIPAA", "EU_AI_ACT", "DPDP_ACT_2023"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000", "IMC_GUIDELINES"],
        "default_risk": "High",
        "auto_review_threshold": "Medium",
        "key_risks": ["Patient data exposure", "Clinical decision liability", "Treatment recommendation errors"]
    },
    "finance": {
        "name": "Finance and Banking",
        "primary_regulations": ["SOC2", "EU_AI_ACT", "DPDP_ACT_2023"],
        "indian_regulations": ["RBI_AI_GUIDELINES_2024", "SEBI_AI_CIRCULAR", "DPDP_ACT_2023"],
        "default_risk": "High",
        "auto_review_threshold": "Medium",
        "key_risks": ["Biased credit decisions", "Financial data exposure", "KYC compliance failures"]
    },
    "hr": {
        "name": "Human Resources",
        "primary_regulations": ["EU_AI_ACT", "DPDP_ACT_2023"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "High",
        "auto_review_threshold": "Low",
        "key_risks": ["Discriminatory hiring decisions", "Employee data privacy", "Bias in performance scoring"]
    },
    "legal": {
        "name": "Legal Services",
        "primary_regulations": ["EU_AI_ACT", "DPDP_ACT_2023", "SOC2"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "High",
        "auto_review_threshold": "Low",
        "key_risks": ["Privileged information exposure", "Incorrect legal advice liability", "Client confidentiality breach"]
    },
    "education": {
        "name": "Education",
        "primary_regulations": ["EU_AI_ACT", "DPDP_ACT_2023"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "Medium",
        "auto_review_threshold": "Medium",
        "key_risks": ["Biased student assessment", "Minor data protection under DPDP Section 9", "Discriminatory admissions"]
    },
    "government": {
        "name": "Government and Public Services",
        "primary_regulations": ["EU_AI_ACT", "DPDP_ACT_2023", "IT_ACT_2000"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "High",
        "auto_review_threshold": "Low",
        "key_risks": ["Discriminatory service denial", "Citizen data privacy", "Fundamental rights violations"]
    },
    "retail": {
        "name": "Retail and E-Commerce",
        "primary_regulations": ["DPDP_ACT_2023", "SOC2", "GDPR"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "Medium",
        "auto_review_threshold": "High",
        "key_risks": ["Customer data profiling", "Children targeting under DPDP Section 9", "Consent violations"]
    },
    "general": {
        "name": "General",
        "primary_regulations": ["DPDP_ACT_2023", "EU_AI_ACT"],
        "indian_regulations": ["DPDP_ACT_2023", "IT_ACT_2000"],
        "default_risk": "Low",
        "auto_review_threshold": "High",
        "key_risks": ["General data privacy risks", "Automated decision transparency"]
    }
}


class PIISanitizer:
    def __init__(self):
        self.patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE_INTL": r'\+?1?\s*\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',
            "CREDIT_CARD": r'\b(?:\d{4}[\s-]?){3}\d{4}\b',
            "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "AADHAAR": r'\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b',
            "PAN_CARD": r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            "INDIAN_PHONE": r'\b[6-9]\d{9}\b',
            "GSTIN": r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b',
            "PATIENT_ID": r'\bPATIENT[_\s]?ID[:\s]+[A-Z0-9]+\b',
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
        return f"PII sanitized. Types masked: {', '.join(types_found)}. Total items: {len(sanitization_map)}."


LEGAL_DISCLAIMER = """
LEGAL NOTICE: This is an AI-assisted recommendation only.
Not legal, medical, or financial advice.
Human review mandatory for High and Critical risk decisions.
DPDP Act 2023 Rules notified 14 November 2025.
Maximum penalty Rs 250 crores per breach.
"""

CLASSIFICATION_PROMPT = """
You are a senior AI regulatory compliance expert.

AGENT: {agent_id}
INDUSTRY: {industry}
REGULATIONS: {applicable_regulations}
KEY RISKS: {key_risks}

SANITIZED INPUT: {sanitized_input}
SANITIZED OUTPUT: {sanitized_output}
PII NOTE: {sanitization_summary}

RISK RULES:
- Critical (80-100): Medical diagnosis, treatment, loan denial, HR termination, children data
- High (60-79): HR screening, financial advisory, insurance, credit scoring
- Medium (40-59): Customer service with personal data, scheduling
- Low (0-39): General information, no personal data

RESPOND WITH VALID JSON ONLY. NO TEXT OUTSIDE JSON.

{{
    "regulatory_domain": "primary regulation code",
    "secondary_regulations": ["list"],
    "risk_tier": "Low or Medium or High or Critical",
    "risk_score": 0,
    "article_triggered": "primary article",
    "secondary_articles": ["list"],
    "recommended_action": "specific action",
    "requires_human_review": true,
    "classification_reasoning": "2-3 sentences",
    "compliance_notes": "specific requirements",
    "indian_law_notes": "Indian law considerations",
    "dpdp_applicability": "DPDP Act 2023 notes",
    "data_handling_recommendation": "data handling guidance"
}}
"""


async def classify_decision(
    decision_id: int,
    input_prompt: str,
    agent_output: str,
    agent_id: str,
    decision_context: str = "{}"
) -> dict:
    start_time = time.time()
    logger.info(f"Starting classification | Decision: {decision_id} | Agent: {agent_id}")

    try:
        context = json.loads(decision_context)
        industry = context.get("industry", "general").lower()
    except Exception:
        industry = "general"

    domain_config = DOMAIN_REGISTRY.get(industry, DOMAIN_REGISTRY["general"])

    sanitizer = PIISanitizer()
    sanitized_input, input_map = sanitizer.sanitize(input_prompt)
    sanitized_output, output_map = sanitizer.sanitize(agent_output)
    combined_map = {**input_map, **output_map}
    sanitization_summary = sanitizer.get_summary(combined_map)

    logger.info(f"PII sanitized | Items masked: {len(combined_map)} | Decision: {decision_id}")

    prompt = CLASSIFICATION_PROMPT.format(
        agent_id=agent_id,
        industry=domain_config["name"],
        applicable_regulations=", ".join(
            domain_config["primary_regulations"] + domain_config["indian_regulations"]
        ),
        key_risks=", ".join(domain_config["key_risks"]),
        sanitized_input=sanitized_input[:2000],
        sanitized_output=sanitized_output[:2000],
        sanitization_summary=sanitization_summary
    )

    classification = await _call_gemini(prompt)

    classification["legal_disclaimer"] = LEGAL_DISCLAIMER
    classification["sanitization_summary"] = sanitization_summary
    classification["pii_items_protected"] = len(combined_map)
    classification["classification_model"] = "gemini-2.5-flash"

    classification_time_ms = int((time.time() - start_time) * 1000)

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


async def _call_gemini(prompt: str) -> dict:
    try:
        client = get_gemini_client()
        full_prompt = (
            "You are a regulatory compliance expert covering "
            "EU AI Act, HIPAA, SOC 2, GDPR, Indian DPDP Act 2023, "
            "IT Act 2000, RBI Guidelines, SEBI Guidelines. "
            "Always respond with valid JSON only. "
            "No markdown. No text outside JSON.\n\n" + prompt
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        raw_response = response.text.strip()

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
        logger.error(f"Gemini classification failed: {str(e)}")
        return {
            "regulatory_domain": "General",
            "secondary_regulations": [],
            "risk_tier": "Medium",
            "risk_score": 50,
            "article_triggered": "Classification pending - manual review required",
            "secondary_articles": [],
            "recommended_action": "Manual classification required due to system error",
            "requires_human_review": True,
            "classification_reasoning": "Automatic classification failed. Defaulting to Medium risk with human review required.",
            "compliance_notes": "Manual review required",
            "indian_law_notes": "Manual review required",
            "dpdp_applicability": "Manual review required",
            "data_handling_recommendation": "Treat as sensitive pending manual review"
        }


async def _update_decision(
    decision_id: int,
    classification: dict,
    classification_time_ms: int,
    sanitization_summary: str
) -> None:
    db = SessionLocal()
    try:
        decision = db.query(DecisionLog).filter(DecisionLog.id == decision_id).first()
        if not decision:
            logger.error(f"Decision {decision_id} not found")
            return

        decision.regulatory_domain = classification.get("regulatory_domain", "General")
        decision.risk_tier = classification.get("risk_tier", "Medium")
        decision.risk_score = classification.get("risk_score", 50)
        decision.article_triggered = classification.get("article_triggered", "None")
        decision.recommended_action = classification.get("recommended_action", "None")
        decision.requires_human_review = classification.get("requires_human_review", False)
        decision.classification_reasoning = (
            classification.get("classification_reasoning", "") + " | " + sanitization_summary
        )
        decision.classification_model = "gemini-2.5-flash"
        decision.classification_time_ms = classification_time_ms

        if decision.requires_human_review:
            decision.review_status = "pending"
        else:
            decision.review_status = "not_required"

        db.commit()
        logger.info(f"Decision {decision_id} updated | Risk: {decision.risk_tier} | Review required: {decision.requires_human_review}")

    except Exception as e:
        logger.error(f"Failed to update decision {decision_id}: {str(e)}")
        db.rollback()
    finally:
        db.close()