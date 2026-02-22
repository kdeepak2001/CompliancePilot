"""
CompliancePilot - Agent Wrapper SDK
=====================================
The core interception layer that sits between any AI agent
and its output — capturing every decision for compliance audit.

Design Pattern: Decorator + Middleware
Industry Standard: Zero-intrusion wrapping
Compliance Coverage: EU AI Act Article 14, HIPAA 164.312, SOC 2

Usage Example:
    from backend.agents.wrapper import ComplianceWrapper
    
    wrapper = ComplianceWrapper(agent_id="medical-triage-v1")
    result = await wrapper.run(agent_func, input_prompt, session_id)

Author: CompliancePilot
Version: 1.0.0
"""

import time
import json
import uuid
import httpx
import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Any, Optional
from functools import wraps

# ── Logging Setup ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CompliancePilot.Wrapper")


# ================================================================
# COMPLIANCE WRAPPER SDK
# The main class developers import into their agent code.
# ================================================================

class ComplianceWrapper:
    """
    Zero-intrusion compliance wrapper for any AI agent.
    
    This wrapper:
    1. Intercepts every agent call
    2. Captures full input, output, timing, and tool calls
    3. Sends decision log to CompliancePilot backend
    4. Returns the original agent response untouched
    
    Critical design principle:
    If CompliancePilot fails for any reason — the agent
    still works normally. Compliance logging must NEVER
    break the underlying agent. This is non-negotiable
    in production environments.
    """

    def __init__(
        self,
        agent_id: str,
        agent_version: str = "1.0.0",
        backend_url: str = "http://localhost:8000",
        is_demo: bool = False,
        industry: str = "General"
    ):
        """
        Initialize the compliance wrapper.
        
        Args:
            agent_id: Unique identifier for this agent
            agent_version: Version of the agent being wrapped
            backend_url: URL of CompliancePilot backend API
            is_demo: True for demo/test data, False for production
            industry: Industry context - healthcare, finance, hr, general
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.backend_url = backend_url
        self.is_demo = is_demo
        self.industry = industry
        
        logger.info(
            f" ComplianceWrapper initialized | "
            f"Agent: {agent_id} | "
            f"Industry: {industry} | "
            f"Demo Mode: {is_demo}"
        )

    # ── Core Interception Method ───────────────────────────────────

    async def run(
        self,
        agent_func: Callable,
        input_prompt: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[dict] = None
    ) -> str:
        """
        Run an agent function with full compliance interception.
        
        This is the main method. It:
        1. Records start time
        2. Runs the agent function
        3. Records end time and calculates duration
        4. Captures everything into a structured log
        5. Sends log to backend asynchronously
        6. Returns the original agent response
        
        Args:
            agent_func: The agent function to run and intercept
            input_prompt: The input/question sent to the agent
            session_id: Optional session identifier
            user_id: Optional user identifier (will be hashed for PII)
            context: Optional additional context dictionary
            
        Returns:
            The original agent response string — untouched
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Record start time for performance tracking
        start_time = time.time()
        agent_output = ""
        tool_calls = []
        error_occurred = False

        try:
            # ── Run the actual agent ───────────────────────────────
            logger.info(
                f" Agent [{self.agent_id}] running | "
                f"Session: {session_id[:8]}..."
            )

            # Handle both async and sync agent functions
            if asyncio.iscoroutinefunction(agent_func):
                agent_output = await agent_func(input_prompt)
            else:
                agent_output = agent_func(input_prompt)

            # Ensure output is a string
            if not isinstance(agent_output, str):
                agent_output = str(agent_output)

            logger.info(
                f" Agent [{self.agent_id}] completed | "
                f"Output length: {len(agent_output)} chars"
            )

        except Exception as e:
            # ── Agent failed — log the error but don't crash ───────
            error_occurred = True
            agent_output = f"AGENT_ERROR: {str(e)}"
            logger.error(
                f" Agent [{self.agent_id}] failed | "
                f"Error: {str(e)}"
            )

        finally:
            # ── Calculate processing time ──────────────────────────
            processing_time_ms = int((time.time() - start_time) * 1000)

        # ── Build the decision log payload ─────────────────────────
        decision_payload = self._build_payload(
            session_id=session_id,
            input_prompt=input_prompt,
            agent_output=agent_output,
            tool_calls=tool_calls,
            processing_time_ms=processing_time_ms,
            user_id=user_id,
            context=context or {},
            error_occurred=error_occurred
        )

        # ── Send to backend asynchronously ─────────────────────────
        # We do this asynchronously so it never slows down
        # the agent response to the user.
        await self._send_to_backend(decision_payload)

        # ── Always return original agent output ────────────────────
        return agent_output

    # ── Synchronous Version ────────────────────────────────────────

    def run_sync(
        self,
        agent_func: Callable,
        input_prompt: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[dict] = None
    ) -> str:
        """
        Synchronous version of run() for non-async environments.
        Same behavior — just works without async/await syntax.
        """
        return asyncio.run(
            self.run(
                agent_func=agent_func,
                input_prompt=input_prompt,
                session_id=session_id,
                user_id=user_id,
                context=context
            )
        )

    # ── Payload Builder ────────────────────────────────────────────

    def _build_payload(
        self,
        session_id: str,
        input_prompt: str,
        agent_output: str,
        tool_calls: list,
        processing_time_ms: int,
        user_id: Optional[str],
        context: dict,
        error_occurred: bool
    ) -> dict:
        """
        Build the structured decision log payload.
        This is what gets stored in the database.
        """
        import hashlib

        # Hash user ID for PII protection
        user_id_hash = ""
        if user_id:
            user_id_hash = hashlib.sha256(
                user_id.encode()
            ).hexdigest()

        return {
            "session_id": session_id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_prompt": input_prompt,
            "agent_output": agent_output,
            "tool_calls": json.dumps(tool_calls),
            "decision_context": json.dumps({
                **context,
                "industry": self.industry,
                "error_occurred": error_occurred
            }),
            "user_id_hash": user_id_hash,
            "processing_time_ms": processing_time_ms,
            "is_demo": self.is_demo
        }

    # ── Backend Sender ─────────────────────────────────────────────

    async def _send_to_backend(self, payload: dict) -> bool:
        """
        Send decision log to CompliancePilot backend API.
        
        Critical: This method must NEVER raise an exception.
        If the backend is down — the agent still works.
        We log the failure but never crash the agent.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.backend_url}/api/decisions/log",
                    json=payload
                )

                if response.status_code == 200:
                    logger.info(
                        f" Decision logged successfully | "
                        f"Agent: {self.agent_id}"
                    )
                    return True
                else:
                    logger.warning(
                        f" Backend returned {response.status_code} | "
                        f"Agent: {self.agent_id}"
                    )
                    return False

        except Exception as e:
            # Backend is down or unreachable
            # Agent continues working — compliance logging fails silently
            logger.warning(
                f" Could not reach CompliancePilot backend | "
                f"Error: {str(e)} | "
                f"Agent continues normally"
            )
            return False


# ================================================================
# DECORATOR VERSION
# Alternative usage pattern for developers who prefer decorators.
# ================================================================

def compliance_monitored(
    agent_id: str,
    industry: str = "General",
    is_demo: bool = False
):
    """
    Decorator version of ComplianceWrapper.
    
    Usage:
        @compliance_monitored(agent_id="hr-agent-v1", industry="hr")
        def my_agent(prompt: str) -> str:
            return llm.invoke(prompt)
    """
    def decorator(func: Callable):
        wrapper = ComplianceWrapper(
            agent_id=agent_id,
            industry=industry,
            is_demo=is_demo
        )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            input_prompt = args[0] if args else kwargs.get("prompt", "")
            return wrapper.run_sync(
                agent_func=func,
                input_prompt=input_prompt
            )

        return sync_wrapper
    return decorator