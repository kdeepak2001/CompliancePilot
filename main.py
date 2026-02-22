"""
CompliancePilot - Application Entry Point
==========================================
Run this file to start the entire system.

Usage:
    python main.py

Author: CompliancePilot
Version: 1.0.0
"""

import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )