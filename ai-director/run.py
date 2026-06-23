"""
AI Director — Entry Point
Run with: python run.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from app.config import settings

if __name__ == "__main__":
    # Allow PORT env override (e.g. preview/dev tooling) without editing config.
    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=port,
        reload=settings.debug,
        log_level="info",
    )
