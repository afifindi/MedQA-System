"""
Medical QA System - API Routes Package.

Exports all APIRouter instances for registration in the FastAPI app.
"""

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.model import router as model_router

__all__ = ["chat_router", "health_router", "model_router"]
