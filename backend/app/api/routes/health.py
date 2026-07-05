"""
Medical QA System - Health Route.

Defines GET /api/health endpoint for liveness and readiness probing.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.controllers.health_controller import HealthController
from app.schemas.response import HealthResponse

router = APIRouter(tags=["Health"])

# Module-level controller instance (stateless except for start_time which is
# set once at import time, giving approximate uptime)
_health_controller = HealthController()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description=(
        "Returns the operational status of all system components.  "
        "Status is 'healthy' when all ML services are loaded and ready; "
        "'degraded' when one or more components are unavailable."
    ),
    responses={
        200: {"description": "Health status (may be 'healthy' or 'degraded')."},
    },
)
async def health_check(request: Request) -> HealthResponse:
    """
    GET /api/health

    Returns the operational status of all ML components.

    Args:
        request: The Starlette request, used to access ``app.state``.

    Returns:
        A :class:`HealthResponse` with overall status and per-component info.
    """
    state = request.app.state
    return _health_controller.check(
        embedding_svc=state.embedding_service,
        retrieval_svc=state.retrieval_service,
        kb_svc=state.kb_service,
        gemma_svc=state.gemma_service,
    )
