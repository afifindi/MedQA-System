"""
Medical QA System - Model Info Routes.

Defines GET /api/model and GET /api/version endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.controllers.model_controller import ModelController
from app.schemas.response import ModelInfoResponse, VersionResponse

router = APIRouter(tags=["Model"])

_model_controller = ModelController()


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Retrieve loaded model information",
    description=(
        "Returns the identifiers and runtime status of the generator model, "
        "embedding model, and LoRA adapter currently loaded in memory."
    ),
)
async def get_model_info(request: Request) -> ModelInfoResponse:
    """
    GET /api/model

    Returns metadata about the AI models currently loaded.

    Args:
        request: The Starlette request, used to access ``app.state``.

    Returns:
        A :class:`ModelInfoResponse` with model identifiers and status.
    """
    state = request.app.state
    return _model_controller.get_model_info(
        gemma_svc=state.gemma_service,
        embedding_svc=state.embedding_service,
        settings=state.settings,
    )


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Retrieve application version",
    description="Returns the application name, semantic version, and build date.",
)
async def get_version(request: Request) -> VersionResponse:
    """
    GET /api/version

    Returns application version metadata.

    Args:
        request: The Starlette request, used to access ``app.state``.

    Returns:
        A :class:`VersionResponse` with version string and app name.
    """
    return _model_controller.get_version(settings=request.app.state.settings)
