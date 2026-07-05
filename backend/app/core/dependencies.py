"""
Medical QA System - FastAPI Dependency Providers.

Provides FastAPI dependency functions that retrieve singleton service instances
stored on ``app.state`` during application startup.  Using ``app.state`` as
the service registry allows the lifespan context manager to control the
lifecycle of heavy ML models while keeping route handlers thin.

Usage in route handlers::

    from fastapi import Depends, Request
    from app.core.dependencies import get_inference_service
    from app.services.inference_service import InferenceService

    @router.post("/chat")
    async def chat(
        request: Request,
        inference_svc: InferenceService = Depends(get_inference_service),
    ) -> ChatResponse:
        ...
"""

from __future__ import annotations

from fastapi import Request

from app.core.config.settings import Settings
from app.services.embedding_service import EmbeddingService
from app.services.gemma_service import GemmaService
from app.services.inference_service import InferenceService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.retrieval_service import RetrievalService


def get_settings(request: Request) -> Settings:
    """
    Return the application settings from ``app.state``.

    Args:
        request: The current HTTP request.

    Returns:
        The :class:`Settings` singleton stored at startup.
    """
    return request.app.state.settings


def get_inference_service(request: Request) -> InferenceService:
    """
    Return the loaded :class:`InferenceService` from ``app.state``.

    Args:
        request: The current HTTP request.

    Returns:
        The singleton :class:`InferenceService` instance.
    """
    return request.app.state.inference_service


def get_embedding_service(request: Request) -> EmbeddingService:
    """
    Return the loaded :class:`EmbeddingService` from ``app.state``.

    Args:
        request: The current HTTP request.

    Returns:
        The singleton :class:`EmbeddingService` instance.
    """
    return request.app.state.embedding_service


def get_retrieval_service(request: Request) -> RetrievalService:
    """
    Return the loaded :class:`RetrievalService` from ``app.state``.

    Args:
        request: The current HTTP request.

    Returns:
        The singleton :class:`RetrievalService` instance.
    """
    return request.app.state.retrieval_service


def get_kb_service(request: Request) -> KnowledgeBaseService:
    """
    Return the loaded :class:`KnowledgeBaseService` from ``app.state``.

    Args:
        request: The current HTTP request.

    Returns:
        The singleton :class:`KnowledgeBaseService` instance.
    """
    return request.app.state.kb_service


def get_gemma_service(request: Request) -> GemmaService:
    """
    Return the loaded :class:`GemmaService` from ``app.state``.

    Args:
        request: The current HTTP request.

    Returns:
        The singleton :class:`GemmaService` instance.
    """
    return request.app.state.gemma_service
