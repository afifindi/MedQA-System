"""
Medical QA System - Health Controller.

Aggregates the loaded/not-loaded status of all ML services and returns a
structured health-check response suitable for monitoring systems (Docker
healthcheck, load balancers, etc.).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List

from app.schemas.response import ComponentStatus, HealthResponse
from app.services.embedding_service import EmbeddingService
from app.services.gemma_service import GemmaService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.retrieval_service import RetrievalService


class HealthController:
    """
    Controller for the GET /api/health endpoint.

    Aggregates service status information into a :class:`HealthResponse`.
    The controller tracks the application start time so that uptime can be
    included in the response.
    """

    def __init__(self) -> None:
        """Initialise the controller and record the start timestamp."""
        self._start_time: float = time.time()

    def check(
        self,
        embedding_svc: EmbeddingService,
        retrieval_svc: RetrievalService,
        kb_svc: KnowledgeBaseService,
        gemma_svc: GemmaService,
    ) -> HealthResponse:
        """
        Build a health response by interrogating each service.

        Args:
            embedding_svc: The :class:`EmbeddingService` instance.
            retrieval_svc: The :class:`RetrievalService` instance.
            kb_svc:        The :class:`KnowledgeBaseService` instance.
            gemma_svc:     The :class:`GemmaService` instance.

        Returns:
            A :class:`HealthResponse` with overall status, component details,
            uptime in seconds, and a UTC ISO-8601 timestamp.
        """
        components: List[ComponentStatus] = [
            ComponentStatus(
                name="embedding_model",
                status="ok" if embedding_svc.is_loaded else "unavailable",
                message=(
                    f"Loaded (took {embedding_svc.load_time:.2f}s)"
                    if embedding_svc.is_loaded
                    else "Not loaded"
                ),
            ),
            ComponentStatus(
                name="faiss_index",
                status="ok" if retrieval_svc.is_loaded else "unavailable",
                message=(
                    f"Loaded – {retrieval_svc.index_size} vectors"
                    if retrieval_svc.is_loaded
                    else "Not loaded"
                ),
            ),
            ComponentStatus(
                name="knowledge_base",
                status="ok" if kb_svc.is_loaded else "unavailable",
                message=(
                    f"Loaded – {kb_svc.get_document_count()} documents"
                    if kb_svc.is_loaded
                    else "Not loaded"
                ),
            ),
            ComponentStatus(
                name="gemma_model",
                status="ok" if gemma_svc.is_loaded else "unavailable",
                message=(
                    f"Loaded on {gemma_svc.device} (took {gemma_svc.load_time:.2f}s)"
                    if gemma_svc.is_loaded
                    else "Not loaded"
                ),
            ),
        ]

        all_ok = all(c.status == "ok" for c in components)
        overall_status = "healthy" if all_ok else "degraded"

        return HealthResponse(
            status=overall_status,
            components=components,
            uptime_seconds=round(time.time() - self._start_time, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
