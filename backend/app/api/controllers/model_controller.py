"""
Medical QA System - Model Info Controller.

Returns metadata about the loaded AI models (generator, embedding, LoRA,
device) and the application version.  Used by the frontend status panel
and monitoring dashboards.
"""

from __future__ import annotations

from datetime import date

from app.core.config.settings import Settings
from app.schemas.response import ModelInfoResponse, VersionResponse
from app.services.embedding_service import EmbeddingService
from app.services.gemma_service import GemmaService


class ModelController:
    """
    Controller for the GET /api/model and GET /api/version endpoints.

    All methods are synchronous (no I/O); they simply read already-loaded
    service state and format it into response models.
    """

    def get_model_info(
        self,
        gemma_svc: GemmaService,
        embedding_svc: EmbeddingService,
        settings: Settings,
    ) -> ModelInfoResponse:
        """
        Return metadata about the loaded AI models.

        Args:
            gemma_svc:     The :class:`GemmaService` instance.
            embedding_svc: The :class:`EmbeddingService` instance.
            settings:      Application settings.

        Returns:
            A :class:`ModelInfoResponse` with model IDs, device, and status.
        """
        return ModelInfoResponse(
            generator_model=settings.HF_GENERATOR_MODEL_ID,
            embedding_model=settings.HF_EMBEDDING_MODEL_ID,
            lora_adapter=settings.LORA_PATH,
            device=gemma_svc.device if gemma_svc.is_loaded else "unknown",
            is_loaded=gemma_svc.is_loaded and embedding_svc.is_loaded,
            version=settings.APP_VERSION,
        )

    def get_version(self, settings: Settings) -> VersionResponse:
        """
        Return the application version metadata.

        Args:
            settings: Application settings.

        Returns:
            A :class:`VersionResponse` with version, app name, and build date.
        """
        return VersionResponse(
            version=settings.APP_VERSION,
            app_name=settings.APP_NAME,
            build_date=str(date.today()),
        )
