"""
Medical QA System - FastAPI Application Entry Point.

Creates and configures the FastAPI application with:
  - Lifespan context manager for startup model loading and graceful shutdown.
  - CORS middleware (reads allowed origins from settings).
  - Rate-limiting middleware via SlowAPI.
  - API routers registered under the /api prefix.
  - OpenAPI metadata for interactive documentation.

Startup sequence:
  1. Configure logging (LoggerService).
  2. Validate all required artefact paths (ConfigurationService).
  3. Load EmbeddingService (MiniLM model).
  4. Load KnowledgeBaseService (CSV).
  5. Load RetrievalService (FAISS index + embeddings).
  6. Load GemmaService (base model + LoRA adapter).
  7. Assemble InferenceService and attach all services to app.state.

All services are stored as singletons on ``app.state`` and retrieved in
route handlers via the dependency providers in ``app.core.dependencies``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes import chat_router, health_router, model_router
from app.core.config.settings import Settings, get_settings
from app.core.middleware.cors import setup_cors
from app.core.middleware.rate_limit import setup_rate_limit
from app.services.configuration_service import ConfigurationService
from app.services.embedding_service import EmbeddingService
from app.services.gemma_service import GemmaService
from app.services.inference_service import InferenceService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.logger_service import get_logger
from app.services.prompt_builder_service import PromptBuilderService
from app.services.retrieval_service import RetrievalService


# ---------------------------------------------------------------------------
# Lifespan – model loading and cleanup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown lifecycle.

    On startup:
      - Configure the logger.
      - Validate all required paths via ConfigurationService.
      - Load all ML services sequentially.
      - Attach loaded services and settings to ``app.state``.

    On shutdown:
      - Log a goodbye message.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running application.

    Raises:
        FileNotFoundError: If a mandatory artefact (FAISS, CSV, LoRA) is missing.
        RuntimeError:      If any model fails to load.
    """
    settings: Settings = get_settings()
    logger = get_logger()

    # --- Configure logging ---
    logger.configure(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)
    logger.log_startup("=" * 60)
    logger.log_startup(f"  {settings.APP_NAME}  v{settings.APP_VERSION}")
    logger.log_startup("  Starting up …")
    logger.log_startup("=" * 60)

    t_boot = time.perf_counter()

    # --- Validate configuration and artefact paths ---
    ConfigurationService.validate_all(settings)

    # --- Resolve compute device ---
    device = ConfigurationService.get_device(settings)

    # --- Load Embedding Service ---
    logger.log_startup("Loading EmbeddingService …")
    embedding_svc = EmbeddingService(settings)
    embedding_svc.load()

    # --- Load Knowledge Base ---
    logger.log_startup("Loading KnowledgeBaseService …")
    kb_svc = KnowledgeBaseService(settings)
    kb_svc.load()

    # --- Load Retrieval Service ---
    logger.log_startup("Loading RetrievalService (FAISS) …")
    retrieval_svc = RetrievalService(settings)
    retrieval_svc.load()

    # --- Load Gemma Service ---
    logger.log_startup("Loading GemmaService (Gemma-2B-IT + LoRA) …")
    gemma_svc = GemmaService(settings)
    gemma_svc.load(device=device)

    # --- Assemble InferenceService ---
    prompt_svc = PromptBuilderService()
    inference_svc = InferenceService(
        embedding_svc=embedding_svc,
        retrieval_svc=retrieval_svc,
        kb_svc=kb_svc,
        prompt_svc=prompt_svc,
        gemma_svc=gemma_svc,
        settings=settings,
    )

    # --- Attach to app.state ---
    app.state.settings = settings
    app.state.embedding_service = embedding_svc
    app.state.retrieval_service = retrieval_svc
    app.state.kb_service = kb_svc
    app.state.gemma_service = gemma_svc
    app.state.inference_service = inference_svc

    boot_time = time.perf_counter() - t_boot
    logger.log_startup(f"All services loaded in {boot_time:.2f}s.  Ready to serve requests.")
    logger.log_startup(f"  ➜  API docs:   http://{settings.HOST}:{settings.PORT}/docs")
    logger.log_startup(f"  ➜  Health:     http://{settings.HOST}:{settings.PORT}/api/health")
    logger.log_startup(f"  ➜  Chat:       POST http://{settings.HOST}:{settings.PORT}/api/chat")

    yield  # --- Application is running ---

    # --- Shutdown ---
    logger.log_startup("Shutting down Medical QA System …")
    logger.log_startup("Goodbye.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        The fully configured :class:`FastAPI` instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "A production-ready Medical Question Answering System using "
            "Retrieval-Augmented Generation (RAG) with Gemma-2B-IT + LoRA "
            "and FAISS-based semantic retrieval."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "Medical QA System",
            "url": "https://github.com/medical-qa-system",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # --- Middleware ---
    setup_cors(app, settings)
    setup_rate_limit(app)

    # --- Routes ---
    app.include_router(chat_router, prefix="/api")
    app.include_router(health_router, prefix="/api")
    app.include_router(model_router, prefix="/api")

    # --- Root endpoint ---
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        """Root redirect info."""
        return JSONResponse(
            content={
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "docs": "/docs",
                "health": "/api/health",
                "chat": "/api/chat",
            }
        )

    return app


# ---------------------------------------------------------------------------
# Application instance (used by uvicorn and tests)
# ---------------------------------------------------------------------------
app = create_app()


# ---------------------------------------------------------------------------
# Development server entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "main:app",
        host=cfg.HOST,
        port=cfg.PORT,
        reload=False,
        workers=1,
        log_level=cfg.LOG_LEVEL.lower(),
        access_log=True,
    )
