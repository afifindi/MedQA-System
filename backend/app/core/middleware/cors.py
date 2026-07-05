"""
Medical QA System - CORS Middleware Configuration.

Configures Cross-Origin Resource Sharing (CORS) for the FastAPI application,
allowing the React frontend to communicate with the backend API.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config.settings import Settings


def setup_cors(app: FastAPI, settings: Settings) -> None:
    """
    Attach the CORSMiddleware to the FastAPI application.

    Should be called once during application startup before any request is
    processed.  The allowed origins are read from ``settings.CORS_ORIGINS``;
    set this to specific domain names in production environments.

    Args:
        app:      The FastAPI application instance.
        settings: The application settings object.

    Example::

        from app.core.middleware.cors import setup_cors
        from app.core.config.settings import get_settings

        setup_cors(app, get_settings())
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-Request-ID"],
    )
