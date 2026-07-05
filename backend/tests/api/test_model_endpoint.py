"""
API Tests - GET /api/model and GET /api/version Endpoints.

Tests model info and version endpoints with mocked service state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.model import router as model_router
from app.core.config.settings import Settings
from app.core.middleware.rate_limit import setup_rate_limit
from app.services.embedding_service import EmbeddingService
from app.services.gemma_service import GemmaService


@pytest.fixture()
def test_app() -> FastAPI:
    """Create a minimal FastAPI app with mocked model state."""
    settings = Settings()
    app = FastAPI()
    setup_rate_limit(app)
    app.include_router(model_router, prefix="/api")

    gemma_svc = MagicMock(spec=GemmaService)
    gemma_svc.is_loaded = True
    gemma_svc.device = "cpu"
    gemma_svc.load_time = 5.0

    embedding_svc = MagicMock(spec=EmbeddingService)
    embedding_svc.is_loaded = True
    embedding_svc.load_time = 1.0

    app.state.settings = settings
    app.state.gemma_service = gemma_svc
    app.state.embedding_service = embedding_svc

    return app


@pytest.fixture()
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app)


class TestModelEndpoint:
    """Tests for GET /api/model."""

    def test_get_model_returns_200(self, client: TestClient) -> None:
        """Model info endpoint must return HTTP 200."""
        resp = client.get("/api/model")
        assert resp.status_code == 200

    def test_get_model_has_generator_model_field(self, client: TestClient) -> None:
        """Response must contain 'generator_model' field."""
        resp = client.get("/api/model")
        data = resp.json()
        assert "generator_model" in data
        assert "gemma" in data["generator_model"].lower()

    def test_get_model_has_embedding_model_field(self, client: TestClient) -> None:
        """Response must contain 'embedding_model' field."""
        resp = client.get("/api/model")
        data = resp.json()
        assert "embedding_model" in data

    def test_get_model_has_device_field(self, client: TestClient) -> None:
        """Response must contain 'device' field."""
        resp = client.get("/api/model")
        data = resp.json()
        assert "device" in data
        assert data["device"] in ("cpu", "cuda", "unknown")

    def test_get_model_has_is_loaded_field(self, client: TestClient) -> None:
        """Response must contain boolean 'is_loaded' field."""
        resp = client.get("/api/model")
        data = resp.json()
        assert "is_loaded" in data
        assert isinstance(data["is_loaded"], bool)


class TestVersionEndpoint:
    """Tests for GET /api/version."""

    def test_get_version_returns_200(self, client: TestClient) -> None:
        """Version endpoint must return HTTP 200."""
        resp = client.get("/api/version")
        assert resp.status_code == 200

    def test_get_version_has_version_field(self, client: TestClient) -> None:
        """Response must contain 'version' string."""
        resp = client.get("/api/version")
        data = resp.json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_get_version_has_app_name_field(self, client: TestClient) -> None:
        """Response must contain 'app_name' string."""
        resp = client.get("/api/version")
        data = resp.json()
        assert "app_name" in data

    def test_get_version_has_build_date_field(self, client: TestClient) -> None:
        """Response must contain 'build_date' string."""
        resp = client.get("/api/version")
        data = resp.json()
        assert "build_date" in data
