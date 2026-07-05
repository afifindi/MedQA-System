"""
API Tests - GET /api/health Endpoint.

Tests the health check endpoint with mocked service states.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.health import router as health_router
from app.core.config.settings import Settings
from app.core.middleware.rate_limit import setup_rate_limit
from app.services.embedding_service import EmbeddingService
from app.services.gemma_service import GemmaService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.retrieval_service import RetrievalService


def _mock_loaded_service(cls, load_time: float = 1.0) -> MagicMock:
    svc = MagicMock(spec=cls)
    svc.is_loaded = True
    svc.load_time = load_time
    return svc


@pytest.fixture()
def test_app() -> FastAPI:
    """Create a minimal FastAPI app with mocked service state."""
    settings = Settings()
    app = FastAPI()
    setup_rate_limit(app)
    app.include_router(health_router, prefix="/api")

    embedding_svc = _mock_loaded_service(EmbeddingService)
    retrieval_svc = _mock_loaded_service(RetrievalService)
    retrieval_svc.index_size = 1000

    kb_svc = _mock_loaded_service(KnowledgeBaseService)
    kb_svc.get_document_count.return_value = 500

    gemma_svc = _mock_loaded_service(GemmaService)
    gemma_svc.device = "cpu"

    app.state.settings = settings
    app.state.embedding_service = embedding_svc
    app.state.retrieval_service = retrieval_svc
    app.state.kb_service = kb_svc
    app.state.gemma_service = gemma_svc

    return app


@pytest.fixture()
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app)


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_get_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint must return HTTP 200."""
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_get_health_has_status_field(self, client: TestClient) -> None:
        """Response must contain a 'status' field."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_get_health_has_components_field(self, client: TestClient) -> None:
        """Response must contain a 'components' list."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "components" in data
        assert isinstance(data["components"], list)
        assert len(data["components"]) > 0

    def test_get_health_has_uptime_field(self, client: TestClient) -> None:
        """Response must contain 'uptime_seconds' >= 0."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_get_health_has_timestamp_field(self, client: TestClient) -> None:
        """Response must contain a 'timestamp' ISO string."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

    def test_all_services_loaded_gives_healthy(self, client: TestClient) -> None:
        """When all services are loaded, status should be 'healthy'."""
        resp = client.get("/api/health")
        data = resp.json()
        assert data["status"] == "healthy"

    def test_components_have_required_fields(self, client: TestClient) -> None:
        """Each component entry must have 'name', 'status', 'message'."""
        resp = client.get("/api/health")
        data = resp.json()
        for comp in data["components"]:
            assert "name" in comp
            assert "status" in comp
            assert "message" in comp
