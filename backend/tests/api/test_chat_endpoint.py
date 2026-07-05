"""
API Tests - POST /api/chat Endpoint.

Uses FastAPI TestClient with all ML services mocked on app.state to test
HTTP-level behaviour without loading real model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.chat import router as chat_router
from app.core.config.settings import Settings
from app.core.middleware.cors import setup_cors
from app.core.middleware.rate_limit import setup_rate_limit
from app.services.inference_service import InferenceService
from app.services.prompt_builder_service import PromptBuilderService


def _make_mock_inference_svc() -> MagicMock:
    """Build a mock InferenceService that returns a canned response."""
    svc = MagicMock(spec=InferenceService)
    svc.answer.return_value = {
        "answer": "Diabetes is a metabolic disease characterised by high blood sugar.",
        "context_used": [
            {"index": 0, "content": "Medical doc about diabetes.", "score": 0.95}
        ],
        "inference_time": 1.23,
        "retrieval_time": 0.01,
        "generation_time": 1.20,
        "embedding_time": 0.02,
        "documents_retrieved": 1,
    }
    return svc


@pytest.fixture()
def test_app() -> FastAPI:
    """Create a minimal FastAPI app with mocked state for testing."""
    settings = Settings()
    app = FastAPI()
    setup_rate_limit(app)
    app.include_router(chat_router, prefix="/api")

    # Attach mocked services to app.state
    app.state.settings = settings
    app.state.inference_service = _make_mock_inference_svc()

    return app


@pytest.fixture()
def client(test_app: FastAPI) -> TestClient:
    """Return a synchronous TestClient for the test app."""
    return TestClient(test_app)


class TestChatEndpoint:
    """Tests for POST /api/chat."""

    def test_valid_question_returns_200(self, client: TestClient) -> None:
        """A valid question must return HTTP 200."""
        resp = client.post("/api/chat", json={"question": "What is diabetes?"})
        assert resp.status_code == 200

    def test_valid_question_response_has_answer(self, client: TestClient) -> None:
        """Response must contain an 'answer' field."""
        resp = client.post("/api/chat", json={"question": "What is diabetes?"})
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_valid_question_response_has_context_used(
        self, client: TestClient
    ) -> None:
        """Response must contain 'context_used' as a list."""
        resp = client.post("/api/chat", json={"question": "What is diabetes?"})
        data = resp.json()
        assert "context_used" in data
        assert isinstance(data["context_used"], list)

    def test_valid_question_response_has_inference_time(
        self, client: TestClient
    ) -> None:
        """Response must contain 'inference_time' as a float >= 0."""
        resp = client.post("/api/chat", json={"question": "What is diabetes?"})
        data = resp.json()
        assert "inference_time" in data
        assert data["inference_time"] >= 0

    def test_empty_question_returns_422(self, client: TestClient) -> None:
        """Empty question string must return HTTP 422 (validation error)."""
        resp = client.post("/api/chat", json={"question": ""})
        assert resp.status_code == 422

    def test_whitespace_only_question_returns_422(self, client: TestClient) -> None:
        """Whitespace-only question must return HTTP 422."""
        resp = client.post("/api/chat", json={"question": "   "})
        assert resp.status_code == 422

    def test_too_long_question_returns_422(self, client: TestClient) -> None:
        """Question > 500 chars must return HTTP 422."""
        resp = client.post("/api/chat", json={"question": "x" * 501})
        assert resp.status_code == 422

    def test_missing_body_returns_422(self, client: TestClient) -> None:
        """Missing request body must return HTTP 422."""
        resp = client.post("/api/chat")
        assert resp.status_code == 422

    def test_missing_question_field_returns_422(self, client: TestClient) -> None:
        """Body without 'question' field must return HTTP 422."""
        resp = client.post("/api/chat", json={"wrong_field": "value"})
        assert resp.status_code == 422

    def test_inference_error_returns_500(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """If InferenceService raises RuntimeError, response must be HTTP 500."""
        test_app.state.inference_service.answer.side_effect = RuntimeError(
            "Simulated model failure"
        )
        resp = client.post("/api/chat", json={"question": "What is hypertension?"})
        assert resp.status_code == 500

    def test_validation_error_returns_400(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """If InferenceService raises ValueError, response must be HTTP 400."""
        test_app.state.inference_service.answer.side_effect = ValueError(
            "Question too long after sanitization"
        )
        resp = client.post("/api/chat", json={"question": "What is pneumonia?"})
        assert resp.status_code == 400
