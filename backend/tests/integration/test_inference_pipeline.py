"""
Integration Tests - InferenceService Pipeline.

Tests the full RAG pipeline using mocked ML services.  No real model weights
are loaded; all dependencies are replaced with unittest.mock.MagicMock
instances to isolate the orchestration logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.config.settings import Settings
from app.services.inference_service import InferenceService
from app.services.prompt_builder_service import PromptBuilderService


@pytest.fixture()
def settings() -> Settings:
    """Return a Settings instance with test-friendly values."""
    return Settings(
        MAX_QUESTION_LENGTH=500,
        TOP_K_RETRIEVAL=3,
        DEVICE="cpu",
    )


@pytest.fixture()
def mock_embedding_svc() -> MagicMock:
    svc = MagicMock()
    svc.encode.return_value = np.zeros(384, dtype=np.float32)
    svc.is_loaded = True
    return svc


@pytest.fixture()
def mock_retrieval_svc() -> MagicMock:
    svc = MagicMock()
    svc.search.return_value = [
        {"index": 0, "distance": 0.1, "score": 0.909},
        {"index": 1, "distance": 0.2, "score": 0.833},
        {"index": 2, "distance": 0.3, "score": 0.769},
    ]
    svc.is_loaded = True
    return svc


@pytest.fixture()
def mock_kb_svc() -> MagicMock:
    svc = MagicMock()
    svc.get_documents_by_indices.return_value = [
        {"answer": "Diabetes is a chronic condition."},
        {"answer": "Blood sugar levels are elevated."},
        {"answer": "Insulin resistance is a key factor."},
    ]
    svc.is_loaded = True
    return svc


@pytest.fixture()
def mock_gemma_svc() -> MagicMock:
    svc = MagicMock()
    svc.generate.return_value = ("Diabetes is a chronic metabolic condition.", 0.5)
    svc.is_loaded = True
    svc.device = "cpu"
    return svc


@pytest.fixture()
def inference_svc(
    mock_embedding_svc: MagicMock,
    mock_retrieval_svc: MagicMock,
    mock_kb_svc: MagicMock,
    mock_gemma_svc: MagicMock,
    settings: Settings,
) -> InferenceService:
    """Return an InferenceService wired with all mock dependencies."""
    return InferenceService(
        embedding_svc=mock_embedding_svc,
        retrieval_svc=mock_retrieval_svc,
        kb_svc=mock_kb_svc,
        prompt_svc=PromptBuilderService(),
        gemma_svc=mock_gemma_svc,
        settings=settings,
    )


class TestInferencePipeline:
    """Integration tests for InferenceService.answer()."""

    def test_answer_returns_required_keys(
        self, inference_svc: InferenceService
    ) -> None:
        """Result dict must contain all required keys."""
        result = inference_svc.answer("What is diabetes?")
        required = {
            "answer",
            "context_used",
            "inference_time",
            "retrieval_time",
            "generation_time",
            "embedding_time",
            "documents_retrieved",
        }
        assert required.issubset(result.keys())

    def test_answer_calls_embedding_encode(
        self, inference_svc: InferenceService, mock_embedding_svc: MagicMock
    ) -> None:
        """EmbeddingService.encode() must be called exactly once."""
        inference_svc.answer("What is diabetes?")
        mock_embedding_svc.encode.assert_called_once()

    def test_answer_calls_retrieval_search(
        self, inference_svc: InferenceService, mock_retrieval_svc: MagicMock
    ) -> None:
        """RetrievalService.search() must be called exactly once."""
        inference_svc.answer("What is diabetes?")
        mock_retrieval_svc.search.assert_called_once()

    def test_answer_calls_gemma_generate(
        self, inference_svc: InferenceService, mock_gemma_svc: MagicMock
    ) -> None:
        """GemmaService.generate() must be called exactly once."""
        inference_svc.answer("What is diabetes?")
        mock_gemma_svc.generate.assert_called_once()

    def test_answer_contains_correct_answer_text(
        self, inference_svc: InferenceService
    ) -> None:
        """The answer field must contain the text returned by the mock."""
        result = inference_svc.answer("What is diabetes?")
        assert "Diabetes is a chronic metabolic condition." in result["answer"]

    def test_answer_documents_retrieved_count(
        self, inference_svc: InferenceService
    ) -> None:
        """documents_retrieved must equal the number of documents returned."""
        result = inference_svc.answer("What is diabetes?")
        assert result["documents_retrieved"] == 3

    def test_answer_context_used_has_correct_structure(
        self, inference_svc: InferenceService
    ) -> None:
        """Each context_used entry must have index, content, and score."""
        result = inference_svc.answer("What is diabetes?")
        for doc in result["context_used"]:
            assert "index" in doc
            assert "content" in doc
            assert "score" in doc

    def test_answer_raises_on_empty_question(
        self, inference_svc: InferenceService
    ) -> None:
        """Empty question must raise ValueError."""
        with pytest.raises(ValueError):
            inference_svc.answer("")

    def test_answer_raises_on_too_long_question(
        self, inference_svc: InferenceService
    ) -> None:
        """Question exceeding MAX_QUESTION_LENGTH must raise ValueError."""
        with pytest.raises(ValueError):
            inference_svc.answer("x" * 600)

    def test_answer_timing_values_are_non_negative(
        self, inference_svc: InferenceService
    ) -> None:
        """All timing fields must be >= 0."""
        result = inference_svc.answer("What is hypertension?")
        assert result["inference_time"] >= 0
        assert result["retrieval_time"] >= 0
        assert result["generation_time"] >= 0
        assert result["embedding_time"] >= 0
