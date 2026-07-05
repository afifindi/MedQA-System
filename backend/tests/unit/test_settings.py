"""
Unit Tests - Application Settings.

Tests default configuration values and environment variable overrides.
"""

from __future__ import annotations

import pytest

from app.core.config.settings import Settings


class TestSettings:
    """Tests for the Settings Pydantic model."""

    def test_default_model_path(self) -> None:
        """Default MODEL_PATH should point to the local gemma-2b-it directory."""
        s = Settings()
        assert "gemma-2b-it" in s.MODEL_PATH

    def test_default_lora_path(self) -> None:
        """Default LORA_PATH should point to the gemma_medical_qa_final directory."""
        s = Settings()
        assert "gemma_medical_qa_final" in s.LORA_PATH

    def test_default_port(self) -> None:
        """Default PORT should be 8000."""
        s = Settings()
        assert s.PORT == 8000

    def test_default_device(self) -> None:
        """Default DEVICE should be 'auto'."""
        s = Settings()
        assert s.DEVICE == "auto"

    def test_default_max_new_tokens(self) -> None:
        """Default MAX_NEW_TOKENS should be 512."""
        s = Settings()
        assert s.MAX_NEW_TOKENS == 512

    def test_default_temperature(self) -> None:
        """Default TEMPERATURE should be 0.7."""
        s = Settings()
        assert s.TEMPERATURE == pytest.approx(0.7)

    def test_default_top_k_retrieval(self) -> None:
        """Default TOP_K_RETRIEVAL should be 3."""
        s = Settings()
        assert s.TOP_K_RETRIEVAL == 3

    def test_default_max_question_length(self) -> None:
        """Default MAX_QUESTION_LENGTH should be 500."""
        s = Settings()
        assert s.MAX_QUESTION_LENGTH == 500

    def test_env_override_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PORT can be overridden via environment variable."""
        monkeypatch.setenv("PORT", "9000")
        s = Settings()
        assert s.PORT == 9000

    def test_env_override_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEVICE can be overridden via environment variable."""
        monkeypatch.setenv("DEVICE", "cpu")
        s = Settings()
        assert s.DEVICE == "cpu"

    def test_env_override_max_new_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_NEW_TOKENS can be overridden via environment variable."""
        monkeypatch.setenv("MAX_NEW_TOKENS", "256")
        s = Settings()
        assert s.MAX_NEW_TOKENS == 256

    def test_invalid_log_level_raises(self) -> None:
        """Invalid LOG_LEVEL should raise a validation error."""
        with pytest.raises(Exception):
            Settings(LOG_LEVEL="INVALID_LEVEL")

    def test_valid_log_levels(self) -> None:
        """All standard log levels should be accepted."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            s = Settings(LOG_LEVEL=level)
            assert s.LOG_LEVEL == level

    def test_app_name_default(self) -> None:
        """Default APP_NAME should be set."""
        s = Settings()
        assert s.APP_NAME == "Medical QA System"

    def test_app_version_default(self) -> None:
        """Default APP_VERSION should be a non-empty string."""
        s = Settings()
        assert s.APP_VERSION
        assert "." in s.APP_VERSION  # Semantic version has dots
