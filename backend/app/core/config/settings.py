"""
Medical QA System - Application Settings.

Manages all configuration via environment variables with Pydantic BaseSettings.
All settings have sensible defaults and can be overridden via a .env file or
environment variables at runtime.

Usage:
    from app.core.config.settings import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration settings.

    Reads values from environment variables (case-insensitive) and from a
    .env file located in the working directory.  All model paths are relative
    to the ``backend/`` directory so that the project layout is consistent
    whether running locally or inside Docker.

    Attributes:
        MODEL_PATH: Local directory for the base Gemma-2B-IT model weights.
        LORA_PATH: Directory containing the fine-tuned LoRA adapter and
            pre-built FAISS index / knowledge base artefacts.
        FAISS_PATH: Full path to the ``faiss_index.bin`` file.
        EMBEDDING_MODEL_PATH: Local directory for the MiniLM embedding model.
        KNOWLEDGE_BASE_PATH: Full path to ``knowledge_base.csv``.
        KB_EMBEDDINGS_PATH: Full path to ``kb_embeddings.npy``.
        DEVICE: Compute device – ``"auto"``, ``"cuda"``, or ``"cpu"``.
        HOST: Uvicorn bind host.
        PORT: Uvicorn bind port.
        MAX_NEW_TOKENS: Maximum tokens the generator may produce.
        TEMPERATURE: Sampling temperature for text generation.
        TOP_P: Nucleus sampling probability mass.
        TOP_K: Top-K sampling value.
        TOP_K_RETRIEVAL: Number of documents returned from FAISS search.
        MAX_QUESTION_LENGTH: Maximum accepted question length (characters).
        LOG_LEVEL: Loguru log level string.
        LOG_DIR: Directory for log files (relative to backend/).
        APP_NAME: Human-readable application name.
        APP_VERSION: Semantic version string.
        CORS_ORIGINS: Allowed CORS origins list.
        HF_EMBEDDING_MODEL_ID: Hugging Face model ID for the embedding model.
        HF_GENERATOR_MODEL_ID: Hugging Face model ID for the generator.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Model Paths
    # -------------------------------------------------------------------------
    MODEL_PATH: str = Field(
        default="../models/gemma-2b-it",
        description="Local path to the base Gemma-2B-IT model directory.",
    )
    LORA_PATH: str = Field(
        default="../models/gemma_medical_qa_final",
        description="Local path to the LoRA adapter / training artefacts directory.",
    )
    FAISS_PATH: str = Field(
        default="../models/gemma_medical_qa_final/faiss_index.bin",
        description="Full path to the pre-built FAISS index binary file.",
    )
    EMBEDDING_MODEL_PATH: str = Field(
        default="../models/all-MiniLM-L6-v2",
        description="Local path to the all-MiniLM-L6-v2 embedding model directory.",
    )
    KNOWLEDGE_BASE_PATH: str = Field(
        default="../models/gemma_medical_qa_final/knowledge_base.csv",
        description="Full path to the medical knowledge base CSV file.",
    )
    KB_EMBEDDINGS_PATH: str = Field(
        default="../models/gemma_medical_qa_final/kb_embeddings.npy",
        description="Full path to the pre-computed knowledge base embeddings (.npy).",
    )

    # -------------------------------------------------------------------------
    # Hugging Face Model IDs (used when local copies are absent)
    # -------------------------------------------------------------------------
    HF_EMBEDDING_MODEL_ID: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face repository ID for the embedding model.",
    )
    HF_GENERATOR_MODEL_ID: str = Field(
        default="google/gemma-2b-it",
        description="Hugging Face repository ID for the Gemma generator model.",
    )

    # -------------------------------------------------------------------------
    # Device Configuration
    # -------------------------------------------------------------------------
    DEVICE: str = Field(
        default="auto",
        description=(
            "Compute device for model inference. "
            "'auto' detects CUDA automatically; 'cpu' forces CPU; 'cuda' forces GPU."
        ),
    )

    # -------------------------------------------------------------------------
    # Server Configuration
    # -------------------------------------------------------------------------
    HOST: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    PORT: int = Field(default=8000, ge=1, le=65535, description="Uvicorn bind port.")

    # -------------------------------------------------------------------------
    # Generation Hyper-parameters
    # -------------------------------------------------------------------------
    MAX_NEW_TOKENS: int = Field(
        default=512, ge=1, le=2048, description="Maximum tokens to generate."
    )
    TEMPERATURE: float = Field(
        default=0.7, gt=0.0, le=2.0, description="Sampling temperature."
    )
    TOP_P: float = Field(
        default=0.9, gt=0.0, le=1.0, description="Nucleus sampling probability."
    )
    TOP_K: int = Field(
        default=50, ge=1, description="Top-K sampling value."
    )
    TOP_K_RETRIEVAL: int = Field(
        default=3, ge=1, le=20, description="Number of documents to retrieve from FAISS."
    )

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------
    MAX_QUESTION_LENGTH: int = Field(
        default=500, ge=1, description="Maximum allowed question length in characters."
    )
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Use specific domains in production.",
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Loguru log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    LOG_DIR: str = Field(
        default="logs",
        description="Directory where log files are written (relative to backend/).",
    )

    # -------------------------------------------------------------------------
    # Application Metadata
    # -------------------------------------------------------------------------
    APP_NAME: str = Field(
        default="Medical QA System",
        description="Human-readable application name.",
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Semantic version of the application.",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure LOG_LEVEL is a valid Loguru level name."""
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'.")
        return upper

    @field_validator("DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure DEVICE is one of the accepted values."""
        allowed = {"auto", "cuda", "cpu"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"DEVICE must be one of {allowed}, got '{v}'.")
        return lower


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached singleton Settings instance.

    Uses ``lru_cache`` so the .env file is read exactly once per process.

    Returns:
        Settings: The application configuration object.

    Example::

        from app.core.config.settings import get_settings
        settings = get_settings()
        print(settings.APP_VERSION)
    """
    return Settings()
