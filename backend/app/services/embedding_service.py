"""
Medical QA System - Embedding Service.

Wraps the SentenceTransformers ``all-MiniLM-L6-v2`` model to produce
normalised L2 dense embeddings for user questions.

On first startup the service checks whether the model weights exist locally.
If absent, they are downloaded from the Hugging Face Hub and saved so that
subsequent starts do not require a network connection.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.core.config.settings import Settings
from app.services.logger_service import get_logger

logger = get_logger()


class EmbeddingService:
    """
    Service responsible for encoding text into dense vector embeddings.

    Uses the ``sentence-transformers/all-MiniLM-L6-v2`` model which produces
    384-dimensional L2-normalised embeddings.  The model is loaded once at
    startup and kept in memory for the lifetime of the application.

    Attributes:
        _settings:   Application configuration.
        _model:      The loaded SentenceTransformer model (None until loaded).
        _is_loaded:  Whether the model has been successfully loaded.
        _load_time:  Seconds taken to load the model.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialise the EmbeddingService without loading the model.

        Args:
            settings: The application settings object.
        """
        self._settings = settings
        self._model: Optional[object] = None
        self._is_loaded: bool = False
        self._load_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the embedding model from disk or download it from Hugging Face.

        If ``settings.EMBEDDING_MODEL_PATH`` points to an existing directory
        the model is loaded from there.  Otherwise, the model identified by
        ``settings.HF_EMBEDDING_MODEL_ID`` is downloaded from the Hugging Face
        Hub and saved to ``settings.EMBEDDING_MODEL_PATH`` for future use.

        Raises:
            RuntimeError: If the model cannot be loaded or downloaded.
        """
        from sentence_transformers import SentenceTransformer

        t_start = time.perf_counter()
        model_path = Path(self._settings.EMBEDDING_MODEL_PATH)

        try:
            if model_path.is_dir():
                logger.log_model(
                    f"Loading embedding model from local path: {model_path.resolve()}"
                )
                self._model = SentenceTransformer(str(model_path))
                logger.log_model("Embedding model loaded from local cache.")
            else:
                model_id = self._settings.HF_EMBEDDING_MODEL_ID
                logger.log_model(
                    f"Embedding model not found locally.  "
                    f"Downloading '{model_id}' from Hugging Face Hub …"
                )
                self._model = SentenceTransformer(model_id)
                # Persist so subsequent starts use the local copy
                model_path.mkdir(parents=True, exist_ok=True)
                self._model.save(str(model_path))
                logger.log_model(
                    f"Embedding model downloaded and saved to: {model_path.resolve()}"
                )

            self._load_time = time.perf_counter() - t_start
            self._is_loaded = True
            logger.log_model(
                f"Embedding model ready.  Load time: {self._load_time:.2f}s"
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model from '{model_path}' or "
                f"'{self._settings.HF_EMBEDDING_MODEL_ID}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a normalised embedding vector.

        Args:
            text: The input text to embed.

        Returns:
            A 1-D float32 NumPy array of shape ``(384,)``.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        self._assert_loaded()
        return self._model.encode(  # type: ignore[union-attr]
            [text], normalize_embeddings=True
        )[0].astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings into a matrix of embedding vectors.

        Args:
            texts: List of input strings.

        Returns:
            A 2-D float32 NumPy array of shape ``(len(texts), 384)``.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        self._assert_loaded()
        return self._model.encode(  # type: ignore[union-attr]
            texts, normalize_embeddings=True
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the embedding model has been successfully loaded."""
        return self._is_loaded

    @property
    def load_time(self) -> float:
        """Seconds elapsed during model loading."""
        return self._load_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        """Raise RuntimeError if the model has not been loaded."""
        if not self._is_loaded or self._model is None:
            raise RuntimeError(
                "EmbeddingService.load() must be called before encoding text."
            )
