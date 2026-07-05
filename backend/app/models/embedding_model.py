"""
Medical QA System - Embedding Model State.

Dataclass that captures the runtime state of a loaded SentenceTransformer
embedding model.  Used internally by :class:`EmbeddingService` to keep
model metadata in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingModelState:
    """
    Holds the runtime state of a loaded embedding model.

    Attributes:
        model:       The SentenceTransformer model object.
        model_path:  Filesystem path from which the model was loaded (or saved).
        model_id:    Hugging Face model identifier string.
        is_loaded:   Whether the model has been successfully loaded.
        load_time:   Seconds elapsed during model loading.
        embedding_dim: Dimensionality of the produced embedding vectors.
    """

    model: Any
    model_path: str
    model_id: str
    is_loaded: bool = field(default=False)
    load_time: float = field(default=0.0)
    embedding_dim: int = field(default=384)
