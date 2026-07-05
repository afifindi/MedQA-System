"""
Medical QA System - Response Schemas.

Defines Pydantic models for all API response bodies.  These models are used
by FastAPI to serialise response data and to generate the OpenAPI schema.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / nested models
# ---------------------------------------------------------------------------

class ContextDocument(BaseModel):
    """
    A single retrieved knowledge-base document included in the chat response.

    Attributes:
        index:   Zero-based row index in the knowledge base CSV.
        content: The text content of the retrieved document.
        score:   Similarity score in the range ``(0, 1]``; higher is better.
    """

    index: int = Field(..., description="Row index in the knowledge base.")
    content: str = Field(..., description="Text content of the retrieved document.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity score (higher = more relevant)."
    )


class ComponentStatus(BaseModel):
    """
    Operational status of a single system component.

    Attributes:
        name:    Machine-readable component identifier.
        status:  ``"ok"`` or ``"unavailable"``.
        message: Human-readable status description.
    """

    name: str = Field(..., description="Component identifier.")
    status: str = Field(..., description="'ok' or 'unavailable'.")
    message: str = Field(..., description="Human-readable status description.")


# ---------------------------------------------------------------------------
# Top-level response models
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):
    """
    Response body for POST /api/chat.

    Attributes:
        answer:              Generated answer text from the Gemma model.
        context_used:        List of retrieved documents used to ground the answer.
        inference_time:      Total wall-clock seconds for the full RAG pipeline.
        retrieval_time:      Seconds spent on FAISS vector search.
        generation_time:     Seconds spent on Gemma text generation.
        embedding_time:      Seconds spent encoding the question.
        documents_retrieved: Number of documents retrieved from the knowledge base.

    Example::

        {
            "answer": "Diabetes mellitus is a metabolic disease...",
            "context_used": [{"index": 42, "content": "...", "score": 0.87}],
            "inference_time": 2.341,
            "retrieval_time": 0.003,
            "generation_time": 2.315,
            "embedding_time": 0.023,
            "documents_retrieved": 3
        }
    """

    answer: str = Field(..., description="AI-generated answer text.")
    context_used: List[ContextDocument] = Field(
        ..., description="Retrieved knowledge-base documents used to generate the answer."
    )
    inference_time: float = Field(
        ..., ge=0.0, description="Total inference time in seconds."
    )
    retrieval_time: float = Field(
        ..., ge=0.0, description="FAISS retrieval time in seconds."
    )
    generation_time: float = Field(
        ..., ge=0.0, description="Gemma generation time in seconds."
    )
    embedding_time: float = Field(
        ..., ge=0.0, description="Question embedding time in seconds."
    )
    documents_retrieved: int = Field(
        ..., ge=0, description="Number of documents retrieved."
    )


class HealthResponse(BaseModel):
    """
    Response body for GET /api/health.

    Attributes:
        status:          ``"healthy"`` or ``"degraded"``.
        components:      Per-component status list.
        uptime_seconds:  Seconds since the application started.
        timestamp:       UTC ISO-8601 timestamp of the health check.
    """

    status: str = Field(..., description="'healthy' or 'degraded'.")
    components: List[ComponentStatus] = Field(
        ..., description="Status of each system component."
    )
    uptime_seconds: float = Field(..., ge=0.0, description="Application uptime in seconds.")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp.")


class ModelInfoResponse(BaseModel):
    """
    Response body for GET /api/model.

    Attributes:
        generator_model: Hugging Face model ID of the generator.
        embedding_model: Hugging Face model ID of the embedding model.
        lora_adapter:    Path to the LoRA adapter artefacts.
        device:          Compute device in use (``"cuda"`` or ``"cpu"``).
        is_loaded:       Whether all models are loaded and ready.
        version:         Application version string.
    """

    generator_model: str = Field(..., description="Generator model identifier.")
    embedding_model: str = Field(..., description="Embedding model identifier.")
    lora_adapter: str = Field(..., description="Path to the LoRA adapter.")
    device: str = Field(..., description="Compute device ('cuda' or 'cpu').")
    is_loaded: bool = Field(..., description="True if all models are loaded and ready.")
    version: str = Field(..., description="Application version.")


class VersionResponse(BaseModel):
    """
    Response body for GET /api/version.

    Attributes:
        version:    Semantic version string.
        app_name:   Human-readable application name.
        build_date: ISO-8601 date of the build.
    """

    version: str = Field(..., description="Semantic version string.")
    app_name: str = Field(..., description="Application name.")
    build_date: str = Field(..., description="ISO-8601 build date.")
