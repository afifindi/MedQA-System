"""
Medical QA System - Request Schemas.

Defines Pydantic models for all inbound API request bodies.
Validation is performed automatically by FastAPI before the route handler is called.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """
    Request body for the POST /api/chat endpoint.

    Attributes:
        question: The medical question to answer.  Must be between 1 and 500
                  characters after whitespace stripping.

    Example::

        {
            "question": "What is diabetes mellitus?"
        }
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The medical question to answer (1–500 characters).",
        examples=["What is diabetes mellitus?"],
    )

    @field_validator("question", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """
        Strip leading and trailing whitespace from the question.

        Args:
            v: The raw question string.

        Returns:
            The stripped question string.

        Raises:
            ValueError: If the stripped string is empty.
        """
        stripped = v.strip() if isinstance(v, str) else v
        if not stripped:
            raise ValueError(
                "Question must not be empty or contain only whitespace."
            )
        return stripped

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What is diabetes mellitus?"},
                {"question": "What are the symptoms of hypertension?"},
                {"question": "How is pneumonia diagnosed?"},
            ]
        }
    }
