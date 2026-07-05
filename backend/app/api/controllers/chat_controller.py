"""
Medical QA System - Chat Controller.

Handles the business logic for the POST /api/chat endpoint.  Delegates all
ML work to :class:`InferenceService` and maps the result to the API response
schema.  HTTP error handling is performed here, not in the route handler.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse, ContextDocument
from app.services.inference_service import InferenceService
from app.services.logger_service import get_logger

logger = get_logger()


class ChatController:
    """
    Controller for the chat inference endpoint.

    Attributes:
        _inference_service: The :class:`InferenceService` that runs the RAG pipeline.
    """

    def __init__(self, inference_service: InferenceService) -> None:
        """
        Initialise the ChatController with an injected InferenceService.

        Args:
            inference_service: A fully-loaded :class:`InferenceService` instance.
        """
        self._inference_service = inference_service

    async def process_question(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat question through the RAG pipeline.

        Args:
            request: The validated :class:`ChatRequest` from the route handler.

        Returns:
            A :class:`ChatResponse` containing the generated answer,
            retrieved context documents, and timing metadata.

        Raises:
            HTTPException 400: If the question fails validation (empty, too
                long, or only contains injection patterns).
            HTTPException 500: If the inference pipeline encounters an
                unexpected error.
        """
        try:
            result = self._inference_service.answer(request.question)

            context_docs = [
                ContextDocument(
                    index=doc["index"],
                    content=doc["content"],
                    score=doc["score"],
                )
                for doc in result["context_used"]
            ]

            return ChatResponse(
                answer=result["answer"],
                context_used=context_docs,
                inference_time=result["inference_time"],
                retrieval_time=result["retrieval_time"],
                generation_time=result["generation_time"],
                embedding_time=result["embedding_time"],
                documents_retrieved=result["documents_retrieved"],
            )

        except ValueError as exc:
            logger.log_error(f"Chat request validation error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        except TimeoutError as exc:
            logger.log_error(f"Inference timeout: {exc}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    "Inference timed out. The model is running on CPU which is slow. "
                    "Try a shorter question or use a GPU-enabled deployment."
                ),
            ) from exc

        except RuntimeError as exc:
            logger.log_error(f"Inference runtime error: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error during inference: {exc}",
            ) from exc

        except Exception as exc:
            logger.log_error(
                f"Unexpected error in ChatController: {exc}", exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred. Please try again.",
            ) from exc
