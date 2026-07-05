"""
Medical QA System - Chat Route.

Defines POST /api/chat endpoint.  Delegates all business logic to
:class:`ChatController` and retrieves services from ``app.state``.
"""


from fastapi import APIRouter, Request, Response

from app.api.controllers.chat_controller import ChatController
from app.core.middleware.rate_limit import limiter
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Answer a medical question",
    description=(
        "Submit a medical question and receive an AI-generated answer "
        "grounded in the retrieved medical knowledge base documents.  "
        "The response includes the generated answer, the retrieved context "
        "documents, and detailed timing information."
    ),
    responses={
        200: {"description": "Successful answer with context and timing."},
        400: {"description": "Invalid question (empty, too long, injection detected)."},
        429: {"description": "Rate limit exceeded."},
        500: {"description": "Internal inference error."},
        504: {"description": "Inference timeout (CPU is very slow)."},
    },
)
@limiter.limit("30/minute")
async def chat(request: Request, response: Response, body: ChatRequest) -> ChatResponse:
    """
    POST /api/chat

    Process a medical question through the RAG pipeline.

    Args:
        request: The raw Starlette request (required by SlowAPI for rate limiting).
        response: The FastAPI response object (required by SlowAPI).
        body:    The validated :class:`ChatRequest` body.

    Returns:
        A :class:`ChatResponse` with the generated answer and metadata.
    """
    inference_service = request.app.state.inference_service
    controller = ChatController(inference_service)
    return await controller.process_question(body)
