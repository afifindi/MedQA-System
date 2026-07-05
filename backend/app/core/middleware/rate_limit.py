"""
Medical QA System - Rate Limiting Middleware.

Uses SlowAPI (a Starlette/FastAPI wrapper around the `limits` library) to
enforce per-IP rate limits on all API endpoints.  The default limit is
60 requests per minute for general endpoints and 30 per minute for the
chat endpoint (configured per-route using the @limiter.limit decorator).
"""

from __future__ import annotations

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Singleton limiter – imported and used as a decorator in route handlers
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    headers_enabled=True,  # Adds X-RateLimit-* response headers
)


def setup_rate_limit(app: FastAPI) -> None:
    """
    Register the SlowAPI limiter and its error handler with the FastAPI app.

    Must be called before the first request is processed.  After calling this
    function, individual routes can be decorated with::

        @limiter.limit("30/minute")
        async def my_endpoint(request: Request): ...

    Args:
        app: The FastAPI application instance.
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def _custom_rate_limit_handler(
    request: Request, exc: RateLimitExceeded
) -> Response:
    """
    Return a structured JSON 429 response instead of the default plain-text one.

    Args:
        request: The incoming HTTP request.
        exc:     The rate-limit-exceeded exception.

    Returns:
        JSONResponse with status 429 and a descriptive error body.
    """
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please slow down your requests.",
            "retry_after": str(exc.retry_after) if hasattr(exc, "retry_after") else "60",
        },
        headers={"Retry-After": "60"},
    )
