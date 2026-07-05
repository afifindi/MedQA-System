"""
Medical QA System - Logger Service.

Centralised logging configuration built on top of Loguru.  Provides four
specialised log sinks:

  * ``startup.log``  – application lifecycle events (boot, shutdown, config)
  * ``request.log``  – per-request audit trail (question, timings)
  * ``error.log``    – ERROR and CRITICAL messages only
  * ``model.log``    – ML model loading and inference timing events

All sinks rotate at 10 MB and are retained for 30 days.  The console sink
uses coloured output for human-readable development experience.

Usage::

    from app.services.logger_service import get_logger

    logger = get_logger()
    logger.log_startup("Application started successfully.")
    logger.log_request("User asked a question", question="...", time=0.42)
    logger.log_model("Gemma model loaded in 12.3 s")
    logger.log_error("FAISS index not found", exc_info=True)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger


class LoggerService:
    """
    Singleton service that configures and exposes categorised logging methods.

    The service wraps Loguru's ``logger`` object and adds four specialised
    sinks so that different concerns (startup, requests, errors, model events)
    are separated into dedicated log files.

    Attributes:
        _configured: Class-level flag to prevent double-initialisation.
        _log_dir:    Path to the logging directory.
    """

    _configured: bool = False

    def __init__(self) -> None:
        """Initialise the LoggerService without configuring sinks."""
        self._log_dir: Path = Path("logs")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, log_dir: str = "logs", log_level: str = "INFO") -> None:
        """
        Configure Loguru sinks.

        Safe to call multiple times; only the first call has any effect
        (guarded by ``_configured`` class-level flag).

        Args:
            log_dir:   Directory path where log files will be written.
            log_level: Minimum log level for the console and startup sinks.
        """
        if LoggerService._configured:
            return

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Remove Loguru's default stderr handler
        _loguru_logger.remove()

        fmt_console = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        fmt_file = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )

        rotation = "10 MB"
        retention = "30 days"

        # Console sink
        _loguru_logger.add(
            sys.stderr,
            format=fmt_console,
            level=log_level,
            colorize=True,
            enqueue=True,
        )

        # startup.log – all levels, DEBUG+
        _loguru_logger.add(
            str(self._log_dir / "startup.log"),
            format=fmt_file,
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            enqueue=True,
        )

        # request.log – INFO+ filtered to request messages
        _loguru_logger.add(
            str(self._log_dir / "request.log"),
            format=fmt_file,
            level="INFO",
            rotation=rotation,
            retention=retention,
            filter=lambda record: "request" in record["extra"],
            enqueue=True,
        )

        # error.log – ERROR and above
        _loguru_logger.add(
            str(self._log_dir / "error.log"),
            format=fmt_file,
            level="ERROR",
            rotation=rotation,
            retention=retention,
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

        # model.log – DEBUG+ filtered to model messages
        _loguru_logger.add(
            str(self._log_dir / "model.log"),
            format=fmt_file,
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            filter=lambda record: "model" in record["extra"],
            enqueue=True,
        )

        LoggerService._configured = True

    def log_startup(self, message: str, **kwargs: Any) -> None:
        """
        Log an application startup / lifecycle event.

        Args:
            message: Human-readable log message.
            **kwargs: Additional structured fields merged into the log record.
        """
        _loguru_logger.opt(depth=1).info(message, **kwargs)

    def log_request(self, message: str, **kwargs: Any) -> None:
        """
        Log an incoming API request event.

        Writes to both the console and ``request.log``.

        Args:
            message: Human-readable log message.
            **kwargs: Structured fields (e.g. question, inference_time).
        """
        _loguru_logger.opt(depth=1).bind(request=True).info(message, **kwargs)

    def log_model(self, message: str, **kwargs: Any) -> None:
        """
        Log a model-related event (load, download, inference timing).

        Writes to both the console and ``model.log``.

        Args:
            message: Human-readable log message.
            **kwargs: Structured fields (e.g. model_id, load_time).
        """
        _loguru_logger.opt(depth=1).bind(model=True).info(message, **kwargs)

    def log_error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """
        Log an error with optional exception traceback.

        Writes to both the console and ``error.log``.

        Args:
            message:  Human-readable error description.
            exc_info: If True, include the current exception traceback.
            **kwargs: Additional structured fields.
        """
        if exc_info:
            _loguru_logger.opt(depth=1, exception=True).error(message, **kwargs)
        else:
            _loguru_logger.opt(depth=1).error(message, **kwargs)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_logger_service = LoggerService()


def get_logger() -> LoggerService:
    """
    Return the module-level LoggerService singleton.

    Returns:
        LoggerService: The configured logger service instance.
    """
    return _logger_service
