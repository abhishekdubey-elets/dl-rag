"""Structured logging via structlog.

Call :func:`configure_logging` once at startup. Everywhere else use
``logger = structlog.get_logger(__name__)`` and log with keyword context, e.g.
``logger.info("retrieval.done", query_type=qt, candidates=40, latency_ms=120)``.
"""

from __future__ import annotations

import logging
import sys

import structlog

from dl_rag.config import get_settings

_configured = False


def configure_logging() -> None:
    """Configure stdlib + structlog. Idempotent."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        timestamper,
    ]

    if settings.log_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, sqlalchemy, httpx) through the same level.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)  # type: ignore[return-value]
