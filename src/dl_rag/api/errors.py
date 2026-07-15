"""Centralised exception handling → consistent ``ErrorResponse`` payloads."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from dl_rag.exceptions import DLRagError
from dl_rag.logging_config import get_logger
from dl_rag.models.api import ErrorResponse

logger = get_logger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DLRagError)
    async def _handle_dlrag(request: Request, exc: DLRagError) -> ORJSONResponse:
        logger.warning(
            "request.error",
            code=exc.code,
            status=exc.status_code,
            message=exc.message,
            request_id=_request_id(request),
        )
        body = ErrorResponse(
            error=exc.code,
            detail=exc.detail or exc.message,
            request_id=_request_id(request),
        )
        return ORJSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        request: Request, exc: RequestValidationError
    ) -> ORJSONResponse:
        body = ErrorResponse(
            error="validation_error",
            detail="Request payload failed validation.",
            request_id=_request_id(request),
            extra={"errors": exc.errors()},
        )
        return ORJSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(
        request: Request, exc: StarletteHTTPException
    ) -> ORJSONResponse:
        body = ErrorResponse(
            error="http_error",
            detail=str(exc.detail),
            request_id=_request_id(request),
        )
        return ORJSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> ORJSONResponse:
        logger.error(
            "request.unhandled",
            error=str(exc),
            request_id=_request_id(request),
            exc_info=exc,
        )
        body = ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred.",
            request_id=_request_id(request),
        )
        return ORJSONResponse(status_code=500, content=body.model_dump())
