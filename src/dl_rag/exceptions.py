"""Application exception hierarchy.

Every raised error is a subclass of :class:`DLRagError` so the API error
handler can translate it to a consistent :class:`~dl_rag.models.api.ErrorResponse`.
"""

from __future__ import annotations


class DLRagError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigurationError(DLRagError):
    status_code = 500
    code = "configuration_error"


class AuthenticationError(DLRagError):
    status_code = 401
    code = "authentication_error"


class RateLimitError(DLRagError):
    status_code = 429
    code = "rate_limited"


class NotFoundError(DLRagError):
    status_code = 404
    code = "not_found"


class ValidationError(DLRagError):
    status_code = 422
    code = "validation_error"


class RetrievalError(DLRagError):
    status_code = 503
    code = "retrieval_error"


class GenerationError(DLRagError):
    status_code = 503
    code = "generation_error"


class IngestionError(DLRagError):
    status_code = 500
    code = "ingestion_error"


class DependencyUnavailableError(DLRagError):
    """A downstream dependency (DB/Redis/Qdrant/LLM) is unreachable."""

    status_code = 503
    code = "dependency_unavailable"
