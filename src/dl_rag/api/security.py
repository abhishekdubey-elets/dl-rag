"""API-key authentication.

A single ``X-API-Key`` header checked against the configured key set. Auth is a
no-op when ``REQUIRE_AUTH`` is false (convenient for local dev), and enforced
otherwise. Wire :func:`require_api_key` as a dependency on protected routers.
"""

from __future__ import annotations

from fastapi import Security
from fastapi.security import APIKeyHeader

from dl_rag.config import get_settings
from dl_rag.exceptions import AuthenticationError

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency: validate the ``X-API-Key`` header.

    Returns the authenticated key (or ``"anonymous"`` when auth is disabled).
    """
    settings = get_settings()
    if not settings.require_auth:
        return api_key or "anonymous"

    if not api_key or api_key not in settings.api_key_set:
        raise AuthenticationError(
            "Missing or invalid API key.",
            detail="Provide a valid key in the 'X-API-Key' header.",
        )
    return api_key
