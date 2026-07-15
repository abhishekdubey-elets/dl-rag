"""HTTP middleware: request-id + structured-log binding, latency metrics, and a
Redis fixed-window rate limiter.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from dl_rag.config import get_settings
from dl_rag.logging_config import get_logger
from dl_rag.observability import metrics

logger = get_logger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"


def _route_template(request: Request) -> str:
    """Low-cardinality path label for metrics (route pattern, not raw path)."""
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path  # type: ignore[no-any-return]
    return request.url.path


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind it to logs, time the request, emit metrics."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            path = _route_template(request)
            try:
                metrics.REQUEST_COUNT.labels(
                    request.method, path, str(status_code)
                ).inc()
                metrics.REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
            except Exception:  # noqa: BLE001 - metrics must never break a request
                pass
            logger.info(
                "request.completed",
                method=request.method,
                path=path,
                status=status_code,
                duration_ms=round(elapsed * 1000, 1),
            )
            response_headers_id = request_id
            # best-effort echo of the id (response may already be sent on error)
            if "response" in locals():
                response.headers[_REQUEST_ID_HEADER] = response_headers_id
            structlog.contextvars.clear_contextvars()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window limiter keyed by API key (falling back to client IP).

    Uses the cache placed on ``app.state.cache`` (any :class:`Cache`). If the
    cache is unavailable the request is allowed (fail-open) — availability of
    the archive beats strict limiting.
    """

    def __init__(self, app, *, limit: int, window_seconds: int) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._limit = limit
        self._window = window_seconds

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings = get_settings()
        # Only guard the API surface; skip health/metrics/docs.
        if not request.url.path.startswith(settings.api_prefix):
            return await call_next(request)

        cache = getattr(request.app.state, "cache", None)
        if cache is None:
            return await call_next(request)

        identity = request.headers.get("X-API-Key") or (
            request.client.host if request.client else "unknown"
        )
        window_id = int(time.time()) // self._window
        key = f"ratelimit:{identity}:{window_id}"
        try:
            count = await cache.incr_window(key, self._window)
        except Exception:  # noqa: BLE001 - fail open
            return await call_next(request)

        if count > self._limit:
            logger.warning("ratelimit.exceeded", identity=identity, count=count)
            retry_after = self._window - (int(time.time()) % self._window)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": f"Rate limit of {self._limit}/{self._window}s exceeded.",
                    "request_id": getattr(request.state, "request_id", None),
                },
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
