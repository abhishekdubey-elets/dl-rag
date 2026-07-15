"""Health + Prometheus metrics (mounted at the app root, no auth)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from dl_rag import __version__
from dl_rag.api.deps import get_container
from dl_rag.models.api import HealthResponse
from dl_rag.observability.metrics import REGISTRY

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    container = get_container(request)
    checks: dict[str, str] = {}

    checks["postgres"] = "ok" if await container.db.healthcheck() else "down"
    checks["redis"] = "ok" if await container.cache.healthcheck() else "down"
    try:
        # Use a real ping — count() intentionally swallows errors (returns 0),
        # which would mask an unreachable Qdrant in a readiness probe.
        await container.vector_store.client.get_collections()
        checks["qdrant"] = "ok"
    except Exception:  # noqa: BLE001
        checks["qdrant"] = "down"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return HealthResponse(status=overall, version=__version__, checks=checks)


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
