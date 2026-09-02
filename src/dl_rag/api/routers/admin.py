"""Admin endpoints: stats, insights, and auto-ingest control.

GET  /api/admin/stats
GET  /api/admin/insights
GET  /api/admin/auto-ingest            — scheduler state + last run
POST /api/admin/auto-ingest/run        — trigger a run now (``?wait=true`` to block)
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query

from dl_rag.api.deps import get_admin_service, get_auto_ingest_service
from dl_rag.api.security import require_api_key
from dl_rag.models.api import (
    AdminInsightsResponse,
    AdminStatsResponse,
    AutoIngestRunResponse,
    AutoIngestRunSummary,
    AutoIngestStatusResponse,
)
from dl_rag.services.admin_service import AdminService
from dl_rag.services.auto_ingest import AutoIngestService

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_api_key)])

# Keep manual-run tasks referenced so they are not garbage-collected mid-flight.
_background_runs: set[asyncio.Task] = set()


@router.get("/stats", response_model=AdminStatsResponse)
async def stats(service: AdminService = Depends(get_admin_service)) -> AdminStatsResponse:
    return await service.stats()


@router.get("/insights", response_model=AdminInsightsResponse)
async def insights(service: AdminService = Depends(get_admin_service)) -> AdminInsightsResponse:
    return await service.insights()


@router.get("/auto-ingest", response_model=AutoIngestStatusResponse)
async def auto_ingest_status(
    service: AutoIngestService = Depends(get_auto_ingest_service),
) -> AutoIngestStatusResponse:
    return AutoIngestStatusResponse(**await service.status())


@router.post("/auto-ingest/run", response_model=AutoIngestRunResponse, status_code=202)
async def auto_ingest_run(
    wait: bool = Query(
        default=False,
        description="Block until the run finishes and return its summary.",
    ),
    service: AutoIngestService = Depends(get_auto_ingest_service),
) -> AutoIngestRunResponse:
    if wait:
        result = await service.run_once(reason="manual")
        summary = AutoIngestRunSummary(**result)
        if summary.skipped:
            return AutoIngestRunResponse(
                accepted=False, message="A run is already in progress.", run=summary
            )
        return AutoIngestRunResponse(message="Run complete.", run=summary)

    task = asyncio.create_task(service.run_once(reason="manual"))
    _background_runs.add(task)
    task.add_done_callback(_background_runs.discard)
    return AutoIngestRunResponse(
        message="Run started in the background; poll GET /api/admin/auto-ingest."
    )
