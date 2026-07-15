"""Admin dashboard endpoints: GET /api/admin/stats, GET /api/admin/insights."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from dl_rag.api.deps import get_admin_service
from dl_rag.api.security import require_api_key
from dl_rag.models.api import AdminInsightsResponse, AdminStatsResponse
from dl_rag.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_api_key)])


@router.get("/stats", response_model=AdminStatsResponse)
async def stats(service: AdminService = Depends(get_admin_service)) -> AdminStatsResponse:
    return await service.stats()


@router.get("/insights", response_model=AdminInsightsResponse)
async def insights(service: AdminService = Depends(get_admin_service)) -> AdminInsightsResponse:
    return await service.insights()
