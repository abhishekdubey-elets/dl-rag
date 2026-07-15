"""Indexing endpoints: POST /api/index, POST /api/reindex, GET /api/index/{job_id}."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from dl_rag.api.deps import get_ingestion_service
from dl_rag.api.security import require_api_key
from dl_rag.exceptions import NotFoundError
from dl_rag.models.api import (
    IndexRequest,
    IndexResponse,
    JobStatusResponse,
)
from dl_rag.services.ingestion_service import IngestionService

router = APIRouter(tags=["index"], dependencies=[Depends(require_api_key)])


@router.post("/index", response_model=IndexResponse, status_code=202)
async def index(
    payload: IndexRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IndexResponse:
    return await service.start_job(payload)


@router.post("/reindex", response_model=IndexResponse, status_code=202)
async def reindex(
    payload: IndexRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IndexResponse:
    payload.reindex = True
    return await service.start_job(payload)


@router.get("/index/{job_id}", response_model=JobStatusResponse)
async def job_status(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> JobStatusResponse:
    status = await service.get_job(job_id)
    if status is None:
        raise NotFoundError(f"No ingestion job with id '{job_id}'.")
    return status
