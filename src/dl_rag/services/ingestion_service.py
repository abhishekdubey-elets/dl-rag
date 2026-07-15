"""IngestionService — accept and track crawl/index jobs.

Jobs run in-process as background asyncio tasks and record progress in the
``crawl_jobs`` table. For a horizontally-scaled deployment, swap the
``asyncio.create_task`` launch for a durable queue (Arq/Celery/RQ) — the
service boundary is designed for that substitution.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from dl_rag.config import Settings
from dl_rag.db.database import Database
from dl_rag.ingestion.pipeline import IngestionPipeline
from dl_rag.logging_config import get_logger
from dl_rag.models.api import IndexRequest, IndexResponse, JobStatusResponse
from dl_rag.models.enums import JobStatus
from dl_rag.repositories.feedback_repository import CrawlJobRepository

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IngestionService:
    def __init__(self, pipeline: IngestionPipeline, db: Database, settings: Settings) -> None:
        self._pipeline = pipeline
        self._db = db
        self._settings = settings
        self._tasks: set[asyncio.Task] = set()

    async def start_job(self, request: IndexRequest) -> IndexResponse:
        job_id = uuid.uuid4().hex
        async with self._db.session() as session:
            await CrawlJobRepository(session).create(
                job_id, request.model_dump(mode="json")
            )

        task = asyncio.create_task(self._run_job(job_id, request))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        logger.info("ingest.job_accepted", job_id=job_id, urls=len(request.urls),
                    full_crawl=request.full_crawl, reindex=request.reindex)
        return IndexResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            accepted_urls=len(request.urls),
            message="Ingestion job accepted and running in the background.",
        )

    async def get_job(self, job_id: str) -> JobStatusResponse | None:
        async with self._db.session() as session:
            row = await CrawlJobRepository(session).get(job_id)
        if not row:
            return None
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus(row.get("status", "pending")),
            pages_crawled=int(row.get("pages_crawled", 0) or 0),
            pages_indexed=int(row.get("pages_indexed", 0) or 0),
            pages_failed=int(row.get("pages_failed", 0) or 0),
            chunks_created=int(row.get("chunks_created", 0) or 0),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            error=row.get("error"),
        )

    # ------------------------------------------------------------------ #
    async def _update(self, job_id: str, **fields: object) -> None:
        async with self._db.session() as session:
            await CrawlJobRepository(session).update(job_id, **fields)

    async def _run_job(self, job_id: str, request: IndexRequest) -> None:
        await self._update(job_id, status=JobStatus.RUNNING.value, started_at=_utcnow())

        async def on_progress(stats: dict) -> None:
            # Throttle DB writes to roughly every 10 crawled pages.
            if int(stats.get("pages_crawled", 0)) % 10 != 0:
                return
            await self._update(
                job_id,
                pages_crawled=int(stats.get("pages_crawled", 0)),
                pages_indexed=int(stats.get("pages_indexed", 0)),
                pages_failed=int(stats.get("pages_failed", 0)),
                chunks_created=int(stats.get("chunks_created", 0)),
            )

        try:
            stats = await self._pipeline.run(
                urls=request.urls or None,
                content_types=request.content_types or None,
                since_date=request.since_date,
                max_pages=request.max_pages,
                full_crawl=request.full_crawl,
                on_progress=on_progress,
            )
            await self._update(
                job_id,
                status=JobStatus.COMPLETED.value,
                finished_at=_utcnow(),
                pages_crawled=int(stats.get("pages_crawled", 0)),
                pages_indexed=int(stats.get("pages_indexed", 0)),
                pages_failed=int(stats.get("pages_failed", 0)),
                chunks_created=int(stats.get("chunks_created", 0)),
            )
            logger.info("ingest.job_completed", job_id=job_id, **dict(stats))
        except Exception as exc:  # noqa: BLE001 - persist the failure for the dashboard
            logger.error("ingest.job_failed", job_id=job_id, error=str(exc), exc_info=exc)
            await self._update(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=_utcnow(),
                error=str(exc)[:2000],
            )
