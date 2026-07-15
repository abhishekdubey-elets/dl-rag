"""AdminService — index/health/insight aggregation for the admin dashboard."""

from __future__ import annotations

from dl_rag.db.database import Database
from dl_rag.logging_config import get_logger
from dl_rag.models.api import (
    AdminInsightsResponse,
    AdminStatsResponse,
    PopularQuestion,
)
from dl_rag.protocols import VectorStore
from dl_rag.repositories.chunk_repository import ChunkRepository
from dl_rag.repositories.document_repository import DocumentRepository
from dl_rag.repositories.feedback_repository import (
    CrawlJobRepository,
    FeedbackRepository,
    QueryLogRepository,
)
from dl_rag.repositories.kg_repository import KGRepository

logger = get_logger(__name__)


class AdminService:
    """Read-only aggregation over Postgres + Qdrant for monitoring endpoints."""

    def __init__(self, db: Database, vector_store: VectorStore) -> None:
        self._db = db
        self._vs = vector_store

    async def stats(self) -> AdminStatsResponse:
        async with self._db.session() as session:
            documents = DocumentRepository(session)
            chunks = ChunkRepository(session)
            kg = KGRepository(session)
            jobs = CrawlJobRepository(session)

            total_docs = await documents.count()
            by_type = await documents.count_by_type()
            by_year = await documents.count_by_year()
            total_chunks = await chunks.count()
            n_entities = await kg.count_entities()
            n_relations = await kg.count_relations()
            latest_job = await jobs.latest()

        try:
            vector_points = await self._vs.count()
        except Exception as exc:  # noqa: BLE001 - dashboard must not hard-fail
            logger.warning("admin.vector_count_failed", error=str(exc))
            vector_points = 0

        failed_pages = int(latest_job.get("pages_failed", 0)) if latest_job else 0
        last_crawl_at = None
        if latest_job:
            last_crawl_at = latest_job.get("finished_at") or latest_job.get("started_at")

        return AdminStatsResponse(
            documents_indexed=total_docs,
            chunks_indexed=total_chunks,
            entities=n_entities,
            relations=n_relations,
            documents_by_type={str(k): v for k, v in by_type.items()},
            documents_by_year={str(k): v for k, v in by_year.items()},
            failed_pages=failed_pages,
            last_crawl_at=last_crawl_at,
            vector_points=vector_points,
        )

    async def insights(self) -> AdminInsightsResponse:
        async with self._db.session() as session:
            query_log = QueryLogRepository(session)
            feedback = FeedbackRepository(session)

            popular = await query_log.popular(limit=20)
            avg_latency = await query_log.avg_latency()
            citation_freq = await query_log.citation_frequency(limit=20)
            positive, negative = await feedback.counts()

        return AdminInsightsResponse(
            popular_questions=[
                PopularQuestion(query=q, count=c, avg_confidence=round(conf, 3))
                for (q, c, conf) in popular
            ],
            avg_latency_ms=round(avg_latency, 1),
            citation_frequency=citation_freq,
            feedback_positive=positive,
            feedback_negative=negative,
        )
