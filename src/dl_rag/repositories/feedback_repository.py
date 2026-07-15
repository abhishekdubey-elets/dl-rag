"""Feedback, query-log analytics, and crawl-job bookkeeping repositories."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy import update as sa_update

from dl_rag.db.orm import CrawlJobORM, FeedbackORM, QueryLogORM
from dl_rag.logging_config import get_logger
from dl_rag.models.enums import FeedbackRating, JobStatus
from dl_rag.repositories.base import BaseRepository

logger = get_logger(__name__)


class FeedbackRepository(BaseRepository):
    """Thumbs up/down feedback on generated answers."""

    async def add(
        self,
        conversation_id: str,
        message_id: str,
        rating: str,
        comment: str | None,
        reason: str | None,
    ) -> None:
        self.session.add(
            FeedbackORM(
                conversation_id=conversation_id,
                message_id=message_id,
                rating=rating,
                comment=comment,
                reason=reason,
            )
        )
        await self.session.flush()

    async def counts(self) -> tuple[int, int]:
        """Return ``(positive, negative)`` feedback totals."""
        stmt = select(FeedbackORM.rating, func.count()).group_by(FeedbackORM.rating)
        rows = (await self.session.execute(stmt)).all()
        by_rating = {str(rating): int(count) for rating, count in rows}
        positive = by_rating.get(FeedbackRating.UP.value, 0)
        negative = by_rating.get(FeedbackRating.DOWN.value, 0)
        return positive, negative


class QueryLogRepository(BaseRepository):
    """Structured log of every answered query, for analytics."""

    async def add(
        self,
        *,
        conversation_id: str,
        message_id: str,
        query: str,
        normalized_query: str,
        query_type: str,
        confidence: float,
        retrieved_documents: int,
        latency_ms: int,
        cited_urls: list[str],
    ) -> None:
        self.session.add(
            QueryLogORM(
                conversation_id=conversation_id,
                message_id=message_id,
                query=query,
                normalized_query=normalized_query,
                query_type=query_type,
                confidence=confidence,
                retrieved_documents=retrieved_documents,
                latency_ms=latency_ms,
                cited_urls=list(cited_urls),
            )
        )
        await self.session.flush()

    async def popular(self, limit: int = 20) -> list[tuple[str, int, float]]:
        """Most frequent normalized queries as ``(query, count, avg_confidence)``."""
        stmt = (
            select(
                QueryLogORM.normalized_query,
                func.count().label("c"),
                func.avg(QueryLogORM.confidence).label("avg_conf"),
            )
            .group_by(QueryLogORM.normalized_query)
            .order_by(func.count().desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            (nq, int(count), float(avg) if avg is not None else 0.0)
            for nq, count, avg in rows
        ]

    async def avg_latency(self) -> float:
        stmt = select(func.avg(QueryLogORM.latency_ms))
        value = (await self.session.execute(stmt)).scalar_one_or_none()
        return float(value) if value is not None else 0.0

    async def citation_frequency(self, limit: int = 20) -> dict[str, int]:
        """Count how often each URL appears across all logged citations."""
        stmt = text(
            """
            SELECT url, COUNT(*) AS c
            FROM query_logs,
                 jsonb_array_elements_text(COALESCE(cited_urls, '[]'::jsonb)) AS url
            GROUP BY url
            ORDER BY c DESC
            LIMIT :limit
            """
        )
        rows = (await self.session.execute(stmt, {"limit": limit})).all()
        return {str(url): int(count) for url, count in rows}


class CrawlJobRepository(BaseRepository):
    """Lifecycle tracking for background crawl/index jobs."""

    async def create(self, job_id: str, params: dict[str, Any]) -> None:
        self.session.add(
            CrawlJobORM(
                id=job_id,
                status=JobStatus.PENDING.value,
                params=dict(params),
            )
        )
        await self.session.flush()

    async def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        await self.session.execute(
            sa_update(CrawlJobORM).where(CrawlJobORM.id == job_id).values(**fields)
        )

    async def get(self, job_id: str) -> dict[str, Any] | None:
        row = await self.session.get(CrawlJobORM, job_id)
        return _job_to_dict(row) if row is not None else None

    async def latest(self) -> dict[str, Any] | None:
        stmt = select(CrawlJobORM).order_by(CrawlJobORM.created_at.desc()).limit(1)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        return _job_to_dict(row) if row is not None else None


def _job_to_dict(row: CrawlJobORM) -> dict[str, Any]:
    return {
        "id": row.id,
        "status": row.status,
        "pages_crawled": row.pages_crawled,
        "pages_indexed": row.pages_indexed,
        "pages_failed": row.pages_failed,
        "chunks_created": row.chunks_created,
        "params": dict(row.params or {}),
        "error": row.error,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "created_at": row.created_at,
    }
