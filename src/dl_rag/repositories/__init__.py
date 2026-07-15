"""Repository layer: thin, async data-access objects over the ORM.

Each repository wraps a single :class:`~sqlalchemy.ext.asyncio.AsyncSession`
and translates ORM rows to/from the domain models. Repositories never open or
close sessions themselves — the caller owns the unit of work.
"""

from __future__ import annotations

from dl_rag.repositories.base import BaseRepository
from dl_rag.repositories.chunk_repository import ChunkRepository
from dl_rag.repositories.document_repository import DocumentRepository
from dl_rag.repositories.feedback_repository import (
    CrawlJobRepository,
    FeedbackRepository,
    QueryLogRepository,
)
from dl_rag.repositories.kg_repository import KGRepository

__all__ = [
    "BaseRepository",
    "ChunkRepository",
    "CrawlJobRepository",
    "DocumentRepository",
    "FeedbackRepository",
    "KGRepository",
    "QueryLogRepository",
]
