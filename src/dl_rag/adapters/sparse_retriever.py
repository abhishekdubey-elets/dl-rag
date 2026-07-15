"""Sparse retriever adapter: Postgres full-text search behind a short session."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dl_rag.models.domain import RetrievedChunk
from dl_rag.repositories.chunk_repository import ChunkRepository


class PostgresFTSRetriever:
    """Implements ``SparseRetriever`` by delegating to :class:`ChunkRepository`."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        async with self._sessionmaker() as session:
            return await ChunkRepository(session).fts_search(query, top_k, filters)

    async def latest(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        phrase: str | None = None,
    ) -> list[RetrievedChunk]:
        """Newest on-topic chunks by publication date (see ChunkRepository)."""
        async with self._sessionmaker() as session:
            return await ChunkRepository(session).latest_by_terms(
                query, top_k, filters, phrase=phrase
            )
