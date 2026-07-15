"""Knowledge-graph adapter: KG protocol over short-lived Postgres sessions."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dl_rag.models.domain import Entity, Relation
from dl_rag.repositories.kg_repository import KGRepository


class PostgresKnowledgeGraph:
    """Implements ``KnowledgeGraph`` by delegating to :class:`KGRepository`."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def add_entities(self, entities: Sequence[Entity]) -> None:
        async with self._sessionmaker() as session:
            await KGRepository(session).upsert_entities(entities)
            await session.commit()

    async def add_relations(self, relations: Sequence[Relation]) -> None:
        async with self._sessionmaker() as session:
            await KGRepository(session).upsert_relations(relations)
            await session.commit()

    async def expand(self, entity_names: Sequence[str], hops: int = 1) -> list[str]:
        async with self._sessionmaker() as session:
            return await KGRepository(session).expand(entity_names, hops)

    async def neighbors(self, entity_name: str) -> list[Relation]:
        async with self._sessionmaker() as session:
            return await KGRepository(session).neighbors(entity_name)
