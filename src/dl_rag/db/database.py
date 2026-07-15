"""Async engine + session lifecycle management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from dl_rag.db.orm import Base
from dl_rag.logging_config import get_logger

logger = get_logger(__name__)


class Database:
    """Owns the async engine and hands out sessions.

    The engine and sessionmaker are created lazily on first access so that
    constructing a :class:`Database` never touches the network.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    # ------------------------------------------------------------------ #
    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            kwargs: dict = dict(
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                future=True,
            )
            if self._is_pooled_dsn(self._dsn):
                # Behind PgBouncer/Supavisor (e.g. Supabase pooler): named
                # prepared statements break across pooled connections, and each
                # client connection pins a scarce upstream slot — disable the
                # asyncpg statement caches and keep the pool modest.
                kwargs.update(
                    pool_size=5,
                    max_overflow=5,
                    connect_args={
                        "statement_cache_size": 0,
                        "prepared_statement_cache_size": 0,
                    },
                )
            self._engine = create_async_engine(self._dsn, **kwargs)
            logger.debug("db.engine.created", pooled=self._is_pooled_dsn(self._dsn))
        return self._engine

    @staticmethod
    def _is_pooled_dsn(dsn: str) -> bool:
        return "pooler.supabase.com" in dsn or "pgbouncer" in dsn.lower()

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        if self._sessionmaker is None:
            self._sessionmaker = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._sessionmaker

    # ------------------------------------------------------------------ #
    async def create_all(self) -> None:
        """Create extensions and every table (idempotent).

        The ``chunks.embedding`` pgvector column is provisioned only where the
        ``vector`` extension is available (e.g. Supabase); on plain Postgres
        images without pgvector the app runs fine without it — Qdrant remains
        the retrieval store either way, pgvector is the durable copy.
        """
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            await conn.run_sync(Base.metadata.create_all)
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.execute(text(
                    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(384)"
                ))
            logger.info("db.pgvector.ready")
        except Exception as exc:  # noqa: BLE001 - pgvector genuinely optional
            logger.info("db.pgvector.unavailable", error=str(exc)[:120])
        logger.info("db.schema.ready")

    async def healthcheck(self) -> bool:
        """Return ``True`` if a trivial query succeeds; never raise."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # noqa: BLE001 - health probe must not raise
            logger.warning("db.healthcheck.failed", error=str(exc))
            return False

    async def dispose(self) -> None:
        """Dispose the engine and drop cached handles."""
        if self._engine is not None:
            await self._engine.dispose()
            logger.debug("db.engine.disposed")
        self._engine = None
        self._sessionmaker = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session, committing on success and rolling back on error."""
        session = self.sessionmaker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
