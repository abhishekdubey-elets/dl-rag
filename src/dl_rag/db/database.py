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
            self._engine = create_async_engine(
                self._dsn,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                future=True,
            )
            logger.debug("db.engine.created")
        return self._engine

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
        """Create the pg_trgm extension and every table (idempotent)."""
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            await conn.run_sync(Base.metadata.create_all)
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
