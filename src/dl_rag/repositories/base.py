"""Common base for all repositories."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """Holds the active :class:`AsyncSession` shared by all repositories."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
