"""Async Redis cache implementing the :class:`~dl_rag.protocols.Cache` protocol."""

from __future__ import annotations

from typing import Any

import orjson
from redis.asyncio import Redis

from dl_rag.logging_config import get_logger

logger = get_logger(__name__)


class RedisCache:
    """JSON get/set with TTL plus an atomic fixed-window counter.

    Values are stored as raw ``orjson`` bytes (``decode_responses=False``).
    """

    def __init__(self, redis_url: str) -> None:
        self._redis: Redis = Redis.from_url(redis_url, decode_responses=False)

    @property
    def client(self) -> Redis:
        return self._redis

    async def get_json(self, key: str) -> Any | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:
            logger.warning("cache.get_json.decode_failed", key=key)
            return None

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        data = orjson.dumps(value)
        if ttl is not None and ttl > 0:
            await self._redis.set(key, data, ex=ttl)
        else:
            await self._redis.set(key, data)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def incr_window(self, key: str, window_seconds: int) -> int:
        """Increment a counter, setting the TTL only on the first increment."""
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, window_seconds, nx=True)
            results = await pipe.execute()
        return int(results[0])

    async def healthcheck(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception as exc:  # noqa: BLE001 - health probe must not raise
            logger.warning("cache.healthcheck.failed", error=str(exc))
            return False

    async def close(self) -> None:
        await self._redis.aclose()
