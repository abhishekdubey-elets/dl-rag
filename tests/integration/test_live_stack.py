"""Integration tests — require live Postgres/Redis/Qdrant (docker compose up).

Each test skips itself when its service isn't reachable, so the suite is safe
to run anywhere; CI brings the services up and exercises them for real.
"""

from __future__ import annotations

import socket
import uuid
from datetime import date

import pytest

from dl_rag.config import Settings
from dl_rag.models.domain import SourceDocument

from tests.conftest import make_chunk

pytestmark = pytest.mark.integration


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    return Settings()  # honours env / .env for service DSNs


# --------------------------------------------------------------------------- #
class TestPostgres:
    @pytest.fixture
    async def db(self, live_settings):
        if not _reachable("localhost", 5432):
            pytest.skip("postgres not reachable")
        from dl_rag.db.database import Database

        db = Database(live_settings.postgres_dsn)
        await db.create_all()
        yield db
        await db.dispose()

    async def test_document_roundtrip_and_fts(self, db):
        from dl_rag.repositories.chunk_repository import ChunkRepository
        from dl_rag.repositories.document_repository import DocumentRepository

        doc_id = f"it-{uuid.uuid4().hex[:12]}"
        doc = SourceDocument(
            id=doc_id,
            url=f"https://example.com/{doc_id}/",
            title="NEP implementation in Karnataka",
            published_date=date(2022, 6, 14),
            content_markdown="Karnataka led NEP implementation with competency-based learning.",
        )
        chunk = make_chunk(0, doc_id=doc_id, url=doc.url, title=doc.title,
                           text=doc.content_markdown)

        async with db.session() as session:
            await DocumentRepository(session).upsert(doc)
            await ChunkRepository(session).bulk_upsert([chunk])

        async with db.session() as session:
            repo = DocumentRepository(session)
            got = await repo.get(doc_id)
            assert got is not None and got.title == doc.title

            hits = await ChunkRepository(session).fts_search("NEP Karnataka", top_k=5)
            assert any(rc.chunk.document_id == doc_id for rc in hits)

        async with db.session() as session:
            await DocumentRepository(session).delete(doc_id)


# --------------------------------------------------------------------------- #
class TestRedis:
    async def test_cache_roundtrip_and_window(self, live_settings):
        if not _reachable("localhost", 6379):
            pytest.skip("redis not reachable")
        from dl_rag.cache.redis_cache import RedisCache

        cache = RedisCache(live_settings.redis_url)
        key = f"it:{uuid.uuid4().hex}"
        try:
            await cache.set_json(key, {"a": 1}, ttl=30)
            assert await cache.get_json(key) == {"a": 1}
            wkey = f"{key}:win"
            assert await cache.incr_window(wkey, 30) == 1
            assert await cache.incr_window(wkey, 30) == 2
            await cache.delete(key)
            assert await cache.get_json(key) is None
        finally:
            await cache.close()


# --------------------------------------------------------------------------- #
class TestQdrant:
    async def test_vector_roundtrip(self, live_settings):
        if not _reachable("localhost", 6333):
            pytest.skip("qdrant not reachable")
        from dl_rag.vectorstore.qdrant_store import QdrantVectorStore

        collection = f"it_{uuid.uuid4().hex[:8]}"
        store = QdrantVectorStore(live_settings.qdrant_url, collection)
        await store.ensure_collection(dimension=8)

        chunk = make_chunk(0, doc_id="it-vec")
        chunk.embedding = [0.1] * 8
        assert await store.upsert([chunk]) == 1

        hits = await store.search([0.1] * 8, top_k=1)
        assert hits and hits[0].chunk.document_id == "it-vec"

        await store.delete_by_document("it-vec")
