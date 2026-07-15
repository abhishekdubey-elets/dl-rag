"""IngestionPipeline — the offline crawl → chunk → embed → index orchestration.

Wires the stateless ingestion components (crawler / chunker / entity extractor)
to the persistence adapters (Postgres, Qdrant, knowledge graph). Idempotent:
re-ingesting a URL replaces its chunks and vectors (content-hash de-dup upstream).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import date

from dl_rag.config import Settings
from dl_rag.db.database import Database
from dl_rag.ingestion.chunking.semantic_chunker import SemanticChunker
from dl_rag.ingestion.crawler.wordpress import WordPressCrawler
from dl_rag.ingestion.entities.extractor import EntityExtractor
from dl_rag.knowledge_graph.builder import KnowledgeGraphBuilder
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import SourceDocument
from dl_rag.models.enums import ContentType
from dl_rag.observability import metrics
from dl_rag.protocols import Embedder, KnowledgeGraph, VectorStore
from dl_rag.repositories.chunk_repository import ChunkRepository
from dl_rag.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)


class IngestionStats(dict):
    """A plain dict with typed convenience accessors for job progress."""

    def bump(self, key: str, by: int = 1) -> None:
        self[key] = int(self.get(key, 0)) + by


class IngestionPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        vector_store: VectorStore,
        embedder: Embedder,
        knowledge_graph: KnowledgeGraph,
        crawler: WordPressCrawler,
        chunker: SemanticChunker,
        entity_extractor: EntityExtractor,
    ) -> None:
        self._settings = settings
        self._db = db
        self._vs = vector_store
        self._embedder = embedder
        self._kg_builder = KnowledgeGraphBuilder(knowledge_graph)
        self._crawler = crawler
        self._chunker = chunker
        self._entities = entity_extractor
        self._collection_ready = False
        # None = unknown (probe on first use); True/False = column present or not.
        self._pg_embeddings: bool | None = None

    # ------------------------------------------------------------------ #
    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        try:
            await self._vs.ensure_collection(self._embedder.dimension)
            self._collection_ready = True
        except Exception as exc:  # noqa: BLE001 - surfaced, not fatal to the loop
            logger.error("ingest.ensure_collection_failed", error=str(exc))
            raise

    async def _store_pg_embeddings(self, chunks: Sequence) -> None:
        """Mirror chunk embeddings into ``chunks.embedding`` (pgvector) when present.

        Keeps Postgres a complete, durable copy of the index so Qdrant can be
        rebuilt from it (``dl-rebuild-qdrant``). Silently skipped on databases
        without the column (e.g. local postgres images lacking pgvector).
        """
        if self._pg_embeddings is False:
            return
        rows = [
            {
                "id": c.id,
                "emb": "[" + ",".join(f"{v:.7f}" for v in c.embedding) + "]",
            }
            for c in chunks
            if c.embedding
        ]
        if not rows:
            return
        from sqlalchemy import text as sqltext

        try:
            async with self._db.session() as session:
                if self._pg_embeddings is None:
                    present = (await session.execute(sqltext(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name='chunks' AND column_name='embedding'"
                    ))).first() is not None
                    self._pg_embeddings = present
                    if not present:
                        logger.info("ingest.pg_embeddings.disabled")
                        return
                await session.execute(
                    sqltext("UPDATE chunks SET embedding = CAST(:emb AS vector) "
                            "WHERE id = :id"),
                    rows,
                )
        except Exception as exc:  # noqa: BLE001 - mirror is best-effort
            logger.warning("ingest.pg_embeddings.failed", error=str(exc)[:150])

    async def ingest_document(self, doc: SourceDocument) -> int:
        """Full per-document pipeline. Returns the number of chunks indexed."""
        # 1. Entities / keywords / relations (document level).
        entities, keywords, relations = self._entities.extract_from_document(doc)
        doc.entities = [e.name for e in entities]
        doc.keywords = keywords
        if doc.content_hash is None:
            doc.content_hash = doc.compute_hash()

        # 2. Semantic chunking.
        chunks = self._chunker.chunk_document(doc)
        if not chunks:
            logger.info("ingest.no_chunks", url=doc.url)
            # Still persist the document row so it is discoverable / not re-crawled.
            async with self._db.session() as session:
                await DocumentRepository(session).upsert(doc)
            return 0

        # 3. Embed all chunk texts in one batched call.
        vectors = await self._embedder.embed_documents([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors, strict=False):
            chunk.embedding = vector

        # 4. Persist document + chunks (Postgres), replacing any prior chunks.
        async with self._db.session() as session:
            await DocumentRepository(session).upsert(doc)
            chunk_repo = ChunkRepository(session)
            await chunk_repo.delete_by_document(doc.id)
            await chunk_repo.bulk_upsert(chunks)
        await self._store_pg_embeddings(chunks)

        # 5. Vectors (Qdrant), replacing any prior points for this document.
        await self._ensure_collection()
        await self._vs.delete_by_document(doc.id)
        await self._vs.upsert(chunks)

        # 6. Knowledge graph.
        if entities or relations:
            await self._kg_builder.add(entities, relations)

        metrics.INGESTED_DOCS.inc()
        logger.info("ingest.document", url=doc.url, chunks=len(chunks),
                    entities=len(entities), relations=len(relations))
        return len(chunks)

    # ------------------------------------------------------------------ #
    async def run(
        self,
        *,
        urls: Sequence[str] | None = None,
        content_types: list[ContentType] | None = None,
        since_date: date | None = None,
        max_pages: int | None = None,
        full_crawl: bool = False,
        on_progress: Callable[[dict], Awaitable[None]] | None = None,
    ) -> IngestionStats:
        """Discover (or take) URLs, crawl, and ingest each document."""
        stats = IngestionStats()

        if urls:
            target_urls = list(urls)
        else:
            target_urls = await self._crawler.discover_urls(
                content_types=content_types,
                since_date=since_date,
                max_pages=max_pages,
            )
        logger.info("ingest.discovered", count=len(target_urls), full_crawl=full_crawl)
        stats["discovered"] = len(target_urls)

        allowed_types = set(content_types) if content_types else None

        async for doc in self._crawler.crawl(target_urls):
            stats.bump("pages_crawled")
            if allowed_types and doc.content_type not in allowed_types:
                stats.bump("pages_skipped")
                continue
            try:
                n_chunks = await self.ingest_document(doc)
                stats.bump("pages_indexed")
                stats.bump("chunks_created", n_chunks)
            except Exception as exc:  # noqa: BLE001 - one bad page must not abort the crawl
                stats.bump("pages_failed")
                logger.error("ingest.page_failed", url=doc.url, error=str(exc))
            if on_progress is not None:
                try:
                    await on_progress(dict(stats))
                except Exception:  # noqa: BLE001
                    pass

        logger.info("ingest.completed", **{k: stats[k] for k in stats})
        return stats
