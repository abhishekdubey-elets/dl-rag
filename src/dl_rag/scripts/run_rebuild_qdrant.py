"""`dl-rebuild-qdrant` — repopulate the Qdrant collection from Postgres.

Embeddings are durably stored in the application database (``chunks.embedding``,
pgvector — e.g. on Supabase). This script streams them into Qdrant, making the
vector store fully disposable: on a fresh deployment, point ``POSTGRES_DSN`` at
the database and run this once instead of shipping Qdrant snapshots.

Examples:
    poetry run dl-rebuild-qdrant                 # rebuild everything
    poetry run dl-rebuild-qdrant --recreate      # drop + recreate the collection first
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from sqlalchemy import text as sqltext

from dl_rag.api.deps import build_container
from dl_rag.config import get_settings
from dl_rag.logging_config import configure_logging, get_logger
from dl_rag.models.domain import Chunk, ChunkMetadata
from dl_rag.models.enums import ContentType

logger = get_logger(__name__)

_BATCH = 500


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild Qdrant from pgvector embeddings.")
    p.add_argument("--recreate", action="store_true",
                   help="Delete the collection before rebuilding.")
    return p.parse_args()


def _row_to_chunk(row) -> Chunk | None:
    if not row.embedding_text:
        return None
    vector = json.loads(row.embedding_text)
    try:
        content_type = ContentType(row.content_type)
    except ValueError:
        content_type = ContentType.OTHER
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        chunk_index=row.chunk_index,
        text=row.text,
        token_count=row.token_count or 0,
        metadata=ChunkMetadata(
            url=row.url,
            title=row.title,
            category=row.category,
            content_type=content_type,
            year=row.year,
            month=row.month,
            author=row.author,
            issue=row.issue,
            published_date=row.published_date,
            tags=list(row.tags or []),
            entities=list(row.entities or []),
            heading_path=list(row.heading_path or []),
        ),
        embedding=vector,
    )


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    container = build_container(settings)
    vs = container.vector_store

    try:
        async with container.db.session() as session:
            total = (await session.execute(sqltext(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
            ))).scalar_one()
        if total == 0:
            raise SystemExit(
                "No embeddings stored in Postgres (chunks.embedding is empty). "
                "Run the embedding backfill first."
            )
        print(f"chunks with stored embeddings: {total}")

        if args.recreate:
            try:
                await vs.client.delete_collection(settings.qdrant_collection)
                print("collection dropped")
            except Exception as exc:  # noqa: BLE001
                logger.warning("qdrant.drop_failed", error=str(exc)[:120])

        dimension: int | None = None
        written = 0
        offset = 0
        started = time.perf_counter()

        while True:
            async with container.db.session() as session:
                rows = (await session.execute(sqltext("""
                    SELECT id, document_id, chunk_index, text, token_count, url,
                           title, category, content_type, year, month, author,
                           issue, published_date, tags, entities, heading_path,
                           embedding::text AS embedding_text
                    FROM chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY id
                    OFFSET :off LIMIT :lim
                """), {"off": offset, "lim": _BATCH})).fetchall()
            if not rows:
                break
            offset += len(rows)

            chunks = [c for c in (_row_to_chunk(r) for r in rows) if c is not None]
            if chunks and dimension is None:
                dimension = len(chunks[0].embedding or [])
                await vs.ensure_collection(dimension)
                print(f"collection ready (dimension={dimension})")
            if chunks:
                written += await vs.upsert(chunks)
            if offset % 5000 < _BATCH:
                rate = written / (time.perf_counter() - started)
                print(f"  … {written}/{total} ({rate:.0f}/s)")

        final = await vs.count()
        print(f"\nQdrant rebuild complete: {written} upserted | collection count: {final}")
    finally:
        await container.db.dispose()
        await container.cache.close()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
