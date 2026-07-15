"""`dl-import-supabase` — merge externally-stored video transcripts into the index.

Source: the user's Supabase Postgres, table ``public.chunks`` with
``source='youtube'`` rows — timestamped transcript segments carrying
``metadata.video_id`` / ``metadata.title`` / ``metadata.timestamp_s``.

For every segment group whose ``video_id`` matches an already-indexed video
document, the segments are stitched (by timestamp) into a full transcript,
merged into the document body under ``## Transcript``, and the document is
re-ingested through the standard pipeline (re-chunk → re-embed → re-index →
KG refresh). Their embeddings are NOT imported — they come from a different
model/dimension than this index; the text is the portable asset.

Examples:
    poetry run dl-import-supabase --limit 5        # trial
    poetry run dl-import-supabase                  # all matched videos
    poetry run dl-import-supabase --overwrite      # also refresh docs that already have transcripts
"""

from __future__ import annotations

import argparse
import asyncio
import time

import asyncpg
from sqlalchemy import text as sqltext

from dl_rag.api.deps import build_container
from dl_rag.config import get_settings
from dl_rag.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

_TRANSCRIPT_HEADING = "## Transcript"
_MAX_TRANSCRIPT_CHARS = 200_000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import video transcripts from Supabase.")
    p.add_argument("--limit", type=int, default=None, help="Only process N videos (trial).")
    p.add_argument("--overwrite", action="store_true",
                   help="Also refresh documents that already contain a transcript.")
    return p.parse_args()


async def _supabase_connect(settings) -> asyncpg.Connection:
    if not settings.supabase_db_host or not settings.supabase_db_user:
        raise SystemExit("SUPABASE_DB_* settings are not configured in .env")
    last: Exception | None = None
    for attempt in range(4):
        try:
            return await asyncpg.connect(
                host=settings.supabase_db_host,
                port=settings.supabase_db_port,
                database=settings.supabase_db_name,
                user=settings.supabase_db_user,
                password=settings.supabase_db_password,
                timeout=30,
                statement_cache_size=0,  # required behind Supabase's pooler
            )
        except Exception as exc:  # noqa: BLE001 - pooler auth hiccups are transient
            last = exc
            logger.warning("supabase.connect_retry", attempt=attempt + 1,
                           error=str(exc)[:120])
            await asyncio.sleep(3 * (attempt + 1))
    raise SystemExit(f"Could not connect to Supabase: {last}")


def _merge_transcript(content_markdown: str, transcript: str) -> str:
    """Replace or append the ``## Transcript`` section of a document body."""
    base = content_markdown or ""
    idx = base.find(_TRANSCRIPT_HEADING)
    if idx != -1:
        base = base[:idx].rstrip()
    transcript = transcript[:_MAX_TRANSCRIPT_CHARS]
    return f"{base}\n\n{_TRANSCRIPT_HEADING}\n\n{transcript}".strip()


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    container = build_container(settings)
    supabase = await _supabase_connect(settings)

    try:
        # --- our side: indexed videos and their transcript state -------------
        async with container.db.session() as session:
            rows = (await session.execute(sqltext("""
                SELECT id, url, metadata_json->>'video_id' AS video_id,
                       (content_markdown LIKE :h) AS has_transcript
                FROM documents
                WHERE content_type = 'video' AND metadata_json->>'video_id' IS NOT NULL
            """), {"h": f"%{_TRANSCRIPT_HEADING}%"})).fetchall()
        ours = {r.video_id: {"id": r.id, "has": bool(r.has_transcript)} for r in rows}

        # --- their side: which of our videos have transcript segments --------
        theirs = {
            r["vid"] for r in await supabase.fetch(
                "SELECT DISTINCT metadata->>'video_id' AS vid FROM public.chunks "
                "WHERE source='youtube' AND metadata->>'video_id' IS NOT NULL"
            )
        }
        targets = [
            vid for vid, info in ours.items()
            if vid in theirs and (args.overwrite or not info["has"])
        ]
        if args.limit:
            targets = targets[: args.limit]
        print(f"indexed videos: {len(ours)} | matched in Supabase: "
              f"{len(set(ours) & theirs)} | to import now: {len(targets)}")

        stats = {"updated": 0, "empty": 0, "failed": 0, "chunks": 0}
        started = time.perf_counter()

        for i, video_id in enumerate(targets, start=1):
            try:
                segments = await supabase.fetch("""
                    SELECT text, COALESCE((metadata->>'timestamp_s')::float, 0) AS ts
                    FROM public.chunks
                    WHERE source='youtube' AND metadata->>'video_id' = $1
                    ORDER BY ts
                """, video_id)
                transcript = " ".join(
                    (r["text"] or "").strip() for r in segments if r["text"]
                ).strip()
                if not transcript:
                    stats["empty"] += 1
                    continue

                doc_id = ours[video_id]["id"]
                async with container.db.session() as session:
                    from dl_rag.repositories.document_repository import (
                        DocumentRepository,
                    )
                    doc = await DocumentRepository(session).get(doc_id)
                if doc is None:
                    stats["failed"] += 1
                    continue
                doc.content_markdown = _merge_transcript(
                    doc.content_markdown, transcript
                )
                doc.metadata["has_transcript"] = True
                doc.metadata["transcript_source"] = "supabase"
                doc.content_hash = doc.compute_hash()

                n_chunks = await container.pipeline.ingest_document(doc)
                stats["updated"] += 1
                stats["chunks"] += n_chunks
            except Exception as exc:  # noqa: BLE001 - continue past bad videos
                stats["failed"] += 1
                logger.error("supabase.import_failed", video=video_id,
                             error=str(exc)[:200])

            if i % 20 == 0:
                rate = i / (time.perf_counter() - started)
                eta = (len(targets) - i) / rate / 60 if rate else 0
                print(f"  … {i}/{len(targets)} (updated={stats['updated']}, "
                      f"rate={rate:.1f}/s, eta={eta:.0f}m)")

        print("\nSupabase transcript import complete:")
        for key, value in stats.items():
            print(f"  {key:10s}: {value}")
    finally:
        await supabase.close()
        await container.db.dispose()
        await container.cache.close()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
