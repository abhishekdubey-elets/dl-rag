"""`dl-ingest-youtube` — ingest a YouTube channel's videos into the archive index.

Catalog → (hydrate metadata) → transcript → SourceDocument → the standard
ingestion pipeline (chunk/embed/index/KG). Idempotent per video URL.

Examples:
    poetry run dl-ingest-youtube --max-videos 60
    poetry run dl-ingest-youtube --match "world education summit|wes"
    poetry run dl-ingest-youtube --channel https://www.youtube.com/@SomeChannel
    poetry run dl-ingest-youtube --skip-existing --max-videos 300
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time

from dl_rag.api.deps import build_container
from dl_rag.config import get_settings
from dl_rag.ingestion.youtube.catalog import YouTubeCatalog
from dl_rag.ingestion.youtube.documents import video_to_document
from dl_rag.ingestion.youtube.transcripts import TranscriptFetcher
from dl_rag.logging_config import configure_logging, get_logger
from dl_rag.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)

_POLITENESS_SECONDS = 0.75  # between per-video hydrate/transcript fetches


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest YouTube channel videos.")
    p.add_argument("--channel", default=None, help="Channel URL/@handle (default from env).")
    p.add_argument("--max-videos", type=int, default=None,
                   help="Catalog cap (default YOUTUBE_MAX_VIDEOS).")
    p.add_argument("--match", default=None,
                   help="Only ingest videos whose title matches this regex (case-insensitive).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip videos already indexed (by URL).")
    p.add_argument("--no-transcripts", action="store_true",
                   help="Index title/description only.")
    p.add_argument("--refresh-transcripts", action="store_true",
                   help="Only (re)process already-indexed videos that are missing "
                        "a transcript — use after configuring a transcript API key.")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    container = build_container(settings)
    catalog = YouTubeCatalog(settings)
    transcripts = TranscriptFetcher(settings)

    try:
        await container.db.create_all()
        videos = await catalog.list_videos(args.channel, args.max_videos)
        print(f"Catalog: {len(videos)} videos from "
              f"{args.channel or settings.youtube_channel_url}")

        if args.match:
            pattern = re.compile(args.match, re.IGNORECASE)
            videos = [v for v in videos if pattern.search(v.title)]
            print(f"After --match filter: {len(videos)} videos")

        stats = {"ingested": 0, "skipped": 0, "no_content": 0, "failed": 0,
                 "with_transcript": 0, "chunks": 0}
        started = time.perf_counter()

        for i, video in enumerate(videos, start=1):
            try:
                if args.refresh_transcripts:
                    # Only revisit indexed videos that still lack a transcript.
                    async with container.db.session() as session:
                        existing = await DocumentRepository(session).get_by_url(video.url)
                    if existing is None or "## Transcript" in existing.content_markdown:
                        stats["skipped"] += 1
                        continue
                elif args.skip_existing:
                    async with container.db.session() as session:
                        existing = await DocumentRepository(session).get_by_url(video.url)
                    if existing is not None:
                        stats["skipped"] += 1
                        continue

                video = await catalog.hydrate(video)
                transcript = None
                if not args.no_transcripts:
                    transcript = await transcripts.fetch(video.video_id)
                    if transcript:
                        stats["with_transcript"] += 1
                if args.refresh_transcripts and not transcript:
                    stats["skipped"] += 1
                    continue  # nothing gained — keep the existing document

                doc = video_to_document(video, transcript)
                if doc is None:
                    stats["no_content"] += 1
                    continue

                n_chunks = await container.pipeline.ingest_document(doc)
                stats["ingested"] += 1
                stats["chunks"] += n_chunks
            except Exception as exc:  # noqa: BLE001 - one video must not stop the run
                stats["failed"] += 1
                logger.error("youtube.ingest_failed", video=video.video_id,
                             error=str(exc)[:200])
            if i % 10 == 0:
                rate = i / (time.perf_counter() - started)
                print(f"  … {i}/{len(videos)} "
                      f"(ingested={stats['ingested']}, transcripts={stats['with_transcript']}, "
                      f"rate={rate:.1f}/s)")
            await asyncio.sleep(_POLITENESS_SECONDS)

        print("\nYouTube ingestion complete:")
        for key, value in stats.items():
            print(f"  {key:16s}: {value}")
    finally:
        await container.db.dispose()
        await container.cache.close()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
