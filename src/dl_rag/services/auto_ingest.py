"""AutoIngestService — keep the index current without manual crawls.

On a fixed cadence (default daily) one *run* does three things, each guarded
so a failure in one never blocks the others:

1. **Articles** — every post on the WordPress site created or edited since the
   last run (``modified_after`` watermark) is mapped via the REST API and, if
   its content hash changed, re-ingested (chunks → embeddings → Postgres +
   Qdrant → knowledge graph).
2. **Videos** — the channel's latest uploads (Atom feed; catalog fallback when
   the 15-entry feed overflows) are filtered to the education vertical by
   title, and every video not yet indexed is ingested with whatever transcript
   is obtainable (title + description at minimum).
3. **Transcript backfill** — a handful of indexed videos still lacking a
   transcript are retried, so captions that appear later (or a newly
   configured transcript provider) flow in without a manual import.

State (watermarks + last-run summary) lives in Redis under one key, so the
loop survives restarts and the admin API can report it. Runs are serialised
by an ``asyncio.Lock``; a run requested while one is active is skipped.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import text as sqltext

from dl_rag.ingestion.youtube.documents import merge_transcript, video_to_document
from dl_rag.ingestion.youtube.feed import (
    VideoRelevanceFilter,
    fetch_channel_feed,
    resolve_channel_id,
)
from dl_rag.logging_config import get_logger
from dl_rag.observability import metrics
from dl_rag.repositories.document_repository import DocumentRepository

if TYPE_CHECKING:
    from dl_rag.config import Settings
    from dl_rag.db.database import Database
    from dl_rag.ingestion.crawler.wordpress import WordPressCrawler
    from dl_rag.ingestion.pipeline import IngestionPipeline
    from dl_rag.ingestion.youtube.catalog import VideoInfo, YouTubeCatalog
    from dl_rag.ingestion.youtube.transcripts import TranscriptFetcher
    from dl_rag.models.domain import SourceDocument
    from dl_rag.protocols import Cache

logger = get_logger(__name__)

STATE_KEY = "autoingest:state"
_FEED_WINDOW = 15  # entries the Atom feed exposes
# Overlap re-applied to the article watermark: absorbs clock skew between us
# and the site, and late edits landing exactly at the boundary.
_ARTICLE_OVERLAP = timedelta(hours=1)
# Consecutive transcript misses that abort the backfill for this run — the
# usual cause is a per-IP block, and hammering it only prolongs it.
_BACKFILL_MISS_LIMIT = 3


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class AutoIngestService:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        cache: Cache,
        pipeline: IngestionPipeline,
        crawler: WordPressCrawler,
        catalog: YouTubeCatalog,
        transcripts: TranscriptFetcher,
    ) -> None:
        self._settings = settings
        self._db = db
        self._cache = cache
        self._pipeline = pipeline
        self._crawler = crawler
        self._catalog = catalog
        self._transcripts = transcripts
        self._filter = VideoRelevanceFilter(
            settings.youtube_title_pattern, settings.youtube_match_description
        )
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._next_run_at: datetime | None = None
        self._last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Background loop
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Launch the periodic loop (idempotent)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="auto-ingest")
            logger.info(
                "autoingest.started",
                interval_hours=self._settings.auto_ingest_interval_hours,
                startup_delay_seconds=self._settings.auto_ingest_startup_delay_seconds,
            )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):  # shutdown path
            await self._task
        self._task = None
        self._next_run_at = None

    @property
    def is_running_loop(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _loop(self) -> None:
        delay = max(0, self._settings.auto_ingest_startup_delay_seconds)
        interval = max(60.0, self._settings.auto_ingest_interval_hours * 3600.0)
        self._next_run_at = _utcnow() + timedelta(seconds=delay)
        await asyncio.sleep(delay)
        while True:
            try:
                await self.run_once(reason="scheduled")
            except Exception as exc:  # noqa: BLE001 - the loop must outlive any run
                logger.error("autoingest.run_crashed", error=str(exc), exc_info=exc)
            self._next_run_at = _utcnow() + timedelta(seconds=interval)
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    async def load_state(self) -> dict[str, Any]:
        try:
            state = await self._cache.get_json(STATE_KEY)
        except Exception as exc:  # noqa: BLE001 - Redis outage → fresh state
            logger.warning("autoingest.state_load_failed", error=str(exc))
            return {}
        return dict(state) if isinstance(state, dict) else {}

    async def save_state(self, state: dict[str, Any]) -> None:
        try:
            await self._cache.set_json(STATE_KEY, state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("autoingest.state_save_failed", error=str(exc))

    async def status(self) -> dict[str, Any]:
        state = await self.load_state()
        return {
            "enabled": self._settings.auto_ingest_enabled,
            "loop_active": self.is_running_loop,
            "interval_hours": self._settings.auto_ingest_interval_hours,
            "running": self._lock.locked(),
            "runs_completed": int(state.get("runs", 0)),
            "next_run_at": self._next_run_at,
            "last_run": self._last_result or state.get("last_run"),
            "watermarks": {
                k: v for k, v in state.items() if k.endswith("_watermark")
            },
            "channel_id": state.get("channel_id") or self._settings.youtube_channel_id,
        }

    # ------------------------------------------------------------------ #
    # One run
    # ------------------------------------------------------------------ #
    async def run_once(self, *, reason: str = "manual") -> dict[str, Any]:
        if self._lock.locked():
            logger.info("autoingest.run_skipped", reason="already_running")
            return {"skipped": "already_running"}

        async with self._lock:
            started = _utcnow()
            clock = time.perf_counter()
            state = await self.load_state()
            result: dict[str, Any] = {
                "started_at": started.isoformat(),
                "reason": reason,
                "articles": {},
                "videos": {},
                "transcripts": {},
                "errors": [],
            }
            logger.info("autoingest.run_started", reason=reason)

            for name, step in (
                ("articles", self._ingest_articles),
                ("videos", self._ingest_videos),
                ("transcripts", self._backfill_transcripts),
            ):
                try:
                    result[name] = await step(state, started)
                except Exception as exc:  # noqa: BLE001 - isolate each stage
                    logger.error(f"autoingest.{name}_failed", error=str(exc), exc_info=exc)
                    result["errors"].append(f"{name}: {str(exc)[:200]}")

            result["finished_at"] = _utcnow().isoformat()
            result["duration_seconds"] = round(time.perf_counter() - clock, 1)
            state["runs"] = int(state.get("runs", 0)) + 1
            state["last_run"] = result
            await self.save_state(state)
            self._last_result = result

            metrics.AUTO_INGEST_RUNS.labels(
                "error" if result["errors"] else "ok"
            ).inc()
            logger.info(
                "autoingest.run_finished",
                duration_seconds=result["duration_seconds"],
                articles=result["articles"],
                videos=result["videos"],
                transcripts=result["transcripts"],
                errors=len(result["errors"]),
            )
            return result

    # ------------------------------------------------------------------ #
    # Stage 1 — articles
    # ------------------------------------------------------------------ #
    async def _ingest_articles(
        self, state: dict[str, Any], run_started: datetime
    ) -> dict[str, int]:
        watermark = _parse_dt(state.get("articles_watermark"))
        if watermark is None:
            watermark = run_started - timedelta(days=self._settings.auto_ingest_lookback_days)
        since = watermark - _ARTICLE_OVERLAP

        documents = await self._crawler.fetch_recent_posts(modified_after=since)
        counts = {"found": len(documents), "new": 0, "updated": 0,
                  "unchanged": 0, "failed": 0, "chunks": 0}
        for document in documents:
            existing = await self._get_document(document.id)
            if existing is not None and existing.content_hash == document.content_hash:
                counts["unchanged"] += 1
                continue
            try:
                counts["chunks"] += await self._pipeline.ingest_document(document)
            except Exception as exc:  # noqa: BLE001 - one bad post must not stop the run
                counts["failed"] += 1
                logger.error("autoingest.article_failed", url=document.url,
                             error=str(exc)[:200])
                continue
            counts["updated" if existing is not None else "new"] += 1
            metrics.AUTO_INGEST_DOCS.labels("article").inc()

        # Advance to the run start (not "now"): anything published while the
        # run was in flight is picked up next time.
        state["articles_watermark"] = run_started.isoformat()
        return counts

    # ------------------------------------------------------------------ #
    # Stage 2 — videos
    # ------------------------------------------------------------------ #
    async def _ingest_videos(
        self, state: dict[str, Any], run_started: datetime
    ) -> dict[str, int]:
        videos = await self._list_new_videos(state)
        counts = {"seen": len(videos), "off_topic": 0, "existing": 0,
                  "ingested": 0, "with_transcript": 0, "empty": 0,
                  "failed": 0, "chunks": 0}
        newest: date | None = None

        for video in videos:
            if video.published_date and (newest is None or video.published_date > newest):
                newest = video.published_date
            if not self._filter.matches(video):
                counts["off_topic"] += 1
                continue
            if await self._get_document_by_url(video.url) is not None:
                counts["existing"] += 1
                continue
            try:
                video = await self._catalog.hydrate(video)
                transcript = await self._fetch_transcript(video.video_id)
                document = video_to_document(video, transcript)
                if document is None:
                    counts["empty"] += 1
                    continue
                document.metadata["ingested_by"] = "auto-ingest"
                counts["chunks"] += await self._pipeline.ingest_document(document)
            except Exception as exc:  # noqa: BLE001 - continue past bad videos
                counts["failed"] += 1
                logger.error("autoingest.video_failed", video=video.video_id,
                             error=str(exc)[:200])
                continue
            counts["ingested"] += 1
            if transcript:
                counts["with_transcript"] += 1
            metrics.AUTO_INGEST_DOCS.labels("video").inc()
            logger.info("autoingest.video_ingested", video=video.video_id,
                        title=video.title[:80], transcript=bool(transcript))

        if newest is not None:
            state["videos_watermark"] = newest.isoformat()
        return counts

    async def _list_new_videos(self, state: dict[str, Any]) -> list[VideoInfo]:
        """Latest uploads: Atom feed first, catalog scan when the feed overflowed."""
        channel_id = self._settings.youtube_channel_id or state.get("channel_id")
        if not channel_id:
            channel_id = await resolve_channel_id(self._settings.youtube_channel_url)
            if channel_id:
                state["channel_id"] = channel_id
        feed = await fetch_channel_feed(channel_id) if channel_id else []

        watermark = _parse_dt(state.get("videos_watermark"))
        watermark_day = watermark.date() if watermark else None
        overflow = (
            len(feed) >= _FEED_WINDOW
            and watermark_day is not None
            and all(v.published_date is None or v.published_date >= watermark_day for v in feed)
        )
        if not feed or overflow:
            # First run, feed failure, or more than a feed's worth of uploads
            # since last time → scan deeper via Data API / yt-dlp (best effort;
            # yt-dlp may be bot-gated from datacenter IPs, hence the guard).
            try:
                extra = await self._catalog.list_videos(
                    max_videos=self._settings.auto_ingest_video_scan_limit
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("autoingest.catalog_scan_failed", error=str(exc)[:160])
                extra = []
            known = {v.video_id for v in feed}
            feed.extend(v for v in extra if v.video_id not in known)
        return feed

    async def _fetch_transcript(self, video_id: str) -> str | None:
        try:
            return await self._transcripts.fetch(video_id)
        except Exception as exc:  # noqa: BLE001 - transcript is optional
            logger.info("autoingest.transcript_failed", video=video_id, error=str(exc)[:120])
            return None

    # ------------------------------------------------------------------ #
    # Stage 3 — transcript backfill
    # ------------------------------------------------------------------ #
    async def _backfill_transcripts(
        self, state: dict[str, Any], run_started: datetime
    ) -> dict[str, int]:
        limit = self._settings.auto_ingest_transcript_backfill
        counts = {"attempted": 0, "filled": 0, "missing": 0, "failed": 0, "chunks": 0}
        if limit <= 0:
            return counts

        candidates = await self._videos_missing_transcripts(limit)
        misses = 0
        for doc_id, video_id in candidates:
            counts["attempted"] += 1
            transcript = await self._fetch_transcript(video_id)
            if not transcript:
                counts["missing"] += 1
                misses += 1
                if misses >= _BACKFILL_MISS_LIMIT:
                    logger.info("autoingest.backfill_paused", consecutive_misses=misses)
                    break
                continue
            misses = 0
            document = await self._get_document(doc_id)
            if document is None:
                counts["failed"] += 1
                continue
            document.content_markdown = merge_transcript(document.content_markdown, transcript)
            document.metadata["has_transcript"] = True
            document.metadata["transcript_source"] = "youtube_captions"
            document.content_hash = document.compute_hash()
            try:
                counts["chunks"] += await self._pipeline.ingest_document(document)
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                logger.error("autoingest.backfill_failed", video=video_id,
                             error=str(exc)[:200])
                continue
            counts["filled"] += 1
            metrics.AUTO_INGEST_DOCS.labels("transcript").inc()
        return counts

    # ------------------------------------------------------------------ #
    # Persistence helpers (kept small so tests can stub them)
    # ------------------------------------------------------------------ #
    async def _get_document(self, doc_id: str) -> SourceDocument | None:
        async with self._db.session() as session:
            return await DocumentRepository(session).get(doc_id)

    async def _get_document_by_url(self, url: str) -> SourceDocument | None:
        async with self._db.session() as session:
            return await DocumentRepository(session).get_by_url(url)

    async def _videos_missing_transcripts(self, limit: int) -> list[tuple[str, str]]:
        """Newest indexed videos without a transcript → ``[(doc_id, video_id)]``."""
        async with self._db.session() as session:
            rows = (await session.execute(sqltext("""
                SELECT id, metadata_json->>'video_id' AS video_id
                FROM documents
                WHERE content_type = 'video'
                  AND metadata_json->>'video_id' IS NOT NULL
                  AND COALESCE(metadata_json->>'has_transcript', 'false') <> 'true'
                ORDER BY published_date DESC NULLS LAST, created_at DESC
                LIMIT :lim
            """), {"lim": limit})).fetchall()
        return [(r.id, r.video_id) for r in rows]
