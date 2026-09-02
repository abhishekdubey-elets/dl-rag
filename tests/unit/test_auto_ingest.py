"""Auto-ingest scheduler tests — hermetic (fakes for crawler/feed/pipeline/DB)."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dl_rag.ingestion.youtube.catalog import VideoInfo
from dl_rag.ingestion.youtube.feed import VideoRelevanceFilter, normalize_title, parse_feed
from dl_rag.models.domain import SourceDocument
from dl_rag.models.enums import ContentType
from dl_rag.services import auto_ingest as module
from dl_rag.services.auto_ingest import STATE_KEY, AutoIngestService

if TYPE_CHECKING:
    from dl_rag.config import Settings

# Decorative "mathematical bold" letters, as the channel really uses in titles.
_BOLD_UNIVERSITY = "𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲"

FEED_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns:media="http://search.yahoo.com/mrss/" xmlns="http://www.w3.org/2005/Atom">
 <title>elets Insights</title>
 <entry>
  <yt:videoId>vid00000001</yt:videoId>
  <title>{_BOLD_UNIVERSITY} leaders on NEP &amp; AI</title>
  <published>2026-09-01T10:00:00+00:00</published>
  <media:group>
   <media:description>Panel at the 36th World Education Summit.</media:description>
  </media:group>
 </entry>
 <entry>
  <yt:videoId>vid00000002</yt:videoId>
  <title>How UP Police is Using AI to Fight Cybercrime</title>
  <published>2026-08-31T09:00:00+00:00</published>
  <media:group><media:description>eGov session.</media:description></media:group>
 </entry>
 <entry>
  <yt:videoId>vid00000003</yt:videoId>
  <title>Back to Campus 2026 | Inaugural Address</title>
  <published>2026-08-30T09:00:00+00:00</published>
  <media:group><media:description>Elets Back to Campus.</media:description></media:group>
 </entry>
</feed>
"""


# --------------------------------------------------------------------------- #
# Feed parsing + relevance
# --------------------------------------------------------------------------- #
class TestFeed:
    def test_parse_feed_entries(self):
        videos = parse_feed(FEED_XML)
        assert [v.video_id for v in videos] == ["vid00000001", "vid00000002", "vid00000003"]
        first = videos[0]
        assert first.url == "https://www.youtube.com/watch?v=vid00000001"
        assert first.published_date == date(2026, 9, 1)
        assert first.channel == "elets Insights"
        assert "World Education Summit" in first.description
        assert "NEP & AI" in first.title  # entities unescaped
        assert first.is_hydrated  # description + date → no yt-dlp hydrate needed

    def test_parse_feed_garbage(self):
        assert parse_feed("<not xml") == []

    def test_normalize_title_folds_decorative_unicode(self):
        assert normalize_title("𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲") == "university"


class TestRelevanceFilter:
    @pytest.fixture
    def flt(self, settings: Settings) -> VideoRelevanceFilter:
        return VideoRelevanceFilter(settings.youtube_title_pattern)

    def _v(self, title: str, description: str = "") -> VideoInfo:
        return VideoInfo(video_id="x" * 11, title=title, url="u", description=description)

    def test_education_titles_match(self, flt):
        assert flt.matches(self._v("Interview at 35th Elets World Education Summit 2026"))
        assert flt.matches(self._v("Back to Campus 2026 | Inaugural Address"))
        assert flt.matches(self._v("𝐖𝐡𝐚𝐭 𝐝𝐨𝐞𝐬 𝐢𝐭 𝐭𝐚𝐤𝐞 𝐭𝐨 𝐜𝐫𝐞𝐚𝐭𝐞 𝐚 𝐟𝐮𝐭𝐮𝐫𝐞-𝐫𝐞𝐚𝐝𝐲 𝐮𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲?"))
        assert flt.matches(self._v("Prof. X, Vice-Chancellor, on skilling"))

    def test_other_verticals_rejected(self, flt):
        assert not flt.matches(self._v("How UP Police is Using AI to Fight Cybercrime"))
        assert not flt.matches(self._v("Deepak Mohanty, Executive Director, Wells Fargo"))
        assert not flt.matches(self._v("Machine learning for hospital operations"))

    def test_description_matching_is_opt_in(self, settings: Settings):
        video = self._v("Fireside chat", description="Recorded at the World Education Summit")
        assert not VideoRelevanceFilter(settings.youtube_title_pattern).matches(video)
        assert VideoRelevanceFilter(settings.youtube_title_pattern, True).matches(video)

    def test_empty_pattern_accepts_all(self):
        assert VideoRelevanceFilter("").matches(self._v("anything"))


# --------------------------------------------------------------------------- #
# Run orchestration
# --------------------------------------------------------------------------- #
def _article(url: str, body: str) -> SourceDocument:
    doc = SourceDocument(
        id=SourceDocument.id_for_url(url), url=url, title="T", content_markdown=body,
        content_type=ContentType.NEWS,
    )
    doc.content_hash = doc.compute_hash()
    return doc


class FakeCrawler:
    def __init__(self, docs):
        self.docs = docs
        self.calls: list[datetime] = []

    async def fetch_recent_posts(self, *, modified_after, limit=500):
        self.calls.append(modified_after)
        return list(self.docs)


class FakePipeline:
    def __init__(self, fail_urls=()):
        self.ingested: list[SourceDocument] = []
        self.fail_urls = set(fail_urls)

    async def ingest_document(self, doc):
        await asyncio.sleep(0)  # yield like real I/O so concurrency tests interleave
        if doc.url in self.fail_urls:
            raise RuntimeError("boom")
        self.ingested.append(doc)
        return 3


class FakeCatalog:
    def __init__(self, extra=()):
        self.extra = list(extra)
        self.scans = 0

    async def list_videos(self, channel_url=None, max_videos=None):
        self.scans += 1
        return list(self.extra)

    async def hydrate(self, video):
        return video


class FakeTranscripts:
    def __init__(self, text=None):
        self.text = text
        self.requested: list[str] = []

    async def fetch(self, video_id):
        self.requested.append(video_id)
        return self.text


@pytest.fixture
def service(settings, fake_cache, monkeypatch):
    """Service with all I/O faked; DB lookups are stubbed per test."""
    articles = [
        _article("https://digitallearning.eletsonline.com/2026/09/new/", "fresh body"),
        _article("https://digitallearning.eletsonline.com/2026/08/same/", "unchanged body"),
        _article("https://digitallearning.eletsonline.com/2026/08/edit/", "edited body v2"),
    ]
    svc = AutoIngestService(
        settings=settings, db=None, cache=fake_cache, pipeline=FakePipeline(),
        crawler=FakeCrawler(articles), catalog=FakeCatalog(),
        transcripts=FakeTranscripts("spoken words " * 20),
    )
    existing = {
        articles[1].id: articles[1],  # identical hash → unchanged
        articles[2].id: _article(articles[2].url, "edited body v1"),  # stale hash → update
    }
    existing_urls = {"https://www.youtube.com/watch?v=vid00000003"}

    async def get_document(doc_id):
        return existing.get(doc_id)

    async def get_by_url(url):
        return existing_urls_doc if url in existing_urls else None

    existing_urls_doc = SourceDocument(id="v3", url="u", title="t")

    async def missing(limit):
        return [("docA", "vidA0000000"), ("docB", "vidB0000000")]

    monkeypatch.setattr(svc, "_get_document", get_document)
    monkeypatch.setattr(svc, "_get_document_by_url", get_by_url)
    monkeypatch.setattr(svc, "_videos_missing_transcripts", missing)

    async def fake_feed(channel_id, client=None):
        return parse_feed(FEED_XML)

    monkeypatch.setattr(module, "fetch_channel_feed", fake_feed)
    return svc


class TestRun:
    async def test_full_run_counts_and_state(self, service, fake_cache):
        result = await service.run_once(reason="test")

        assert result["errors"] == []
        assert result["articles"] == {
            "found": 3, "new": 1, "updated": 1, "unchanged": 1, "failed": 0, "chunks": 6,
        }
        # feed: 1 education (new) + 1 off-topic + 1 education already indexed
        assert result["videos"]["seen"] == 3
        assert result["videos"]["ingested"] == 1
        assert result["videos"]["off_topic"] == 1
        assert result["videos"]["existing"] == 1
        assert result["videos"]["with_transcript"] == 1
        # backfill: two candidates, both filled (docA/docB not in existing → "failed"
        # lookup path) — transcript was fetched for each
        assert result["transcripts"]["attempted"] == 2

        ingested = service._pipeline.ingested
        assert "https://www.youtube.com/watch?v=vid00000001" in [d.url for d in ingested]
        video_doc = next(d for d in ingested if d.content_type == ContentType.VIDEO)
        assert video_doc.metadata["ingested_by"] == "auto-ingest"
        assert "## Transcript" in video_doc.content_markdown

        state = fake_cache.store[STATE_KEY]
        assert state["runs"] == 1
        assert state["articles_watermark"] == result["started_at"]
        assert state["videos_watermark"] == "2026-09-01"
        assert state["last_run"]["reason"] == "test"

    async def test_watermark_drives_next_query(self, service, fake_cache):
        stamp = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)
        fake_cache.store[STATE_KEY] = {"articles_watermark": stamp.isoformat()}
        await service.run_once()
        # queried from the watermark minus the safety overlap
        assert service._crawler.calls[-1] == stamp - timedelta(hours=1)

    async def test_first_run_uses_lookback(self, service, settings):
        await service.run_once()
        since = service._crawler.calls[-1]
        expected = timedelta(days=settings.auto_ingest_lookback_days) + timedelta(hours=1)
        assert abs((datetime.now(tz=UTC) - since) - expected) < timedelta(minutes=5)

    async def test_stage_failure_is_isolated(self, service, monkeypatch):
        async def boom(state, started):
            raise RuntimeError("feed exploded")

        monkeypatch.setattr(service, "_ingest_videos", boom)
        result = await service.run_once()
        assert result["articles"]["new"] == 1  # other stages still ran
        assert result["errors"] == ["videos: feed exploded"]

    async def test_per_document_failure_counted(self, service):
        service._pipeline.fail_urls = {"https://digitallearning.eletsonline.com/2026/09/new/"}
        result = await service.run_once()
        assert result["articles"]["failed"] == 1
        assert result["articles"]["new"] == 0

    async def test_backfill_pauses_after_consecutive_misses(self, service, monkeypatch):
        service._transcripts.text = None

        async def missing(limit):
            return [(f"doc{i}", f"vid{i:08d}") for i in range(10)]

        monkeypatch.setattr(service, "_videos_missing_transcripts", missing)
        result = await service.run_once()
        assert result["transcripts"]["attempted"] == 3  # stopped at the miss limit
        assert result["transcripts"]["missing"] == 3

    async def test_concurrent_run_is_skipped(self, service):
        first, second = await asyncio.gather(service.run_once(), service.run_once())
        results = sorted([first, second], key=lambda r: "skipped" in r)
        assert "skipped" not in results[0]
        assert results[1] == {"skipped": "already_running"}

    async def test_feed_overflow_triggers_catalog_scan(self, service, fake_cache, monkeypatch):
        # 15 entries all newer than the watermark → the feed window may have
        # missed uploads → catalog scan should run and its extras be merged.
        many = [
            VideoInfo(video_id=f"vid{i:08d}", title=f"School leaders {i}", url=f"u{i}",
                      description="d", published_date=date(2026, 9, 1))
            for i in range(15)
        ]

        async def big_feed(channel_id, client=None):
            return list(many)

        monkeypatch.setattr(module, "fetch_channel_feed", big_feed)
        service._catalog.extra = [
            VideoInfo(video_id="extra000001", title="University summit", url="ux",
                      description="d", published_date=date(2026, 8, 31))
        ]
        fake_cache.store[STATE_KEY] = {"videos_watermark": "2026-08-25"}
        result = await service.run_once()
        assert service._catalog.scans == 1
        assert result["videos"]["seen"] == 16

    async def test_status_reports_state(self, service):
        await service.run_once(reason="x")
        status = await service.status()
        assert status["runs_completed"] == 1
        assert status["running"] is False
        assert status["last_run"]["reason"] == "x"
        assert "articles_watermark" in status["watermarks"]
