"""YouTube ingestion unit tests — hermetic (no network)."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from dl_rag.config import Settings
from dl_rag.ingestion.youtube.catalog import VideoInfo, _parse_upload_date
from dl_rag.ingestion.youtube.documents import video_to_document
from dl_rag.ingestion.youtube.transcripts import TranscriptFetcher
from dl_rag.models.enums import ContentType
from dl_rag.retrieval.query_understanding import HeuristicQueryAnalyzer


def _video(**overrides) -> VideoInfo:
    base = dict(
        video_id="abc123XYZ_-",
        title="27th World Education Summit 2023 | Inaugural Session",
        url="https://www.youtube.com/watch?v=abc123XYZ_-",
        description="Highlights from the inaugural session of the 27th WES.",
        published_date=date(2023, 10, 2),
        duration_seconds=3725,
        channel="Elets Videos",
    )
    base.update(overrides)
    return VideoInfo(**base)


class TestCatalogParsing:
    def test_parse_upload_date_formats(self):
        assert _parse_upload_date("20231002") == date(2023, 10, 2)
        assert _parse_upload_date("2023-10-02T10:00:00Z") == date(2023, 10, 2)
        assert _parse_upload_date(None) is None
        assert _parse_upload_date("not-a-date") is None


class TestVideoDocument:
    def test_builds_video_document(self):
        doc = video_to_document(_video(), transcript="Welcome to the summit. " * 50)
        assert doc is not None
        assert doc.content_type == ContentType.VIDEO
        assert doc.url.startswith("https://www.youtube.com/watch")
        assert doc.issue_year == 2023
        assert "## Transcript" in doc.content_markdown
        assert doc.metadata["has_transcript"] is True
        assert doc.metadata["duration"] == "1:02:05"
        assert "youtube" in doc.tags

    def test_no_transcript_still_indexable(self):
        doc = video_to_document(_video(), transcript=None)
        assert doc is not None
        assert doc.metadata["has_transcript"] is False
        assert "## Transcript" not in doc.content_markdown

    def test_nothing_indexable_returns_none(self):
        assert video_to_document(_video(description=""), transcript=None) is None


class TestTranscriptPayloadParsing:
    def _resp(self, content: bytes | str, content_type: str) -> httpx.Response:
        return httpx.Response(
            200,
            content=content if isinstance(content, bytes) else content.encode(),
            headers={"content-type": content_type},
            request=httpx.Request("GET", "https://t.example/v"),
        )

    def test_plain_text(self):
        resp = self._resp("hello transcript", "text/plain")
        assert TranscriptFetcher._parse_keyed_payload(resp) == "hello transcript"

    def test_json_dict(self):
        resp = self._resp('{"transcript": "full text here"}', "application/json")
        assert TranscriptFetcher._parse_keyed_payload(resp) == "full text here"

    def test_json_segments(self):
        resp = self._resp(
            '[{"text": "part one"}, {"text": "part two"}]', "application/json"
        )
        assert TranscriptFetcher._parse_keyed_payload(resp) == "part one part two"

    def test_youtube_transcript_io_shape(self):
        payload = (
            '[{"id": "vid1", "tracks": [{"language": "en", "transcript": '
            '[{"text": "welcome to"}, {"text": "the summit"}]}]}]'
        )
        resp = self._resp(payload, "application/json")
        assert TranscriptFetcher._parse_keyed_payload(resp) == "welcome to the summit"


class TestVideoQueryRouting:
    @pytest.fixture
    def analyzer(self, settings: Settings) -> HeuristicQueryAnalyzer:
        return HeuristicQueryAnalyzer(settings)

    async def test_video_query_filters_to_video(self, analyzer):
        a = await analyzer.analyze("give me wes 2023 videos link")
        assert a.content_type_filter == [ContentType.VIDEO]
        assert a.time_range.from_year == 2023
        assert "World Education Summit" in a.entities

    async def test_non_video_query_unaffected(self, analyzer):
        a = await analyzer.analyze("What is SWAYAM?")
        assert ContentType.VIDEO not in a.content_type_filter