"""YouTube channel Atom feed + vertical relevance filter.

The public feed ``https://www.youtube.com/feeds/videos.xml?channel_id=…`` lists
a channel's latest 15 uploads with title, description and publish time. It is
keyless and — unlike the caption/innertube endpoints — not bot-gated, so it is
the cheapest reliable "what's new" signal for the scheduler, even from a
datacenter IP. Deeper scans fall back to :class:`YouTubeCatalog`.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime
from html import unescape

import httpx

from dl_rag.ingestion.youtube.catalog import VideoInfo
from dl_rag.logging_config import get_logger

logger = get_logger(__name__)

FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}
_CHANNEL_ID_RES = (
    re.compile(r'"externalId":"(UC[\w-]{20,})"'),
    re.compile(r'<link rel="canonical" href="https://www\.youtube\.com/channel/(UC[\w-]{20,})"'),
    re.compile(r'<meta itemprop="identifier" content="(UC[\w-]{20,})"'),
)


def _text(node: ET.Element | None) -> str:
    return unescape((node.text or "").strip()) if node is not None else ""


def parse_feed(xml_text: str) -> list[VideoInfo]:
    """Parse a channel Atom feed into flat :class:`VideoInfo` entries (newest first)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("youtube.feed.parse_failed", error=str(exc)[:120])
        return []

    channel = _text(root.find("atom:title", _NS))
    videos: list[VideoInfo] = []
    for entry in root.findall("atom:entry", _NS):
        video_id = _text(entry.find("yt:videoId", _NS))
        if not video_id:
            continue
        published: datetime | None = None
        raw = _text(entry.find("atom:published", _NS))
        if raw:
            try:
                published = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                published = None
        group = entry.find("media:group", _NS)
        description = _text(group.find("media:description", _NS)) if group is not None else ""
        videos.append(
            VideoInfo(
                video_id=video_id,
                title=_text(entry.find("atom:title", _NS)),
                url=_WATCH_URL.format(video_id=video_id),
                description=description,
                published_date=published.date() if published else None,
                channel=channel,
            )
        )
    return videos


async def fetch_channel_feed(
    channel_id: str, client: httpx.AsyncClient | None = None
) -> list[VideoInfo]:
    """Download + parse the channel feed; ``[]`` on any failure."""
    url = FEED_URL.format(channel_id=channel_id)
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=30, follow_redirects=True)
    try:
        response = await client.get(url)
        response.raise_for_status()
        videos = parse_feed(response.text)
        logger.info("youtube.feed.fetched", channel_id=channel_id, entries=len(videos))
        return videos
    except Exception as exc:  # noqa: BLE001 - feed is best-effort input
        logger.warning("youtube.feed.failed", channel_id=channel_id, error=str(exc)[:160])
        return []
    finally:
        if own_client:
            await client.aclose()


async def resolve_channel_id(
    channel_url: str, client: httpx.AsyncClient | None = None
) -> str | None:
    """Resolve a channel URL (``/@handle``, ``/user/x``, ``/channel/UC…``) to its id."""
    if match := re.search(r"/channel/(UC[\w-]{20,})", channel_url):
        return match.group(1)
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=30, follow_redirects=True)
    try:
        response = await client.get(channel_url.rstrip("/") + "/videos")
        response.raise_for_status()
        for pattern in _CHANNEL_ID_RES:
            if match := pattern.search(response.text):
                return match.group(1)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("youtube.channel_id.resolve_failed", url=channel_url,
                       error=str(exc)[:160])
        return None
    finally:
        if own_client:
            await client.aclose()


def normalize_title(text: str) -> str:
    """NFKC-fold decorative Unicode ("𝐮𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲" → "university") and casefold."""
    return unicodedata.normalize("NFKC", text or "").casefold()


class VideoRelevanceFilter:
    """Keep only videos that belong to the digitalLEARNING (education) vertical."""

    def __init__(self, pattern: str, match_description: bool = False) -> None:
        self._pattern = re.compile(pattern, re.IGNORECASE) if pattern.strip() else None
        self._match_description = match_description

    def matches(self, video: VideoInfo) -> bool:
        if self._pattern is None:
            return True
        haystack = normalize_title(video.title)
        if self._match_description:
            haystack = f"{haystack}\n{normalize_title(video.description)}"
        return self._pattern.search(haystack) is not None
