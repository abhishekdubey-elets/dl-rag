"""Channel catalog: list a YouTube channel's videos with metadata.

Two backends, chosen by configuration:

* **YouTube Data API v3** (``YOUTUBE_API_KEY`` set) — official, quota-cheap for
  catalogs (1 unit per 50 videos via the uploads playlist), full metadata.
* **yt-dlp** (keyless fallback) — extracts the channel's uploads via the public
  web endpoints. Flat extraction returns id/title cheaply; descriptions and
  exact dates are hydrated per-video only for the entries actually ingested.

Both are wrapped in ``asyncio.to_thread`` — the underlying clients are blocking.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import httpx

from dl_rag.config import Settings
from dl_rag.logging_config import get_logger

logger = get_logger(__name__)

_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"
_API_BASE = "https://www.googleapis.com/youtube/v3"


@dataclass(slots=True)
class VideoInfo:
    video_id: str
    title: str
    url: str
    description: str = ""
    published_date: date | None = None
    duration_seconds: int | None = None
    channel: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def is_hydrated(self) -> bool:
        return bool(self.description or self.published_date)


def _parse_upload_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value)
    try:
        if re.fullmatch(r"\d{8}", text):  # yt-dlp: YYYYMMDD
            return datetime.strptime(text, "%Y%m%d").date()
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


class YouTubeCatalog:
    """List and hydrate videos for a channel."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.youtube_api_key

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def list_videos(
        self, channel_url: str | None = None, max_videos: int | None = None
    ) -> list[VideoInfo]:
        """Flat catalog (id/title/url; dates when the backend provides them)."""
        channel = channel_url or self._settings.youtube_channel_url
        limit = max_videos or self._settings.youtube_max_videos
        if self._api_key:
            try:
                return await self._list_via_api(channel, limit)
            except Exception as exc:  # noqa: BLE001 - fall back to yt-dlp
                logger.warning("youtube.api_catalog_failed", error=str(exc))
        return await asyncio.to_thread(self._list_via_ytdlp, channel, limit)

    async def hydrate(self, video: VideoInfo) -> VideoInfo:
        """Fill description/date/duration for a single video (yt-dlp path)."""
        if video.is_hydrated:
            return video
        return await asyncio.to_thread(self._hydrate_via_ytdlp, video)

    # ------------------------------------------------------------------ #
    # Data API v3 backend
    # ------------------------------------------------------------------ #
    async def _list_via_api(self, channel_url: str, limit: int) -> list[VideoInfo]:
        async with httpx.AsyncClient(timeout=30) as client:
            uploads_id = await self._api_uploads_playlist(client, channel_url)
            videos: list[VideoInfo] = []
            page_token: str | None = None
            while len(videos) < limit:
                params: dict[str, Any] = {
                    "part": "snippet,contentDetails",
                    "playlistId": uploads_id,
                    "maxResults": 50,
                    "key": self._api_key,
                }
                if page_token:
                    params["pageToken"] = page_token
                resp = await client.get(f"{_API_BASE}/playlistItems", params=params)
                resp.raise_for_status()
                payload = resp.json()
                for item in payload.get("items", []):
                    snippet = item.get("snippet", {})
                    video_id = (
                        item.get("contentDetails", {}).get("videoId")
                        or snippet.get("resourceId", {}).get("videoId")
                    )
                    if not video_id:
                        continue
                    videos.append(
                        VideoInfo(
                            video_id=video_id,
                            title=snippet.get("title", ""),
                            url=_WATCH_URL.format(video_id=video_id),
                            description=snippet.get("description", ""),
                            published_date=_parse_upload_date(
                                snippet.get("publishedAt")
                            ),
                            channel=snippet.get("channelTitle", ""),
                        )
                    )
                    if len(videos) >= limit:
                        break
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break
        logger.info("youtube.catalog.api", videos=len(videos))
        return videos

    async def _api_uploads_playlist(
        self, client: httpx.AsyncClient, channel_url: str
    ) -> str:
        """Resolve any channel URL form to its uploads playlist id."""
        params: dict[str, Any] = {"part": "contentDetails", "key": self._api_key}
        if match := re.search(r"/channel/([\w\-]+)", channel_url):
            params["id"] = match.group(1)
        elif match := re.search(r"/(?:user|c)/([\w\-.]+)", channel_url):
            params["forUsername"] = match.group(1)
        elif match := re.search(r"/@([\w\-.]+)", channel_url):
            params["forHandle"] = match.group(1)
        else:
            params["id"] = channel_url  # assume a bare channel id
        resp = await client.get(f"{_API_BASE}/channels", params=params)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            raise ValueError(f"channel not found for {channel_url!r}")
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # ------------------------------------------------------------------ #
    # yt-dlp backend (keyless)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ytdlp(opts: dict[str, Any]) -> Any:
        import yt_dlp  # noqa: PLC0415 - lazy heavy import

        base = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "ignoreerrors": True,
        }
        return yt_dlp.YoutubeDL({**base, **opts})

    def _list_via_ytdlp(self, channel_url: str, limit: int) -> list[VideoInfo]:
        url = channel_url.rstrip("/")
        if not url.endswith("/videos"):
            url = f"{url}/videos"
        with self._ytdlp({"extract_flat": "in_playlist", "playlistend": limit}) as ydl:
            info = ydl.extract_info(url, download=False) or {}
        videos: list[VideoInfo] = []
        for entry in info.get("entries") or []:
            if not entry:
                continue
            video_id = entry.get("id")
            if not video_id:
                continue
            videos.append(
                VideoInfo(
                    video_id=video_id,
                    title=entry.get("title") or "",
                    url=_WATCH_URL.format(video_id=video_id),
                    description=entry.get("description") or "",
                    published_date=_parse_upload_date(entry.get("upload_date")),
                    duration_seconds=(
                        int(entry["duration"]) if entry.get("duration") else None
                    ),
                    channel=info.get("channel") or info.get("uploader") or "",
                )
            )
        logger.info("youtube.catalog.ytdlp", videos=len(videos))
        return videos

    def _hydrate_via_ytdlp(self, video: VideoInfo) -> VideoInfo:
        try:
            with self._ytdlp({}) as ydl:
                info = ydl.extract_info(video.url, download=False) or {}
        except Exception as exc:  # noqa: BLE001 - keep the flat entry
            logger.warning("youtube.hydrate_failed", video=video.video_id,
                           error=str(exc))
            return video
        video.description = info.get("description") or video.description
        video.published_date = (
            _parse_upload_date(info.get("upload_date")) or video.published_date
        )
        video.duration_seconds = (
            int(info["duration"]) if info.get("duration") else video.duration_seconds
        )
        video.channel = info.get("channel") or video.channel
        video.tags = list(info.get("tags") or [])[:10]
        return video
