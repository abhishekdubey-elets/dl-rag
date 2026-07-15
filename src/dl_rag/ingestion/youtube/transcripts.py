"""Transcript fetching with a keyed-provider option and a keyless fallback.

Order of attempts:

1. **Keyed HTTP provider** when ``TRANSCRIPT_API_URL`` is configured — a URL
   template containing ``{video_id}``; the key goes in the configured header.
   Accepts either plain text or common JSON shapes (``{"transcript": ...}``,
   ``{"text": ...}``, or a list of ``{"text": ...}`` segments).
2. **youtube-transcript-api** (keyless) — fetches YouTube's own caption tracks,
   preferring the configured language list, falling back to any available
   (including auto-generated) track.

Every failure path returns ``None`` — a video without a transcript is still
ingested from its title/description.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from dl_rag.config import Settings
from dl_rag.logging_config import get_logger

logger = get_logger(__name__)

_MAX_TRANSCRIPT_CHARS = 120_000


class TranscriptFetcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._languages = [
            lang.strip() for lang in settings.transcript_languages.split(",")
            if lang.strip()
        ]

    async def fetch(self, video_id: str) -> str | None:
        if self._settings.transcript_api_url:
            text = await self._fetch_keyed(video_id)
            if text:
                return text[:_MAX_TRANSCRIPT_CHARS]
        text = await asyncio.to_thread(self._fetch_keyless, video_id)
        return text[:_MAX_TRANSCRIPT_CHARS] if text else None

    # ------------------------------------------------------------------ #
    async def _fetch_keyed(self, video_id: str) -> str | None:
        settings = self._settings
        url = str(settings.transcript_api_url)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if "youtube-transcript.io" in url:
                    # Provider-specific protocol: POST + Basic token auth.
                    resp = await client.post(
                        url,
                        headers={
                            "Authorization": f"Basic {settings.transcript_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"ids": [video_id]},
                    )
                else:
                    headers = {}
                    if settings.transcript_api_key:
                        headers[settings.transcript_api_key_header] = (
                            settings.transcript_api_key
                        )
                    resp = await client.get(
                        url.format(video_id=video_id), headers=headers
                    )
                resp.raise_for_status()
                return self._parse_keyed_payload(resp)
        except Exception as exc:  # noqa: BLE001 - fall through to keyless
            logger.warning("transcript.keyed_failed", video=video_id,
                           error=str(exc)[:200])
            return None

    @classmethod
    def _parse_keyed_payload(cls, resp: httpx.Response) -> str | None:
        content_type = resp.headers.get("content-type", "")
        if "json" not in content_type:
            text = resp.text.strip()
            return text or None
        return cls._extract_text(resp.json())

    @classmethod
    def _extract_text(cls, payload: Any) -> str | None:
        """Pull transcript text out of the common provider JSON shapes.

        Handles: a bare string; {"transcript"/"text"/"content": str|list};
        {"tracks": [{"transcript": [...]}]} (youtube-transcript.io); a list of
        segments ({"text": ...}); and a list of per-video objects containing
        any of the above.
        """
        if isinstance(payload, str):
            return payload.strip() or None
        if isinstance(payload, dict):
            for key in ("transcript", "text", "content", "tracks"):
                if key in payload:
                    found = cls._extract_text(payload[key])
                    if found:
                        return found
            return None
        if isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                if isinstance(item, str):
                    parts.append(item.strip())
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"].strip())
                else:
                    nested = cls._extract_text(item)
                    if nested:
                        return nested  # per-video/track wrapper — first hit wins
            joined = " ".join(p for p in parts if p)
            return joined or None
        return None

    # ------------------------------------------------------------------ #
    def _fetch_keyless(self, video_id: str) -> str | None:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # noqa: PLC0415
        except ImportError:
            logger.warning("transcript.library_missing")
            return None
        try:
            segments = self._fetch_segments(YouTubeTranscriptApi, video_id)
            if segments is None:
                return None
            parts = [
                getattr(seg, "text", None)
                or (seg.get("text") if isinstance(seg, dict) else "")
                for seg in segments
            ]
            joined = " ".join(p.strip() for p in parts if p and p.strip())
            return joined or None
        except Exception as exc:  # noqa: BLE001 - no captions / blocked / private
            logger.info("transcript.unavailable", video=video_id, error=str(exc)[:120])
            return None

    def _fetch_segments(self, api_cls: Any, video_id: str) -> Any | None:
        """Handle both youtube-transcript-api generations.

        * ≥1.0: instance methods — ``api.list(id)`` / ``api.fetch(id, languages)``
        * <1.0: classmethods — ``list_transcripts(id)`` / ``get_transcript(id)``
        """
        if hasattr(api_cls, "list_transcripts"):  # legacy classmethod API
            try:
                transcript_list = api_cls.list_transcripts(video_id)
                transcript = self._pick_transcript(transcript_list)
                return transcript.fetch() if transcript is not None else None
            except Exception:  # noqa: BLE001 - try the blunt getter before giving up
                return api_cls.get_transcript(video_id, languages=self._languages)
        api = api_cls()  # modern instance API
        transcript_list = api.list(video_id)
        transcript = self._pick_transcript(transcript_list)
        return transcript.fetch() if transcript is not None else None

    def _pick_transcript(self, transcript_list: Any) -> Any | None:
        try:
            return transcript_list.find_transcript(self._languages)
        except Exception:  # noqa: BLE001 - any available track beats none
            for candidate in transcript_list:
                return candidate
            return None
