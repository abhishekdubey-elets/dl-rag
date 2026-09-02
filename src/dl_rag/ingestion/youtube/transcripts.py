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
_BATCH_PROVIDER = "youtube-transcript.io"


class TranscriptProviderError(Exception):
    """The keyed provider answered with an HTTP error (rate limit, credits, outage)."""

    def __init__(self, status_code: int, detail: str = "") -> None:
        super().__init__(f"transcript provider HTTP {status_code}: {detail}")
        self.status_code = status_code


class TranscriptFetcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._languages = [
            lang.strip() for lang in settings.transcript_languages.split(",")
            if lang.strip()
        ]

    @property
    def supports_batch(self) -> bool:
        """True when the configured keyed provider accepts many ids per request."""
        url = self._settings.transcript_api_url or ""
        return _BATCH_PROVIDER in url and bool(self._settings.transcript_api_key)

    async def fetch(self, video_id: str) -> str | None:
        if self._settings.transcript_api_url:
            text = await self._fetch_keyed(video_id)
            if text:
                return text[:_MAX_TRANSCRIPT_CHARS]
        text = await asyncio.to_thread(self._fetch_keyless, video_id)
        return text[:_MAX_TRANSCRIPT_CHARS] if text else None

    async def fetch_many(
        self, video_ids: list[str], batch_size: int = 10
    ) -> dict[str, str | None]:
        """Keyed batch fetch: ``{video_id: transcript | None}`` for every id asked.

        ``None`` means the provider returned no caption track for that video.
        HTTP failures raise :class:`TranscriptProviderError` (never partial
        silent loss) so callers can back off or stop on exhausted credits.
        The keyed HTTP API is not IP-gated, unlike YouTube's own endpoints, so
        this works equally from a datacenter host.
        """
        if not self.supports_batch:
            raise ValueError("fetch_many requires the youtube-transcript.io provider")
        settings = self._settings
        results: dict[str, str | None] = {}
        async with httpx.AsyncClient(timeout=120) as client:
            for start in range(0, len(video_ids), batch_size):
                batch = video_ids[start:start + batch_size]
                resp = await client.post(
                    str(settings.transcript_api_url),
                    headers={
                        "Authorization": f"Basic {settings.transcript_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"ids": batch},
                )
                if resp.status_code >= 400:
                    raise TranscriptProviderError(resp.status_code, resp.text[:200])
                results.update(self._parse_batch(resp.json(), batch))
        return results

    @classmethod
    def _parse_batch(cls, payload: Any, requested: list[str]) -> dict[str, str | None]:
        """Map a per-video response list back onto the requested ids."""
        out: dict[str, str | None] = dict.fromkeys(requested)
        if not isinstance(payload, list):
            return out
        for item in payload:
            if not isinstance(item, dict) or item.get("id") not in out:
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                text = cls._extract_text(item.get("tracks")) or ""
            text = text.strip()
            out[item["id"]] = text[:_MAX_TRANSCRIPT_CHARS] if text else None
        return out

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
