"""`dl-fetch-captions` — bulk-fetch YouTube transcripts for a list of videos.

Input: a CSV with a ``url`` column of YouTube watch URLs (e.g.
``videos_missing_transcripts.csv``). Output: a JSON file keyed by video_id in
the shape ``dl-import-transcripts`` consumes::

    {"<video_id>": {"title", "url", "transcript"|null, "error"|null}, ...}

Two modes, picked automatically:

* **Keyed batch** — when ``TRANSCRIPT_API_URL`` points at youtube-transcript.io
  (with ``TRANSCRIPT_API_KEY``): many ids per request, not IP-gated, so it runs
  fine on a server. Stops cleanly on auth/credit errors (re-run later).
* **Keyless** — YouTube's own caption tracks via youtube-transcript-api. YouTube
  hard-blocks datacenter IPs and throttles residential ones; pace with
  ``--delay`` and re-run after a cool-down.

Resumable either way: fetched videos are skipped on re-run; videos that failed
with a retryable error are retried.

Examples:
    poetry run dl-fetch-captions videos_missing_transcripts.csv data/fetched.json
    poetry run dl-fetch-captions v.csv data/fetched.json --delay 8 --batch 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from dl_rag.config import get_settings
from dl_rag.ingestion.youtube.transcripts import TranscriptFetcher, TranscriptProviderError
from dl_rag.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

_MAX_CHARS = 120_000
RETRYABLE_ERRORS = {
    # keyless library
    "IpBlocked", "RequestBlocked", "TooManyRequests", "YouTubeRequestFailed",
    # keyed provider
    "RateLimited", "ProviderError", "NotReturned",
}
_MAX_CONSECUTIVE_BLOCKS = 5


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk-fetch YouTube transcripts to JSON.")
    p.add_argument("csv_file", type=Path, help="CSV with a 'url' column.")
    p.add_argument("out_file", type=Path, help="Output JSON (appended/resumed).")
    p.add_argument("--delay", type=float, default=2.0,
                   help="Base seconds between requests/batches (default 2).")
    p.add_argument("--batch", type=int, default=10,
                   help="Ids per request in keyed mode (default 10).")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N fetch attempts this run.")
    p.add_argument("--keyless", action="store_true",
                   help="Force the keyless library even if a keyed provider is set.")
    return p.parse_args()


def _load_videos(path: Path) -> list[dict[str, str]]:
    videos: list[dict[str, str]] = []
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            match = re.search(r"[?&]v=([\w-]{11})", row.get("url", ""))
            if match:
                videos.append({
                    "video_id": match.group(1),
                    "title": row.get("title", ""),
                    "url": row["url"],
                })
    return videos


def _pending(videos: list[dict[str, str]], results: dict[str, Any]) -> list[dict[str, str]]:
    return [
        v for v in videos
        if v["video_id"] not in results
        or (not results[v["video_id"]].get("transcript")
            and results[v["video_id"]].get("error") in RETRYABLE_ERRORS)
    ]


class _Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.results: dict[str, Any] = {}
        if path.exists():
            self.results = json.loads(path.read_text(encoding="utf-8"))

    def record(self, video: dict[str, str], transcript: str | None, error: str | None) -> None:
        self.results[video["video_id"]] = {
            "title": video["title"], "url": video["url"],
            "transcript": transcript, "error": error,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.results, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def fetched(self) -> int:
        return sum(1 for r in self.results.values() if r.get("transcript"))

    def summary(self) -> None:
        errors: dict[str, int] = {}
        for r in self.results.values():
            if r.get("error"):
                errors[r["error"]] = errors.get(r["error"], 0) + 1
        print(f"\nfetched {self.fetched()}/{len(self.results)} -> {self.path}", flush=True)
        if errors:
            print("errors: " + json.dumps(errors), flush=True)


# --------------------------------------------------------------------------- #
# Keyed batch mode
# --------------------------------------------------------------------------- #
async def _run_keyed(
    fetcher: TranscriptFetcher, pending: list[dict[str, str]], store: _Store,
    args: argparse.Namespace,
) -> None:
    by_id = {v["video_id"]: v for v in pending}
    ids = list(by_id)
    for start in range(0, len(ids), args.batch):
        batch = ids[start:start + args.batch]
        try:
            got = await fetcher.fetch_many(batch, batch_size=len(batch))
        except TranscriptProviderError as exc:
            if exc.status_code == 429:
                print(f"  rate limited — waiting 30s ({exc})", flush=True)
                await asyncio.sleep(30)
                try:
                    got = await fetcher.fetch_many(batch, batch_size=len(batch))
                except TranscriptProviderError as exc2:
                    for vid in batch:
                        store.record(by_id[vid], None, "RateLimited")
                    store.save()
                    print(f"aborting: provider still rate-limiting ({exc2}); "
                          "re-run later (progress saved)", flush=True)
                    return
            elif exc.status_code in (401, 402, 403):
                store.save()
                print(f"aborting: provider refused ({exc}) — check the key / credits; "
                      "unattempted videos remain pending", flush=True)
                return
            else:
                for vid in batch:
                    store.record(by_id[vid], None, "ProviderError")
                store.save()
                print(f"  provider error on batch ({exc}) — recorded as retryable",
                      flush=True)
                await asyncio.sleep(args.delay + 5)
                continue

        for vid in batch:
            text = got.get(vid)
            if text:
                store.record(by_id[vid], text[:_MAX_CHARS], None)
            elif vid in got:
                store.record(by_id[vid], None, "NoCaptions")
            else:
                store.record(by_id[vid], None, "NotReturned")
        store.save()
        done = start + len(batch)
        print(f"  progress {done}/{len(ids)} — with transcript: {store.fetched()}",
              flush=True)
        if done < len(ids):
            await asyncio.sleep(args.delay + random.random())


# --------------------------------------------------------------------------- #
# Keyless mode
# --------------------------------------------------------------------------- #
def _fetch_keyless(api: Any, video_id: str, languages: list[str]) -> str | None:
    fetched = api.fetch(video_id, languages=languages)
    text = " ".join(seg.text.strip() for seg in fetched if seg.text).strip()
    return text[:_MAX_CHARS] if text else None


def _run_keyless(  # noqa: C901 - linear CLI flow
    pending: list[dict[str, str]], store: _Store, args: argparse.Namespace,
    languages: list[str],
) -> None:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    consecutive_blocks = 0
    for i, video in enumerate(pending, start=1):
        vid = video["video_id"]
        transcript: str | None = None
        error: str | None = None
        try:
            transcript = _fetch_keyless(api, vid, languages)
            if not transcript:
                error = "empty"
            consecutive_blocks = 0
        except Exception as exc:  # noqa: BLE001 - classify and continue
            error = type(exc).__name__
            if error in RETRYABLE_ERRORS:
                consecutive_blocks += 1
                wait = min(600, 60 * consecutive_blocks)
                print(f"  [{i}] {vid} blocked ({error}) — backing off {wait}s", flush=True)
                time.sleep(wait)
                if consecutive_blocks >= _MAX_CONSECUTIVE_BLOCKS:
                    store.record(video, None, error)
                    store.save()
                    print("aborting: persistent rate-limiting — re-run later "
                          "to continue (progress saved)", flush=True)
                    return
            else:
                consecutive_blocks = 0

        store.record(video, transcript, error)
        if i % 10 == 0 or i == len(pending):
            store.save()
            print(f"  progress {i}/{len(pending)} — with transcript: {store.fetched()}",
                  flush=True)
        time.sleep(args.delay + random.random())


def main() -> None:
    configure_logging()
    args = _parse_args()
    settings = get_settings()
    fetcher = TranscriptFetcher(settings)
    languages = [
        lang.strip() for lang in settings.transcript_languages.split(",") if lang.strip()
    ]

    videos = _load_videos(args.csv_file)
    store = _Store(args.out_file)
    pending = _pending(videos, store.results)
    if args.limit:
        pending = pending[: args.limit]
    keyed = fetcher.supports_batch and not args.keyless
    print(f"total={len(videos)} done={len(store.results)} pending={len(pending)} "
          f"mode={'keyed-batch' if keyed else 'keyless'}", flush=True)

    if keyed:
        asyncio.run(_run_keyed(fetcher, pending, store, args))
    else:
        _run_keyless(pending, store, args, languages)
    store.save()
    store.summary()


if __name__ == "__main__":
    main()
