"""`dl-import-transcripts` — merge locally-saved video transcripts into the index.

Source: a JSON file keyed by YouTube ``video_id``::

    {
      "<video_id>": {"title": "...", "url": "...", "transcript": "...", "error": null},
      ...
    }

(as produced by the bulk caption fetcher, but any tool can emit the shape —
entries with a null/empty ``transcript`` are skipped).

For every entry whose ``video_id`` matches an already-indexed video document,
the transcript is merged into the document body under ``## Transcript`` and
the document is re-ingested through the standard pipeline (re-chunk →
re-embed → re-index → KG refresh) — the same flow as ``dl-import-supabase``,
just with a file instead of the transcripts database as the source.

Examples:
    poetry run dl-import-transcripts data/missing_transcripts_fetched.json --limit 3
    poetry run dl-import-transcripts data/missing_transcripts_fetched.json
    poetry run dl-import-transcripts fetched.json --overwrite   # refresh existing too
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from sqlalchemy import text as sqltext

from dl_rag.api.deps import build_container
from dl_rag.config import get_settings
from dl_rag.ingestion.youtube.documents import TRANSCRIPT_HEADING as _TRANSCRIPT_HEADING
from dl_rag.ingestion.youtube.documents import merge_transcript as _merge_transcript
from dl_rag.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Import video transcripts from a local JSON file."
    )
    p.add_argument("file", type=Path, help="JSON file keyed by video_id.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process N videos (trial).")
    p.add_argument("--overwrite", action="store_true",
                   help="Also refresh documents that already contain a transcript.")
    return p.parse_args()


def _load_transcripts(path: Path) -> dict[str, str]:
    """Return ``{video_id: transcript}`` for entries with usable text."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a JSON object keyed by video_id")
    out: dict[str, str] = {}
    for video_id, entry in payload.items():
        if isinstance(entry, str):  # tolerate a flat {id: text} shape
            text = entry.strip()
        elif isinstance(entry, dict):
            text = (entry.get("transcript") or "").strip()
        else:
            continue
        if text:
            out[video_id] = text
    return out


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    container = build_container(settings)

    transcripts = _load_transcripts(args.file)
    print(f"transcripts in file: {len(transcripts)}")

    try:
        # --- indexed videos and their transcript state ----------------------
        async with container.db.session() as session:
            rows = (await session.execute(sqltext("""
                SELECT id, metadata_json->>'video_id' AS video_id,
                       (content_markdown LIKE :h) AS has_transcript
                FROM documents
                WHERE content_type = 'video' AND metadata_json->>'video_id' IS NOT NULL
            """), {"h": f"%{_TRANSCRIPT_HEADING}%"})).fetchall()
        ours = {r.video_id: {"id": r.id, "has": bool(r.has_transcript)} for r in rows}

        targets = [
            vid for vid in transcripts
            if vid in ours and (args.overwrite or not ours[vid]["has"])
        ]
        if args.limit:
            targets = targets[: args.limit]
        print(f"indexed videos: {len(ours)} | matched in file: "
              f"{len(set(ours) & set(transcripts))} | to import now: {len(targets)}")

        stats = {"updated": 0, "failed": 0, "chunks": 0}
        started = time.perf_counter()

        for i, video_id in enumerate(targets, start=1):
            try:
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
                    doc.content_markdown, transcripts[video_id]
                )
                doc.metadata["has_transcript"] = True
                doc.metadata["transcript_source"] = "youtube_captions"
                doc.content_hash = doc.compute_hash()

                n_chunks = await container.pipeline.ingest_document(doc)
                stats["updated"] += 1
                stats["chunks"] += n_chunks
            except Exception as exc:  # noqa: BLE001 - continue past bad videos
                stats["failed"] += 1
                logger.error("transcripts.import_failed", video=video_id,
                             error=str(exc)[:200])

            if i % 20 == 0:
                rate = i / (time.perf_counter() - started)
                eta = (len(targets) - i) / rate / 60 if rate else 0
                print(f"  … {i}/{len(targets)} (updated={stats['updated']}, "
                      f"rate={rate:.1f}/s, eta={eta:.0f}m)")

        print("\nLocal transcript import complete:")
        for key, value in stats.items():
            print(f"  {key:10s}: {value}")
    finally:
        await container.db.dispose()
        await container.cache.close()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
