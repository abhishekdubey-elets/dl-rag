"""Turn a YouTube video (+ optional transcript) into a SourceDocument.

The document then flows through the standard ingestion pipeline — semantic
chunking, embedding, Postgres/Qdrant indexing, entity/KG extraction — exactly
like an article, so citations carry the clickable watch URL.
"""

from __future__ import annotations

from datetime import datetime, timezone

from dl_rag.ingestion.youtube.catalog import VideoInfo
from dl_rag.models.domain import SourceDocument
from dl_rag.models.enums import ContentType
from dl_rag.utils.text import clean_whitespace


def _duration_label(seconds: int | None) -> str | None:
    if not seconds:
        return None
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def video_to_document(video: VideoInfo, transcript: str | None) -> SourceDocument | None:
    """Build the document, or None when there is nothing indexable."""
    title = clean_whitespace(video.title)
    description = clean_whitespace(video.description or "")
    transcript = clean_whitespace(transcript or "")
    if not title or not (description or transcript):
        return None

    sections: list[str] = []
    if description:
        sections.append(description)
    if transcript:
        sections.append(f"## Transcript\n\n{transcript}")
    body = "\n\n".join(sections)

    doc = SourceDocument(
        id=SourceDocument.id_for_url(video.url),
        url=video.url,
        title=title,
        author=video.channel or None,
        published_date=video.published_date,
        category="Videos",
        content_type=ContentType.VIDEO,
        tags=["video", "youtube", *video.tags[:8]],
        content_markdown=body,
        issue_year=video.published_date.year if video.published_date else None,
        metadata={
            "video_id": video.video_id,
            "channel": video.channel,
            "duration": _duration_label(video.duration_seconds),
            "has_transcript": bool(transcript),
        },
        crawled_at=datetime.now(tz=timezone.utc),
    )
    doc.content_hash = doc.compute_hash()
    return doc
