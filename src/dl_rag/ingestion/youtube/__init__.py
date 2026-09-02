"""YouTube ingestion: channel catalog + transcripts → SourceDocuments."""

from dl_rag.ingestion.youtube.catalog import VideoInfo, YouTubeCatalog
from dl_rag.ingestion.youtube.documents import merge_transcript, video_to_document
from dl_rag.ingestion.youtube.feed import (
    VideoRelevanceFilter,
    fetch_channel_feed,
    parse_feed,
    resolve_channel_id,
)
from dl_rag.ingestion.youtube.transcripts import TranscriptFetcher

__all__ = [
    "VideoInfo",
    "YouTubeCatalog",
    "TranscriptFetcher",
    "VideoRelevanceFilter",
    "fetch_channel_feed",
    "merge_transcript",
    "parse_feed",
    "resolve_channel_id",
    "video_to_document",
]
