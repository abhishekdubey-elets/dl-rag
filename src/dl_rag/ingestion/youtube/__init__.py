"""YouTube ingestion: channel catalog + transcripts → SourceDocuments."""

from dl_rag.ingestion.youtube.catalog import VideoInfo, YouTubeCatalog
from dl_rag.ingestion.youtube.documents import video_to_document
from dl_rag.ingestion.youtube.transcripts import TranscriptFetcher

__all__ = ["VideoInfo", "YouTubeCatalog", "TranscriptFetcher", "video_to_document"]
