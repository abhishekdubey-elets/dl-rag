"""API request/response schemas — the public wire contract of the service."""

from __future__ import annotations

from datetime import date as _date, datetime
from typing import Any

from pydantic import BaseModel, Field

from dl_rag.models.enums import (
    ConfidenceBand,
    ContentType,
    FeedbackRating,
    JobStatus,
    QueryType,
)


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class ChatFilters(BaseModel):
    year_from: int | None = None
    year_to: int | None = None
    content_types: list[ContentType] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    conversation_id: str | None = None
    stream: bool = False
    filters: ChatFilters | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceRef(BaseModel):
    index: int
    title: str
    url: str
    date: _date | None = None
    content_type: ContentType = ContentType.OTHER
    category: str | None = None
    issue: str | None = None
    author: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.LOW
    query_type: QueryType = QueryType.GENERAL
    retrieved_documents: int = 0
    conversation_id: str
    message_id: str
    latency_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
class IndexRequest(BaseModel):
    """Trigger ingestion. Either specify explicit URLs or crawl parameters."""

    urls: list[str] = Field(default_factory=list)
    content_types: list[ContentType] = Field(default_factory=list)
    since_date: _date | None = None
    max_pages: int | None = Field(default=None, ge=1)
    full_crawl: bool = False
    reindex: bool = False


class IndexResponse(BaseModel):
    job_id: str
    status: JobStatus
    accepted_urls: int = 0
    message: str = ""


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    pages_crawled: int = 0
    pages_indexed: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
class DocumentResponse(BaseModel):
    id: str
    url: str
    title: str
    subtitle: str | None = None
    author: str | None = None
    published_date: _date | None = None
    content_type: ContentType = ContentType.OTHER
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    issue_name: str | None = None
    issue_year: int | None = None
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    word_count: int = 0
    chunk_count: int = 0
    content_markdown: str | None = None


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str
    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=2000)
    reason: str | None = None


class FeedbackResponse(BaseModel):
    accepted: bool = True
    message: str = "Thanks — feedback recorded."


# --------------------------------------------------------------------------- #
# System / admin
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    checks: dict[str, str] = Field(default_factory=dict)


class AdminStatsResponse(BaseModel):
    documents_indexed: int = 0
    chunks_indexed: int = 0
    entities: int = 0
    relations: int = 0
    documents_by_type: dict[str, int] = Field(default_factory=dict)
    documents_by_year: dict[str, int] = Field(default_factory=dict)
    failed_pages: int = 0
    last_crawl_at: datetime | None = None
    vector_points: int = 0


class PopularQuestion(BaseModel):
    query: str
    count: int
    avg_confidence: float = 0.0


class AdminInsightsResponse(BaseModel):
    popular_questions: list[PopularQuestion] = Field(default_factory=list)
    avg_latency_ms: float = 0.0
    citation_frequency: dict[str, int] = Field(default_factory=dict)
    feedback_positive: int = 0
    feedback_negative: int = 0


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
