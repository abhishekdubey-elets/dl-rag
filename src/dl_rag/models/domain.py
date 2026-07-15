"""Internal domain models — the objects passed *between* pipeline stages.

These are distinct from the API schemas in :mod:`dl_rag.models.api`: domain
models may carry heavy fields (embeddings, full text) that never cross the wire.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

from dl_rag.models.enums import (
    ConfidenceBand,
    ContentType,
    EntityType,
    QueryType,
    RelationType,
    RetrievalSource,
)


# --------------------------------------------------------------------------- #
# Ingestion-side models
# --------------------------------------------------------------------------- #
class SourceDocument(BaseModel):
    """A single crawled page/article normalised to clean markdown."""

    id: str = Field(..., description="Stable id — sha1 of the canonical URL.")
    url: str
    title: str
    subtitle: str | None = None
    author: str | None = None
    published_date: date | None = None
    updated_date: date | None = None
    category: str | None = None
    content_type: ContentType = ContentType.OTHER
    tags: list[str] = Field(default_factory=list)
    content_markdown: str = ""
    featured_image: str | None = None
    issue_name: str | None = None
    issue_month: str | None = None
    issue_year: int | None = None
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    language: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)
    crawled_at: datetime | None = None
    content_hash: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        return len(self.content_markdown.split())

    def compute_hash(self) -> str:
        digest = hashlib.sha256(
            (self.title + "\n" + self.content_markdown).encode("utf-8")
        ).hexdigest()
        return digest

    @staticmethod
    def id_for_url(url: str) -> str:
        return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()  # noqa: S324


class ChunkMetadata(BaseModel):
    """Denormalised metadata copied onto every chunk for filter-time access."""

    url: str
    title: str
    category: str | None = None
    content_type: ContentType = ContentType.OTHER
    year: int | None = None
    month: str | None = None
    author: str | None = None
    issue: str | None = None
    published_date: date | None = None
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """A semantically-coherent slice of a document, the unit of retrieval."""

    id: str = Field(..., description="Deterministic id: '<doc_id>:<chunk_index>'.")
    document_id: str
    chunk_index: int
    text: str
    token_count: int = 0
    metadata: ChunkMetadata
    embedding: list[float] | None = None

    @staticmethod
    def make_id(document_id: str, index: int) -> str:
        return f"{document_id}:{index}"


# --------------------------------------------------------------------------- #
# Knowledge-graph models
# --------------------------------------------------------------------------- #
class Entity(BaseModel):
    id: str
    name: str
    normalized_name: str
    type: EntityType = EntityType.OTHER
    aliases: list[str] = Field(default_factory=list)
    mention_count: int = 0

    @staticmethod
    def normalize(name: str) -> str:
        return " ".join(name.lower().split())

    @staticmethod
    def make_id(name: str) -> str:
        return hashlib.sha1(Entity.normalize(name).encode("utf-8")).hexdigest()[:16]  # noqa: S324


class Relation(BaseModel):
    subject_id: str
    subject_name: str
    predicate: RelationType
    object_id: str
    object_name: str
    source_url: str
    evidence: str | None = None
    confidence: float = 0.5


# --------------------------------------------------------------------------- #
# Retrieval-side models
# --------------------------------------------------------------------------- #
class TimeRange(BaseModel):
    from_year: int | None = None
    to_year: int | None = None

    def is_set(self) -> bool:
        return self.from_year is not None or self.to_year is not None


class QueryAnalysis(BaseModel):
    """Structured understanding of a user query."""

    original_query: str
    normalized_query: str
    query_type: QueryType = QueryType.GENERAL
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    time_range: TimeRange = Field(default_factory=TimeRange)
    content_type_filter: list[ContentType] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    recency_sensitive: bool = Field(
        default=False,
        description="Query asks about 'next/upcoming/latest' — bias retrieval to fresh content.",
    )
    reasoning: str | None = None


class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval, annotated with scoring provenance."""

    chunk: Chunk
    score: float = 0.0
    rerank_score: float | None = None
    sources: list[RetrievalSource] = Field(default_factory=list)
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


class Citation(BaseModel):
    """A numbered source reference attached to an answer."""

    index: int
    title: str
    url: str
    published_date: date | None = None
    content_type: ContentType = ContentType.OTHER
    issue: str | None = None
    author: str | None = None


class GeneratedAnswer(BaseModel):
    """The full result of an answer-generation pass."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    query_type: QueryType = QueryType.GENERAL
    confidence: float = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.LOW
    retrieved_documents: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    grounded: bool = True
