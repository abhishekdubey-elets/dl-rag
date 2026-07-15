"""SQLAlchemy 2.0 declarative ORM models for the digitalLEARNING RAG store.

Every table maps 1:1 (or denormalised) onto the domain models in
:mod:`dl_rag.models.domain`. Columns use ``Mapped`` / ``mapped_column`` typing
so the schema is both a runtime contract and a static one.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
class DocumentORM(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    url: Mapped[str] = mapped_column(String, unique=True, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    subtitle: Mapped[str | None] = mapped_column(String, nullable=True)
    author: Mapped[str | None] = mapped_column(String, nullable=True)
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    updated_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    content_type: Mapped[str] = mapped_column(String, nullable=False, default="other")
    tags: Mapped[list[str]] = mapped_column(JSONB, default=list)
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False, default="")
    featured_image: Mapped[str | None] = mapped_column(String, nullable=True)
    issue_name: Mapped[str | None] = mapped_column(String, nullable=True)
    issue_month: Mapped[str | None] = mapped_column(String, nullable=True)
    issue_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    entities: Mapped[list[str]] = mapped_column(JSONB, default=list)
    keywords: Mapped[list[str]] = mapped_column(JSONB, default=list)
    language: Mapped[str] = mapped_column(String, nullable=False, default="en")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    crawled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #
class ChunkORM(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    url: Mapped[str] = mapped_column(String, nullable=False, default="")
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    content_type: Mapped[str] = mapped_column(String, nullable=False, default="other")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    month: Mapped[str | None] = mapped_column(String, nullable=True)
    author: Mapped[str | None] = mapped_column(String, nullable=True)
    issue: Mapped[str | None] = mapped_column(String, nullable=True)
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSONB, default=list)
    entities: Mapped[list[str]] = mapped_column(JSONB, default=list)
    heading_path: Mapped[list[str]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Full-text index maintained by Postgres from ``text``.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(text,''))", persisted=True),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
        Index("ix_chunks_entities", "entities", postgresql_using="gin"),
    )


# --------------------------------------------------------------------------- #
# Knowledge graph
# --------------------------------------------------------------------------- #
class EntityORM(Base):
    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, default="")
    normalized_name: Mapped[str] = mapped_column(
        String, unique=True, index=True, nullable=False
    )
    type: Mapped[str] = mapped_column(String, nullable=False, default="other")
    aliases: Mapped[list[str]] = mapped_column(JSONB, default=list)
    mention_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class RelationORM(Base):
    __tablename__ = "relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    subject_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    predicate: Mapped[str] = mapped_column(String, nullable=False, default="related_to")
    object_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    object_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    source_url: Mapped[str] = mapped_column(String, nullable=False, default="")
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "subject_id",
            "predicate",
            "object_id",
            "source_url",
            name="uq_relation_edge",
        ),
    )


# --------------------------------------------------------------------------- #
# Feedback / analytics / jobs
# --------------------------------------------------------------------------- #
class FeedbackORM(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    message_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    rating: Mapped[str] = mapped_column(String, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class QueryLogORM(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False)
    message_id: Mapped[str] = mapped_column(String, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query: Mapped[str] = mapped_column(String, index=True, nullable=False)
    query_type: Mapped[str] = mapped_column(String, nullable=False, default="general")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    retrieved_documents: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cited_urls: Mapped[list[str]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


class CrawlJobORM(Base):
    __tablename__ = "crawl_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    pages_crawled: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pages_indexed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pages_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
