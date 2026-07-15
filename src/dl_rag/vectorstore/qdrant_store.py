"""Qdrant-backed dense vector store implementing the ``VectorStore`` protocol."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from dl_rag.logging_config import get_logger
from dl_rag.models.domain import Chunk, ChunkMetadata, RetrievedChunk
from dl_rag.models.enums import ContentType, RetrievalSource

logger = get_logger(__name__)

_UPSERT_BATCH = 128


def _coerce_content_type(value: str | None) -> ContentType:
    if not value:
        return ContentType.OTHER
    try:
        return ContentType(value)
    except ValueError:
        return ContentType.OTHER


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id derived from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _payload(chunk: Chunk) -> dict[str, Any]:
    meta = chunk.metadata
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "url": meta.url,
        "title": meta.title,
        "category": meta.category,
        "content_type": meta.content_type.value,
        "year": meta.year,
        "month": meta.month,
        "author": meta.author,
        "issue": meta.issue,
        "published_date": meta.published_date.isoformat()
        if meta.published_date
        else None,
        "tags": list(meta.tags),
        "entities": list(meta.entities),
        "heading_path": list(meta.heading_path),
    }


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    meta = ChunkMetadata(
        url=payload.get("url") or "",
        title=payload.get("title") or "",
        category=payload.get("category"),
        content_type=_coerce_content_type(payload.get("content_type")),
        year=payload.get("year"),
        month=payload.get("month"),
        author=payload.get("author"),
        issue=payload.get("issue"),
        published_date=_parse_date(payload.get("published_date")),
        tags=list(payload.get("tags") or []),
        entities=list(payload.get("entities") or []),
        heading_path=list(payload.get("heading_path") or []),
    )
    return Chunk(
        id=payload.get("chunk_id") or "",
        document_id=payload.get("document_id") or "",
        chunk_index=int(payload.get("chunk_index") or 0),
        text=payload.get("text") or "",
        token_count=int(payload.get("token_count") or 0),
        metadata=meta,
    )


def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    if not filters:
        return None
    must: list[models.FieldCondition] = []

    year_from = filters.get("year_from")
    year_to = filters.get("year_to")
    if year_from is not None or year_to is not None:
        must.append(
            models.FieldCondition(
                key="year",
                range=models.Range(gte=year_from, lte=year_to),
            )
        )

    content_types = filters.get("content_types")
    if content_types:
        must.append(
            models.FieldCondition(
                key="content_type", match=models.MatchAny(any=list(content_types))
            )
        )

    authors = filters.get("authors")
    if authors:
        must.append(
            models.FieldCondition(
                key="author", match=models.MatchAny(any=list(authors))
            )
        )

    tags = filters.get("tags")
    if tags:
        must.append(
            models.FieldCondition(key="tags", match=models.MatchAny(any=list(tags)))
        )

    entities = filters.get("entities")
    if entities:
        must.append(
            models.FieldCondition(
                key="entities", match=models.MatchAny(any=list(entities))
            )
        )

    if not must:
        return None
    return models.Filter(must=must)


class QdrantVectorStore:
    """Dense-vector persistence and ANN search over chunk embeddings."""

    def __init__(self, url: str, collection: str, api_key: str | None = None) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key)
        self._collection = collection

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    async def ensure_collection(self, dimension: int) -> None:
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )
            logger.info(
                "qdrant.collection.created",
                collection=self._collection,
                dimension=dimension,
            )

        # Payload indexes for fast filtering (idempotent: ignore if present).
        for field_name, schema in (
            ("year", models.PayloadSchemaType.INTEGER),
            ("content_type", models.PayloadSchemaType.KEYWORD),
            ("author", models.PayloadSchemaType.KEYWORD),
        ):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:  # noqa: BLE001 - index may already exist
                logger.debug(
                    "qdrant.payload_index.skip", field=field_name, error=str(exc)
                )

    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        points: list[models.PointStruct] = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning("qdrant.upsert.skip_no_embedding", chunk_id=chunk.id)
                continue
            points.append(
                models.PointStruct(
                    id=_point_id(chunk.id),
                    vector=list(chunk.embedding),
                    payload=_payload(chunk),
                )
            )

        if not points:
            return 0

        for start in range(0, len(points), _UPSERT_BATCH):
            batch = points[start : start + _UPSERT_BATCH]
            await self._client.upsert(collection_name=self._collection, points=batch)

        logger.debug("qdrant.upsert", count=len(points))
        return len(points)

    async def search(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        # qdrant-client >= 1.13 removed the classic `.search()`; `query_points`
        # is the unified API. Keep a fallback for older client versions.
        if hasattr(self._client, "query_points"):
            response = await self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                query_filter=_build_filter(filters),
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        else:  # pragma: no cover - legacy client path
            hits = await self._client.search(
                collection_name=self._collection,
                query_vector=list(vector),
                query_filter=_build_filter(filters),
                limit=top_k,
                with_payload=True,
            )
        retrieved: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            retrieved.append(
                RetrievedChunk(
                    chunk=_payload_to_chunk(payload),
                    score=float(hit.score),
                    sources=[RetrievalSource.DENSE],
                )
            )
        return retrieved

    async def delete_by_document(self, document_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    async def count(self) -> int:
        try:
            result = await self._client.count(
                collection_name=self._collection, exact=True
            )
            return int(result.count)
        except Exception as exc:  # noqa: BLE001 - missing collection → 0
            logger.warning("qdrant.count.failed", error=str(exc))
            return 0
