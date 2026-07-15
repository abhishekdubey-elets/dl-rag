"""Chunk persistence and Postgres full-text (sparse) search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import String, Text, and_, cast
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import ARRAY, TSQUERY
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql.selectable import Select

from dl_rag.db.orm import ChunkORM
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import Chunk, ChunkMetadata, RetrievedChunk
from dl_rag.models.enums import ContentType, RetrievalSource
from dl_rag.repositories.base import BaseRepository

logger = get_logger(__name__)

_CHUNK_UPDATE_COLS = (
    "document_id",
    "chunk_index",
    "text",
    "token_count",
    "url",
    "title",
    "category",
    "content_type",
    "year",
    "month",
    "author",
    "issue",
    "published_date",
    "tags",
    "entities",
    "heading_path",
)


def _coerce_content_type(value: str | None) -> ContentType:
    if not value:
        return ContentType.OTHER
    try:
        return ContentType(value)
    except ValueError:
        return ContentType.OTHER


def _chunk_to_row(chunk: Chunk) -> dict[str, Any]:
    meta = chunk.metadata
    return {
        "id": chunk.id,
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
        "published_date": meta.published_date,
        "tags": list(meta.tags),
        "entities": list(meta.entities),
        "heading_path": list(meta.heading_path),
    }


def _orm_to_chunk(row: ChunkORM) -> Chunk:
    meta = ChunkMetadata(
        url=row.url or "",
        title=row.title or "",
        category=row.category,
        content_type=_coerce_content_type(row.content_type),
        year=row.year,
        month=row.month,
        author=row.author,
        issue=row.issue,
        published_date=row.published_date,
        tags=list(row.tags or []),
        entities=list(row.entities or []),
        heading_path=list(row.heading_path or []),
    )
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        chunk_index=row.chunk_index,
        text=row.text or "",
        token_count=row.token_count or 0,
        metadata=meta,
    )


def _text_array(values: Sequence[str]) -> Any:
    """A ``text[]`` SQL literal usable as the RHS of ``jsonb_exists_any``."""
    return cast(list(values), ARRAY(String))


def _apply_filters(stmt: Select[Any], filters: dict[str, Any] | None) -> Select[Any]:
    """Apply the shared filter schema as ``WHERE`` conjuncts."""
    if not filters:
        return stmt
    conds: list[Any] = []

    year_from = filters.get("year_from")
    year_to = filters.get("year_to")
    if year_from is not None:
        conds.append(ChunkORM.year >= year_from)
    if year_to is not None:
        conds.append(ChunkORM.year <= year_to)

    content_types = filters.get("content_types")
    if content_types:
        conds.append(ChunkORM.content_type.in_(list(content_types)))

    authors = filters.get("authors")
    if authors:
        conds.append(ChunkORM.author.in_(list(authors)))

    tags = filters.get("tags")
    if tags:
        conds.append(func.jsonb_exists_any(ChunkORM.tags, _text_array(tags)))

    entities = filters.get("entities")
    if entities:
        conds.append(func.jsonb_exists_any(ChunkORM.entities, _text_array(entities)))

    if conds:
        stmt = stmt.where(and_(*conds))
    return stmt


class ChunkRepository(BaseRepository):
    """Bulk write + lexical retrieval for :class:`Chunk`."""

    async def bulk_upsert(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        rows = [_chunk_to_row(c) for c in chunks]
        stmt = pg_insert(ChunkORM)
        update_set = {c: getattr(stmt.excluded, c) for c in _CHUNK_UPDATE_COLS}
        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=update_set)
        await self.session.execute(stmt, rows)
        logger.debug("chunks.bulk_upsert", count=len(rows))
        return len(rows)

    async def delete_by_document(self, document_id: str) -> None:
        await self.session.execute(
            sa_delete(ChunkORM).where(ChunkORM.document_id == document_id)
        )

    async def get_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids:
            return []
        stmt = select(ChunkORM).where(ChunkORM.id.in_(list(ids)))
        rows = (await self.session.execute(stmt)).scalars().all()
        by_id = {row.id: _orm_to_chunk(row) for row in rows}
        # Preserve the caller's ordering.
        return [by_id[i] for i in ids if i in by_id]

    async def count(self) -> int:
        stmt = select(func.count()).select_from(ChunkORM)
        return int((await self.session.execute(stmt)).scalar_one())

    async def fts_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Two-pass lexical search.

        Pass 1 uses ``websearch_to_tsquery`` (AND semantics — precise, but a
        natural-language question like "What has digitalLEARNING covered about
        IIT Madras recently?" requires *every* term and often matches nothing).
        When it yields no rows, pass 2 relaxes to OR semantics by rewriting the
        ``plainto_tsquery`` conjunction, so the strongest term matches still
        surface and ``ts_rank_cd`` orders them sensibly.
        """
        if not query or not query.strip():
            return []

        strict = func.websearch_to_tsquery("english", query)
        results = await self._run_fts(strict, top_k, filters)
        if len(results) >= top_k:
            return results

        # OR-relaxed top-up: strict AND-matching often starves on natural
        # questions (few chunks contain *every* term). Merge relaxed matches
        # after the strict ones until top_k is reached.
        relaxed = func.cast(
            func.replace(
                func.cast(func.plainto_tsquery("english", query), Text), "&", "|"
            ),
            TSQUERY,
        )
        seen = {rc.chunk.id for rc in results}
        for rc in await self._run_fts(relaxed, top_k, filters):
            if rc.chunk.id not in seen:
                results.append(rc)
                seen.add(rc.chunk.id)
            if len(results) >= top_k:
                break
        return results

    async def latest_by_terms(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        phrase: str | None = None,
    ) -> list[RetrievedChunk]:
        """Newest chunks matching the query — ordered by date, not rank.

        Serves "next / latest / upcoming" questions, which are argmax-by-date:
        the correct evidence is the most recent on-topic coverage, which pure
        relevance ranking systematically buries under older, wordier articles.

        When ``phrase`` is given (typically the canonical entity name, e.g.
        "World Education Summit") chunks must contain it as a *phrase* — a much
        stronger topicality guarantee than OR-matching scattered query terms,
        which lets any fresh-but-off-topic chunk win the date sort.
        """
        if not query or not query.strip():
            return []
        if phrase and phrase.strip():
            tsq = func.phraseto_tsquery("english", phrase)
        else:
            tsq = func.cast(
                func.replace(
                    func.cast(func.plainto_tsquery("english", query), Text), "&", "|"
                ),
                TSQUERY,
            )
        rank = func.ts_rank_cd(ChunkORM.search_vector, tsq).label("rank")
        stmt = select(ChunkORM, rank).where(
            ChunkORM.search_vector.bool_op("@@")(tsq),
            ChunkORM.published_date.is_not(None),
        )
        stmt = _apply_filters(stmt, filters)
        order = [ChunkORM.published_date.desc(), rank.desc()]
        if phrase and phrase.strip():
            # Articles ABOUT the entity (title mentions it) beat articles that
            # merely mention it in passing (e.g. site-wide promo paragraphs).
            order.insert(0, ChunkORM.title.ilike(f"%{phrase}%").desc())
        stmt = stmt.order_by(*order).limit(top_k)

        result = await self.session.execute(stmt)
        return [
            RetrievedChunk(
                chunk=_orm_to_chunk(orm_row),
                score=float(rank_value or 0.0),
                sources=[RetrievalSource.SPARSE],
            )
            for orm_row, rank_value in result.all()
        ]

    async def _run_fts(
        self,
        tsquery: Any,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        rank = func.ts_rank_cd(ChunkORM.search_vector, tsquery).label("rank")
        stmt = select(ChunkORM, rank).where(
            ChunkORM.search_vector.bool_op("@@")(tsquery)
        )
        stmt = _apply_filters(stmt, filters)
        stmt = stmt.order_by(rank.desc()).limit(top_k)

        result = await self.session.execute(stmt)
        retrieved: list[RetrievedChunk] = []
        for orm_row, rank_value in result.all():
            retrieved.append(
                RetrievedChunk(
                    chunk=_orm_to_chunk(orm_row),
                    score=float(rank_value or 0.0),
                    sources=[RetrievalSource.SPARSE],
                )
            )
        return retrieved
