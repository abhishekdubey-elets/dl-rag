"""Document persistence and aggregate statistics."""

from __future__ import annotations

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, nullslast, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dl_rag.db.orm import ChunkORM, DocumentORM
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import SourceDocument
from dl_rag.models.enums import ContentType
from dl_rag.repositories.base import BaseRepository

logger = get_logger(__name__)


def _coerce_content_type(value: str | None) -> ContentType:
    if not value:
        return ContentType.OTHER
    try:
        return ContentType(value)
    except ValueError:
        return ContentType.OTHER


def _to_domain(row: DocumentORM) -> SourceDocument:
    return SourceDocument(
        id=row.id,
        url=row.url,
        title=row.title,
        subtitle=row.subtitle,
        author=row.author,
        published_date=row.published_date,
        updated_date=row.updated_date,
        category=row.category,
        content_type=_coerce_content_type(row.content_type),
        tags=list(row.tags or []),
        content_markdown=row.content_markdown or "",
        featured_image=row.featured_image,
        issue_name=row.issue_name,
        issue_month=row.issue_month,
        issue_year=row.issue_year,
        entities=list(row.entities or []),
        keywords=list(row.keywords or []),
        language=row.language or "en",
        metadata=dict(row.metadata_json or {}),
        crawled_at=row.crawled_at,
        content_hash=row.content_hash,
    )


class DocumentRepository(BaseRepository):
    """CRUD + counters for :class:`SourceDocument`."""

    async def upsert(self, doc: SourceDocument) -> None:
        values = {
            "id": doc.id,
            "url": doc.url,
            "title": doc.title,
            "subtitle": doc.subtitle,
            "author": doc.author,
            "published_date": doc.published_date,
            "updated_date": doc.updated_date,
            "category": doc.category,
            "content_type": doc.content_type.value,
            "tags": list(doc.tags),
            "content_markdown": doc.content_markdown,
            "featured_image": doc.featured_image,
            "issue_name": doc.issue_name,
            "issue_month": doc.issue_month,
            "issue_year": doc.issue_year,
            "entities": list(doc.entities),
            "keywords": list(doc.keywords),
            "language": doc.language,
            "metadata_json": dict(doc.metadata),
            "crawled_at": doc.crawled_at,
            "content_hash": doc.content_hash or doc.compute_hash(),
            "word_count": doc.word_count,
        }
        stmt = pg_insert(DocumentORM).values(**values)
        update_set = {k: getattr(stmt.excluded, k) for k in values if k != "id"}
        update_set["updated_at"] = func.now()
        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=update_set)
        await self.session.execute(stmt)

    async def get(self, doc_id: str) -> SourceDocument | None:
        row = await self.session.get(DocumentORM, doc_id)
        return _to_domain(row) if row is not None else None

    async def get_by_url(self, url: str) -> SourceDocument | None:
        stmt = select(DocumentORM).where(DocumentORM.url == url)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        return _to_domain(row) if row is not None else None

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        content_type: str | None = None,
        year: int | None = None,
    ) -> list[SourceDocument]:
        stmt = select(DocumentORM)
        if content_type:
            stmt = stmt.where(DocumentORM.content_type == content_type)
        if year is not None:
            stmt = stmt.where(
                or_(
                    func.extract("year", DocumentORM.published_date) == year,
                    DocumentORM.issue_year == year,
                )
            )
        stmt = (
            stmt.order_by(
                nullslast(DocumentORM.published_date.desc()),
                DocumentORM.created_at.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [_to_domain(r) for r in rows]

    async def count(self) -> int:
        stmt = select(func.count()).select_from(DocumentORM)
        return int((await self.session.execute(stmt)).scalar_one())

    async def count_by_type(self) -> dict[str, int]:
        stmt = select(DocumentORM.content_type, func.count()).group_by(
            DocumentORM.content_type
        )
        rows = (await self.session.execute(stmt)).all()
        return {str(ct): int(c) for ct, c in rows}

    async def count_by_year(self) -> dict[str, int]:
        year_expr = func.coalesce(
            func.extract("year", DocumentORM.published_date),
            DocumentORM.issue_year,
        )
        stmt = select(year_expr.label("y"), func.count()).group_by(year_expr)
        rows = (await self.session.execute(stmt)).all()
        return {str(int(y)): int(c) for y, c in rows if y is not None}

    async def chunk_count(self, doc_id: str) -> int:
        stmt = (
            select(func.count())
            .select_from(ChunkORM)
            .where(ChunkORM.document_id == doc_id)
        )
        return int((await self.session.execute(stmt)).scalar_one())

    async def delete(self, doc_id: str) -> None:
        await self.session.execute(
            sa_delete(DocumentORM).where(DocumentORM.id == doc_id)
        )
