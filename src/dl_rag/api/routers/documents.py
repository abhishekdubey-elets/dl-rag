"""GET /api/document/{id} — fetch a source document + metadata."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from dl_rag.api.deps import get_db
from dl_rag.api.security import require_api_key
from dl_rag.db.database import Database
from dl_rag.exceptions import NotFoundError
from dl_rag.models.api import DocumentResponse
from dl_rag.repositories.document_repository import DocumentRepository

router = APIRouter(tags=["documents"], dependencies=[Depends(require_api_key)])


@router.get("/document/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    include_content: bool = Query(default=False, description="Include full markdown body."),
    db: Database = Depends(get_db),
) -> DocumentResponse:
    async with db.session() as session:
        repo = DocumentRepository(session)
        doc = await repo.get(doc_id)
        if doc is None:
            raise NotFoundError(f"No document with id '{doc_id}'.")
        chunk_count = await repo.chunk_count(doc_id)

    return DocumentResponse(
        id=doc.id,
        url=doc.url,
        title=doc.title,
        subtitle=doc.subtitle,
        author=doc.author,
        published_date=doc.published_date,
        content_type=doc.content_type,
        category=doc.category,
        tags=doc.tags,
        issue_name=doc.issue_name,
        issue_year=doc.issue_year,
        entities=doc.entities,
        keywords=doc.keywords,
        word_count=doc.word_count,
        chunk_count=chunk_count,
        content_markdown=doc.content_markdown if include_content else None,
    )
