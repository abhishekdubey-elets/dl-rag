"""`dl-kg-extract` — (re)build the knowledge graph from already-ingested documents.

Re-runs entity/keyword/relation extraction over the stored corpus — no crawling,
no re-embedding. Use after extending the gazetteer or relation triggers so the
graph reflects the new vocabulary (e.g. the WES event entity and the person-gated
``spoke_at`` predicate).

Examples:
    poetry run dl-kg-extract --rebuild           # wipe + rebuild the whole graph
    poetry run dl-kg-extract --limit 200         # trial pass on 200 documents
    poetry run dl-kg-extract --no-spacy          # gazetteer only (no person NER)
"""

from __future__ import annotations

import argparse
import asyncio
import time

from sqlalchemy import func, select, text, update

from dl_rag.config import get_settings
from dl_rag.db.database import Database
from dl_rag.db.orm import DocumentORM, EntityORM, RelationORM
from dl_rag.ingestion.entities.extractor import EntityExtractor
from dl_rag.knowledge_graph.builder import KnowledgeGraphBuilder
from dl_rag.adapters.knowledge_graph import PostgresKnowledgeGraph
from dl_rag.logging_config import configure_logging, get_logger
from dl_rag.models.domain import SourceDocument

logger = get_logger(__name__)

_BATCH = 200


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild the knowledge graph from stored documents.")
    p.add_argument("--rebuild", action="store_true",
                   help="TRUNCATE entities+relations first (clean rebuild).")
    p.add_argument("--limit", type=int, default=None, help="Only process N documents (trial).")
    p.add_argument("--no-spacy", action="store_true", help="Disable person NER.")
    p.add_argument("--update-doc-entities", action="store_true", default=True,
                   help="Also refresh documents.entities (default on).")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    db = Database(settings.postgres_dsn)
    extractor = EntityExtractor(settings, use_spacy=not args.no_spacy)
    builder = KnowledgeGraphBuilder(PostgresKnowledgeGraph(db.sessionmaker))

    if args.rebuild:
        async with db.session() as session:
            await session.execute(text("TRUNCATE TABLE relations"))
            await session.execute(text("TRUNCATE TABLE entities"))
        logger.info("kg.rebuild.truncated")

    async with db.session() as session:
        total = (
            await session.execute(select(func.count()).select_from(DocumentORM))
        ).scalar_one()
    target = min(total, args.limit) if args.limit else total
    logger.info("kg.extract.start", documents=target, spacy=not args.no_spacy)

    processed = n_entities = n_relations = n_speakers = 0
    started = time.perf_counter()
    offset = 0

    while processed < target:
        async with db.session() as session:
            rows = (
                await session.execute(
                    select(
                        DocumentORM.id, DocumentORM.url,
                        DocumentORM.title, DocumentORM.content_markdown,
                    )
                    .order_by(DocumentORM.id)
                    .offset(offset)
                    .limit(min(_BATCH, target - processed))
                )
            ).all()
        if not rows:
            break
        offset += len(rows)

        for doc_id, url, title, content in rows:
            doc = SourceDocument(id=doc_id, url=url, title=title or "",
                                 content_markdown=content or "")
            try:
                entities, _keywords, relations = extractor.extract_from_document(doc)
            except Exception as exc:  # noqa: BLE001 - one bad doc must not kill the run
                logger.warning("kg.extract.doc_failed", url=url, error=str(exc))
                continue
            if entities or relations:
                await builder.add(entities, relations)
            if args.update_doc_entities and entities:
                names = [e.name for e in entities][:50]
                async with db.session() as session:
                    await session.execute(
                        update(DocumentORM)
                        .where(DocumentORM.id == doc_id)
                        .values(entities=names)
                    )
            n_entities += len(entities)
            n_relations += len(relations)
            n_speakers += sum(1 for r in relations if r.predicate.value == "spoke_at")
            processed += 1

        elapsed = time.perf_counter() - started
        rate = processed / elapsed if elapsed else 0.0
        logger.info(
            "kg.extract.progress",
            processed=processed,
            total=target,
            rate_per_s=round(rate, 1),
            eta_min=round((target - processed) / rate / 60, 1) if rate else None,
            relations=n_relations,
            spoke_at=n_speakers,
        )

    async with db.session() as session:
        graph_entities = (
            await session.execute(select(func.count()).select_from(EntityORM))
        ).scalar_one()
        graph_relations = (
            await session.execute(select(func.count()).select_from(RelationORM))
        ).scalar_one()

    print("\nKnowledge-graph extraction complete:")
    print(f"  documents processed : {processed}")
    print(f"  entity mentions     : {n_entities}")
    print(f"  relations extracted : {n_relations} (spoke_at: {n_speakers})")
    print(f"  graph now holds     : {graph_entities} entities, {graph_relations} relations")
    await db.dispose()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
