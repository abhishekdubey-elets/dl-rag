"""Knowledge-graph persistence: entities, relations, and BFS expansion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import orjson
from sqlalchemy import func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dl_rag.db.orm import EntityORM, RelationORM
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import Entity, Relation
from dl_rag.models.enums import RelationType
from dl_rag.repositories.base import BaseRepository

logger = get_logger(__name__)

# Upsert an entity: accumulate mention_count and union aliases (deduplicated).
_UPSERT_ENTITY_SQL = text(
    """
    INSERT INTO entities (id, name, normalized_name, type, aliases, mention_count)
    VALUES (:id, :name, :normalized_name, :type, CAST(:aliases AS jsonb), :mention_count)
    ON CONFLICT (id) DO UPDATE SET
        mention_count = entities.mention_count + EXCLUDED.mention_count,
        name = COALESCE(NULLIF(entities.name, ''), EXCLUDED.name),
        type = CASE
                   WHEN entities.type = 'other' THEN EXCLUDED.type
                   ELSE entities.type
               END,
        aliases = (
            SELECT COALESCE(jsonb_agg(DISTINCT elem), '[]'::jsonb)
            FROM (
                SELECT jsonb_array_elements_text(entities.aliases) AS elem
                UNION
                SELECT jsonb_array_elements_text(EXCLUDED.aliases) AS elem
            ) merged
        )
    """
)


def _coerce_relation_type(value: str | None) -> RelationType:
    if not value:
        return RelationType.RELATED_TO
    try:
        return RelationType(value)
    except ValueError:
        return RelationType.RELATED_TO


def _to_relation(row: RelationORM) -> Relation:
    return Relation(
        subject_id=row.subject_id,
        subject_name=row.subject_name,
        predicate=_coerce_relation_type(row.predicate),
        object_id=row.object_id,
        object_name=row.object_name,
        source_url=row.source_url,
        evidence=row.evidence,
        confidence=row.confidence,
    )


class KGRepository(BaseRepository):
    """Entity/relation upserts and graph traversal."""

    async def upsert_entities(self, entities: Sequence[Entity]) -> None:
        if not entities:
            return
        rows = [
            {
                "id": ent.id,
                "name": ent.name,
                "normalized_name": ent.normalized_name,
                "type": ent.type.value,
                "aliases": orjson.dumps(list(ent.aliases)).decode("utf-8"),
                "mention_count": max(1, ent.mention_count),
            }
            for ent in entities
        ]
        await self.session.execute(_UPSERT_ENTITY_SQL, rows)
        logger.debug("kg.entities.upsert", count=len(rows))

    async def upsert_relations(self, relations: Sequence[Relation]) -> None:
        if not relations:
            return
        rows = [
            {
                "subject_id": rel.subject_id,
                "subject_name": rel.subject_name,
                "predicate": rel.predicate.value,
                "object_id": rel.object_id,
                "object_name": rel.object_name,
                "source_url": rel.source_url,
                "evidence": rel.evidence,
                "confidence": rel.confidence,
            }
            for rel in relations
        ]
        stmt = pg_insert(RelationORM).on_conflict_do_nothing(
            index_elements=["subject_id", "predicate", "object_id", "source_url"]
        )
        await self.session.execute(stmt, rows)
        logger.debug("kg.relations.upsert", count=len(rows))

    async def neighbors(self, entity_name: str) -> list[Relation]:
        norm = Entity.normalize(entity_name)
        ent_id = Entity.make_id(entity_name)
        stmt = select(RelationORM).where(
            or_(
                RelationORM.subject_id == ent_id,
                RelationORM.object_id == ent_id,
                func.lower(RelationORM.subject_name) == norm,
                func.lower(RelationORM.object_name) == norm,
            )
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [_to_relation(r) for r in rows]

    async def expand(self, entity_names: Sequence[str], hops: int = 1) -> list[str]:
        if not entity_names or hops < 1:
            return []

        seeds_norm = {Entity.normalize(n) for n in entity_names}
        frontier_ids = {Entity.make_id(n) for n in entity_names}
        frontier_norm = set(seeds_norm)
        visited_norm = set(seeds_norm)

        result: list[str] = []
        result_seen: set[str] = set()

        for _ in range(hops):
            conds: list[Any] = []
            if frontier_ids:
                conds.append(RelationORM.subject_id.in_(frontier_ids))
                conds.append(RelationORM.object_id.in_(frontier_ids))
            if frontier_norm:
                conds.append(func.lower(RelationORM.subject_name).in_(frontier_norm))
                conds.append(func.lower(RelationORM.object_name).in_(frontier_norm))
            if not conds:
                break

            stmt = select(RelationORM).where(or_(*conds))
            rows = (await self.session.execute(stmt)).scalars().all()

            next_ids: set[str] = set()
            next_norm: set[str] = set()
            for row in rows:
                for name, eid in (
                    (row.subject_name, row.subject_id),
                    (row.object_name, row.object_id),
                ):
                    norm = Entity.normalize(name)
                    if norm in visited_norm:
                        continue
                    if norm not in result_seen:
                        result_seen.add(norm)
                        result.append(name)
                    next_ids.add(eid)
                    next_norm.add(norm)

            visited_norm |= next_norm
            frontier_ids = next_ids
            frontier_norm = next_norm
            if not frontier_ids and not frontier_norm:
                break

        return result

    async def count_entities(self) -> int:
        stmt = select(func.count()).select_from(EntityORM)
        return int((await self.session.execute(stmt)).scalar_one())

    async def count_relations(self) -> int:
        stmt = select(func.count()).select_from(RelationORM)
        return int((await self.session.execute(stmt)).scalar_one())
