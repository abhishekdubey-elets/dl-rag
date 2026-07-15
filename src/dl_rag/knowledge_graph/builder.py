"""KnowledgeGraphBuilder — persist extracted entities/relations into the graph.

Entity extraction itself lives in ``ingestion.entities.EntityExtractor`` (it runs
per-document during ingestion). This builder handles cross-mention merging within
a batch and delegates persistence to any :class:`KnowledgeGraph` adapter.
"""

from __future__ import annotations

from collections.abc import Sequence

from dl_rag.logging_config import get_logger
from dl_rag.models.domain import Entity, Relation
from dl_rag.protocols import KnowledgeGraph

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self._kg = knowledge_graph

    @staticmethod
    def _merge_entities(entities: Sequence[Entity]) -> list[Entity]:
        """Collapse duplicate mentions (same normalized name) within a batch."""
        merged: dict[str, Entity] = {}
        for ent in entities:
            key = ent.normalized_name
            existing = merged.get(key)
            if existing is None:
                merged[key] = ent.model_copy(deep=True)
                continue
            existing.mention_count += max(1, ent.mention_count)
            for alias in ent.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            # Upgrade an "other"-typed node if a more specific type appears.
            if existing.type.value == "other" and ent.type.value != "other":
                existing.type = ent.type
        return list(merged.values())

    async def add(
        self, entities: Sequence[Entity], relations: Sequence[Relation]
    ) -> tuple[int, int]:
        """Persist a batch of entities + relations. Returns (n_entities, n_relations)."""
        merged = self._merge_entities(entities)
        if merged:
            await self._kg.add_entities(merged)
        if relations:
            await self._kg.add_relations(list(relations))
        logger.debug("kg.persisted", entities=len(merged), relations=len(relations))
        return len(merged), len(relations)
