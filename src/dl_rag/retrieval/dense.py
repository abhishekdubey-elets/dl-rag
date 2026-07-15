"""Dense (embedding / ANN) retrieval stage.

Thin orchestration over the injected :class:`Embedder` and :class:`VectorStore`
Protocols: embed the query once, run an approximate-nearest-neighbour search,
and stamp positional ``dense_rank`` provenance onto every result.
"""

from __future__ import annotations

from typing import Any

from dl_rag.logging_config import get_logger
from dl_rag.models.domain import RetrievedChunk
from dl_rag.models.enums import RetrievalSource
from dl_rag.protocols import Embedder, VectorStore

logger = get_logger(__name__)


class DenseRetriever:
    """Embed a query and search the vector store for nearest chunks."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks ranked by dense similarity.

        Each result gets its 1-based position recorded on ``dense_rank`` and
        :data:`RetrievalSource.DENSE` appended to ``sources`` (deduplicated).
        """
        if top_k <= 0:
            return []

        vector = await self._embedder.embed_query(query)
        results = await self._vector_store.search(vector, top_k, filters)

        for position, item in enumerate(results):
            item.dense_rank = position + 1
            if RetrievalSource.DENSE not in item.sources:
                item.sources.append(RetrievalSource.DENSE)

        logger.debug(
            "retrieval.dense.done",
            query_chars=len(query),
            top_k=top_k,
            results=len(results),
            filtered=filters is not None,
        )
        return results
