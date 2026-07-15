"""Cross-encoder reranking (plus a scoring-only no-op fallback).

The heavy ``sentence_transformers`` import and model load are deferred until the
first ``rerank`` call and executed off the event loop via ``asyncio.to_thread``
so importing this module stays cheap and the API process starts fast.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from dl_rag.logging_config import get_logger
from dl_rag.models.domain import RetrievedChunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Rerank candidates with a ``sentence_transformers`` CrossEncoder."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        """Import + construct the CrossEncoder lazily (blocking; run in thread)."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "retrieval.reranker.load",
                model=self._model_name,
                device=self._device,
            )
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Score each candidate against ``query`` and return the top ``top_k``."""
        items = list(candidates)
        if not items or top_k <= 0:
            return []

        pairs = [(query, item.chunk.text) for item in items]
        model = await asyncio.to_thread(self._load)
        scores = await asyncio.to_thread(model.predict, pairs)

        for item, score in zip(items, scores, strict=False):
            item.rerank_score = float(score)

        items.sort(
            key=lambda rc: rc.rerank_score if rc.rerank_score is not None else 0.0,
            reverse=True,
        )
        logger.debug(
            "retrieval.reranker.done",
            candidates=len(items),
            top_k=top_k,
        )
        return items[:top_k]


class NoopReranker:
    """Fallback reranker: sort by existing ``.score`` and mirror it to
    ``rerank_score``. Used when cross-encoder reranking is disabled or the model
    is unavailable, keeping the downstream contract (a populated
    ``rerank_score``) intact.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        items = list(candidates)
        if not items or top_k <= 0:
            return []

        items.sort(key=lambda rc: rc.score, reverse=True)
        for item in items:
            item.rerank_score = float(item.score)

        return items[:top_k]
