"""Dense text embedder backed by sentence-transformers.

The heavy ML dependency is imported lazily inside :meth:`_load` so importing
this module (e.g. for wiring the DI container) stays cheap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from dl_rag.logging_config import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbedder:
    """Encodes text into L2-normalised dense vectors.

    Normalised embeddings mean cosine similarity == dot product, which matches
    the COSINE distance configured on the Qdrant collection.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._model: Any | None = None
        self._dimension: int | None = None

    def _load(self) -> Any:
        """Load and cache the model + its dimension (blocking, CPU/GPU-bound)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "embedder.loading", model=self._model_name, device=self._device
            )
            model = SentenceTransformer(self._model_name, device=self._device)
            self._model = model
            self._dimension = int(model.get_sentence_embedding_dimension())
            logger.info(
                "embedder.loaded", model=self._model_name, dimension=self._dimension
            )
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        assert self._dimension is not None
        return self._dimension

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        prefixed = f"{self._query_prefix}{text}" if self._query_prefix else text
        vectors = await asyncio.to_thread(self._encode, [prefixed])
        return vectors[0] if vectors else []
