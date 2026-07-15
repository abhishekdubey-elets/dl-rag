"""Structural interfaces (typing.Protocol) for every swappable component.

Depending on these Protocols — not concrete classes — is what keeps the
architecture SOLID: services accept the interface, DI supplies the
implementation, and tests supply fakes. Every concrete adapter in the codebase
is expected to satisfy the relevant Protocol here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from dl_rag.models.domain import (
    Chunk,
    Entity,
    QueryAnalysis,
    Relation,
    RetrievedChunk,
)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors."""

    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
@runtime_checkable
class VectorStore(Protocol):
    """Dense-vector persistence + ANN search."""

    async def ensure_collection(self, dimension: int) -> None: ...

    async def upsert(self, chunks: Sequence[Chunk]) -> int: ...

    async def search(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    async def delete_by_document(self, document_id: str) -> None: ...

    async def count(self) -> int: ...


# --------------------------------------------------------------------------- #
# Sparse (keyword / BM25-style) retrieval
# --------------------------------------------------------------------------- #
@runtime_checkable
class SparseRetriever(Protocol):
    """Lexical retrieval — Postgres full-text search in the default impl."""

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranking of candidate chunks against the query."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]: ...


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
@runtime_checkable
class LLMClient(Protocol):
    """OpenAI-compatible chat completion client."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Return (text, usage) where usage has prompt/completion/total tokens."""
        ...

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
@runtime_checkable
class Cache(Protocol):
    async def get_json(self, key: str) -> Any | None: ...

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def incr_window(self, key: str, window_seconds: int) -> int:
        """Increment a fixed-window counter and return the current count."""
        ...


# --------------------------------------------------------------------------- #
# Knowledge graph
# --------------------------------------------------------------------------- #
@runtime_checkable
class KnowledgeGraph(Protocol):
    async def add_entities(self, entities: Sequence[Entity]) -> None: ...

    async def add_relations(self, relations: Sequence[Relation]) -> None: ...

    async def expand(self, entity_names: Sequence[str], hops: int = 1) -> list[str]:
        """Return related entity names reachable within ``hops`` edges."""
        ...

    async def neighbors(self, entity_name: str) -> list[Relation]: ...


# --------------------------------------------------------------------------- #
# Query understanding
# --------------------------------------------------------------------------- #
@runtime_checkable
class QueryAnalyzer(Protocol):
    async def analyze(self, query: str) -> QueryAnalysis: ...
