"""Shared fixtures: hermetic settings, fake collaborators, chunk factories.

Tests never read `.env` (``_env_file=None``) and never touch the network —
fakes stand in for the LLM, cache, embedder, and stores.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dl_rag.config import Settings  # noqa: E402
from dl_rag.models.domain import Chunk, ChunkMetadata, RetrievedChunk  # noqa: E402
from dl_rag.models.enums import ContentType, RetrievalSource  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    """Hermetic settings — defaults only, no .env, no OS surprises."""
    return Settings(_env_file=None)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Deterministic LLMClient double."""

    def __init__(self, reply: str = "Answer citing sources [1][2].") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, *, temperature=None, max_tokens=None,
                       response_format=None):
        self.calls.append(messages)
        return self.reply, {"prompt_tokens": 100, "completion_tokens": 50,
                            "total_tokens": 150}

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.calls.append(messages)
        for part in self.reply.split(" "):
            yield part + " "


class FailingLLM:
    async def complete(self, messages, **kwargs):
        raise RuntimeError("llm down")

    async def stream(self, messages, **kwargs):
        raise RuntimeError("llm down")
        yield  # pragma: no cover


class FakeCache:
    """In-memory Cache double."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.counters: dict[str, int] = {}

    async def get_json(self, key: str):
        return self.store.get(key)

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        self.store[key] = value

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)

    async def incr_window(self, key: str, window_seconds: int) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_cache() -> FakeCache:
    return FakeCache()


# --------------------------------------------------------------------------- #
# Chunk factories
# --------------------------------------------------------------------------- #
def make_chunk(
    i: int = 0,
    *,
    doc_id: str = "doc1",
    url: str | None = None,
    title: str | None = None,
    text: str | None = None,
    year: int | None = 2022,
    content_type: ContentType = ContentType.POLICY,
) -> Chunk:
    return Chunk(
        id=Chunk.make_id(doc_id, i),
        document_id=doc_id,
        chunk_index=i,
        text=text or f"Chunk {i}: NEP implementation progressed across states in {year}.",
        token_count=20,
        metadata=ChunkMetadata(
            url=url or f"https://digitallearning.eletsonline.com/a{i}/",
            title=title or f"Article {i}",
            content_type=content_type,
            year=year,
            month="June",
            author="Ravi Kumar",
            published_date=date(year or 2022, 6, 14),
            tags=["NEP"],
            entities=["NEP 2020"],
        ),
    )


def make_retrieved(
    i: int = 0,
    *,
    score: float = 0.5,
    rerank: float | None = 5.0,
    source: RetrievalSource = RetrievalSource.DENSE,
    **chunk_kwargs,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(i, **chunk_kwargs),
        score=score,
        rerank_score=rerank,
        sources=[source],
    )


@pytest.fixture
def retrieved_chunks() -> list[RetrievedChunk]:
    return [make_retrieved(i) for i in range(3)]
