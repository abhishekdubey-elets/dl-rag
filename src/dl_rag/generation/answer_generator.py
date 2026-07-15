"""Answer generation: orchestrates prompt assembly, the LLM call, citation
reconciliation and a deterministic confidence score into a
:class:`GeneratedAnswer`.

Supports both the buffered path (:meth:`AnswerGenerator.generate`) and the
streaming path (:meth:`AnswerGenerator.stream_tokens` +
:meth:`AnswerGenerator.finalize`).
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING

from dl_rag.constants import NO_EVIDENCE_MESSAGE
from dl_rag.generation.citations import (
    build_citations,
    extract_cited_indices,
    used_citations,
)
from dl_rag.generation.prompts import build_messages
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import GeneratedAnswer
from dl_rag.models.enums import ConfidenceBand

if TYPE_CHECKING:
    from dl_rag.config import Settings
    from dl_rag.models.domain import QueryAnalysis, RetrievedChunk
    from dl_rag.protocols import LLMClient

logger = get_logger(__name__)

# Confidence weighting (documented, deterministic).
_RELEVANCE_WEIGHT = 0.6
_COVERAGE_WEIGHT = 0.4
# Citing this many distinct sources is treated as full citation coverage.
_COVERAGE_TARGET = 3
# Band thresholds.
_HIGH_THRESHOLD = 0.66
_MEDIUM_THRESHOLD = 0.4


def _sigmoid(value: float) -> float:
    """Numerically-stable logistic mapping unbounded logits to ``(0, 1)``."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _minmax_mean(values: Sequence[float]) -> float:
    """Mean of min-max-normalised values; neutral 0.5 when all are equal."""
    if not values:
        return 0.0
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return 0.5
    return _mean([(value - low) / (high - low) for value in values])


class AnswerGenerator:
    """Builds grounded, cited answers from retrieved chunks."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    # ------------------------------------------------------------------ #
    # Buffered generation
    # ------------------------------------------------------------------ #
    async def generate(
        self,
        analysis: QueryAnalysis,
        chunks: Sequence[RetrievedChunk],
        history_summary: str | None = None,
        history_turns: list[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        """Generate a full answer in one shot."""
        if not chunks:
            return self._no_evidence_answer(analysis)

        messages = build_messages(analysis, chunks, history_summary, history_turns)
        text, usage = await self._llm.complete(messages)
        return self._assemble(analysis, chunks, text, usage)

    # ------------------------------------------------------------------ #
    # Streaming generation
    # ------------------------------------------------------------------ #
    async def stream_tokens(
        self,
        analysis: QueryAnalysis,
        chunks: Sequence[RetrievedChunk],
        history_summary: str | None = None,
        history_turns: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Yield answer tokens as they arrive from the LLM."""
        if not chunks:
            yield NO_EVIDENCE_MESSAGE
            return

        messages = build_messages(analysis, chunks, history_summary, history_turns)
        async for delta in self._llm.stream(messages):
            yield delta

    def finalize(
        self,
        analysis: QueryAnalysis,
        chunks: Sequence[RetrievedChunk],
        full_text: str,
        usage: dict[str, int] | None = None,
    ) -> GeneratedAnswer:
        """Assemble the final answer object from accumulated streamed text."""
        if not chunks:
            return self._no_evidence_answer(analysis)
        return self._assemble(analysis, chunks, full_text, usage)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _assemble(
        self,
        analysis: QueryAnalysis,
        chunks: Sequence[RetrievedChunk],
        text: str,
        usage: dict[str, int] | None,
    ) -> GeneratedAnswer:
        citations = build_citations(chunks)
        cited = used_citations(citations, text)
        grounded = not self._is_no_evidence(text)
        confidence = self._confidence(analysis, chunks, text)
        band = self._band(confidence)
        usage = usage or {}
        return GeneratedAnswer(
            answer=text,
            citations=cited,
            query_type=analysis.query_type,
            confidence=confidence,
            confidence_band=band,
            retrieved_documents=len(chunks),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            grounded=grounded,
        )

    def _no_evidence_answer(self, analysis: QueryAnalysis) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=NO_EVIDENCE_MESSAGE,
            citations=[],
            query_type=analysis.query_type,
            confidence=0.0,
            confidence_band=ConfidenceBand.LOW,
            retrieved_documents=0,
            prompt_tokens=0,
            completion_tokens=0,
            grounded=False,
        )

    @staticmethod
    def _is_no_evidence(text: str) -> bool:
        if not text or not text.strip():
            return True
        return NO_EVIDENCE_MESSAGE.strip().lower() in text.strip().lower()

    @staticmethod
    def _band(confidence: float) -> ConfidenceBand:
        if confidence >= _HIGH_THRESHOLD:
            return ConfidenceBand.HIGH
        if confidence >= _MEDIUM_THRESHOLD:
            return ConfidenceBand.MEDIUM
        return ConfidenceBand.LOW

    def _confidence(
        self,
        analysis: QueryAnalysis,
        chunks: Sequence[RetrievedChunk],
        answer_text: str,
    ) -> float:
        """Deterministic 0..1 confidence.

        Combines three signals:

        * **relevance** — the mean reranker score over the top-k chunks, squashed
          through a logistic so unbounded cross-encoder logits land in ``(0, 1)``.
          When no rerank scores are present, falls back to the min-max-normalised
          mean of the fused retrieval scores.
        * **coverage** — how many distinct sources the answer actually cites,
          divided by a small target (``_COVERAGE_TARGET``), capped at 1.
        * **grounding penalty** — if the answer is the no-evidence message (i.e.
          not grounded), the score collapses to 0.

        The relevance/coverage blend is ``0.6 * relevance + 0.4 * coverage``.
        """
        if not chunks:
            return 0.0

        top_k = min(len(chunks), self._settings.final_top_k)
        head = list(chunks[:top_k])

        rerank_scores = [
            chunk.rerank_score for chunk in head if chunk.rerank_score is not None
        ]
        if rerank_scores:
            relevance = _sigmoid(_mean(rerank_scores))
        else:
            relevance = _minmax_mean([chunk.score for chunk in head])

        citations = build_citations(chunks)
        valid_indices = {citation.index for citation in citations}
        distinct_cited = len(extract_cited_indices(answer_text) & valid_indices)
        target = max(1, min(_COVERAGE_TARGET, len(citations)))
        coverage = min(1.0, distinct_cited / target)

        score = _RELEVANCE_WEIGHT * relevance + _COVERAGE_WEIGHT * coverage

        if self._is_no_evidence(answer_text):
            score = 0.0

        return round(max(0.0, min(1.0, score)), 4)
