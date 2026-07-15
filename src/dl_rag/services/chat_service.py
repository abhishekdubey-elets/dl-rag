"""ChatService — the end-to-end RAG use-case.

analyze → retrieve (hybrid) → generate (cited analyst answer) → remember → log.
Exposes a buffered ``chat`` and a token-level ``stream`` for SSE.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from dl_rag.config import Settings
from dl_rag.db.database import Database
from dl_rag.generation.answer_generator import AnswerGenerator
from dl_rag.generation.citations import to_source_refs
from dl_rag.logging_config import get_logger
from dl_rag.memory.conversation_memory import ConversationMemory
from dl_rag.models.api import ChatRequest, ChatResponse
from dl_rag.models.domain import GeneratedAnswer, QueryAnalysis, RetrievedChunk
from dl_rag.observability import metrics
from dl_rag.protocols import QueryAnalyzer
from dl_rag.repositories.feedback_repository import QueryLogRepository
from dl_rag.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger(__name__)


class ChatService:
    def __init__(
        self,
        *,
        analyzer: QueryAnalyzer,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        memory: ConversationMemory,
        db: Database,
        settings: Settings,
    ) -> None:
        self._analyzer = analyzer
        self._retriever = retriever
        self._generator = generator
        self._memory = memory
        self._db = db
        self._settings = settings

    # ------------------------------------------------------------------ #
    async def _prepare(
        self, request: ChatRequest
    ) -> tuple[QueryAnalysis, list[RetrievedChunk], str | None, list[dict[str, str]]]:
        analysis = await self._analyzer.analyze(request.query)
        metrics.QUERY_TYPE_COUNT.labels(analysis.query_type.value).inc()

        with metrics.observe(metrics.RETRIEVAL_LATENCY):
            chunks = await self._retriever.retrieve(analysis, request.filters)

        top_k = request.top_k or self._settings.final_top_k
        chunks = chunks[:top_k]

        conversation_id = request.conversation_id or uuid.uuid4().hex
        summary, turns = await self._load_history(conversation_id)
        return analysis, chunks, summary, turns

    async def _load_history(
        self, conversation_id: str
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Conversation history is best-effort — a memory outage must not fail chat."""
        try:
            return await self._memory.history_for_prompt(conversation_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat.memory_load_failed", error=str(exc))
            return None, []

    async def _remember(
        self, conversation_id: str, user_message: str, assistant_message: str
    ) -> None:
        try:
            await self._memory.append(conversation_id, user_message, assistant_message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat.memory_append_failed", error=str(exc))

    # ------------------------------------------------------------------ #
    async def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        conversation_id = request.conversation_id or uuid.uuid4().hex
        message_id = uuid.uuid4().hex

        with metrics.observe(metrics.CHAT_LATENCY):
            analysis = await self._analyzer.analyze(request.query)
            metrics.QUERY_TYPE_COUNT.labels(analysis.query_type.value).inc()

            with metrics.observe(metrics.RETRIEVAL_LATENCY):
                chunks = await self._retriever.retrieve(analysis, request.filters)
            chunks = chunks[: (request.top_k or self._settings.final_top_k)]

            summary, turns = await self._load_history(conversation_id)
            answer = await self._generator.generate(analysis, chunks, summary, turns)

        latency_ms = int((time.perf_counter() - start) * 1000)
        self._record_metrics(answer)
        await self._remember(conversation_id, request.query, answer.answer)
        await self._log(conversation_id, message_id, request.query, analysis, answer, latency_ms)

        return self._to_response(answer, conversation_id, message_id, latency_ms)

    # ------------------------------------------------------------------ #
    async def stream(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-friendly events: 'meta' → many 'token' → 'done' (or 'error')."""
        start = time.perf_counter()
        conversation_id = request.conversation_id or uuid.uuid4().hex
        message_id = uuid.uuid4().hex
        try:
            analysis, chunks, summary, turns = await self._prepare(request)
        except Exception as exc:  # noqa: BLE001 - report as an SSE error event
            logger.error("chat.stream.prepare_failed", error=str(exc))
            yield {"event": "error", "data": {"message": "retrieval_failed"}}
            return

        yield {
            "event": "meta",
            "data": {
                "conversation_id": conversation_id,
                "message_id": message_id,
                "query_type": analysis.query_type.value,
                "retrieved_documents": len(chunks),
            },
        }

        parts: list[str] = []
        try:
            async for delta in self._generator.stream_tokens(analysis, chunks, summary, turns):
                parts.append(delta)
                yield {"event": "token", "data": delta}
        except Exception as exc:  # noqa: BLE001
            logger.error("chat.stream.generation_failed", error=str(exc))
            yield {"event": "error", "data": {"message": "generation_failed"}}
            return

        full_text = "".join(parts)
        answer = self._generator.finalize(analysis, chunks, full_text)
        latency_ms = int((time.perf_counter() - start) * 1000)

        self._record_metrics(answer)
        await self._remember(conversation_id, request.query, full_text)
        await self._log(conversation_id, message_id, request.query, analysis, answer, latency_ms)

        response = self._to_response(answer, conversation_id, message_id, latency_ms)
        yield {"event": "done", "data": response.model_dump(mode="json")}

    # ------------------------------------------------------------------ #
    def _to_response(
        self, answer: GeneratedAnswer, conversation_id: str, message_id: str, latency_ms: int
    ) -> ChatResponse:
        sources = to_source_refs(answer.citations)
        return ChatResponse(
            answer=answer.answer,
            sources=sources,
            confidence=round(answer.confidence, 3),
            confidence_band=answer.confidence_band,
            query_type=answer.query_type,
            retrieved_documents=answer.retrieved_documents,
            conversation_id=conversation_id,
            message_id=message_id,
            latency_ms=latency_ms,
            token_usage={
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "total_tokens": answer.prompt_tokens + answer.completion_tokens,
            },
        )

    def _record_metrics(self, answer: GeneratedAnswer) -> None:
        if answer.prompt_tokens:
            metrics.LLM_TOKENS.labels("prompt").inc(answer.prompt_tokens)
        if answer.completion_tokens:
            metrics.LLM_TOKENS.labels("completion").inc(answer.completion_tokens)
        if not answer.grounded or answer.confidence < 0.4:
            metrics.LOW_CONFIDENCE_COUNT.inc()

    async def _log(
        self,
        conversation_id: str,
        message_id: str,
        query: str,
        analysis: QueryAnalysis,
        answer: GeneratedAnswer,
        latency_ms: int,
    ) -> None:
        try:
            cited_urls = [c.url for c in answer.citations]
            async with self._db.session() as session:
                await QueryLogRepository(session).add(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    query=query,
                    normalized_query=analysis.normalized_query,
                    query_type=analysis.query_type.value,
                    confidence=answer.confidence,
                    retrieved_documents=answer.retrieved_documents,
                    latency_ms=latency_ms,
                    cited_urls=cited_urls,
                )
        except Exception as exc:  # noqa: BLE001 - logging must never fail the request
            logger.warning("chat.log_failed", error=str(exc))
