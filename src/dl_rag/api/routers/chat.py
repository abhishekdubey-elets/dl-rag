"""POST /api/chat — buffered JSON answer or SSE token stream."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from dl_rag.api.deps import get_chat_service
from dl_rag.api.security import require_api_key
from dl_rag.models.api import ChatRequest, ChatResponse
from dl_rag.services.chat_service import ChatService

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    if not payload.stream:
        return await service.chat(payload)

    async def event_stream() -> AsyncIterator[dict]:
        async for event in service.stream(payload):
            data = event["data"]
            yield {
                "event": event["event"],
                "data": data if isinstance(data, str) else json.dumps(data),
            }

    return EventSourceResponse(event_stream())
