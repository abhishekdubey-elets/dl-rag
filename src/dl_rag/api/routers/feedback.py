"""POST /api/feedback — record thumbs up/down + optional comment on an answer."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from dl_rag.api.deps import get_db
from dl_rag.api.security import require_api_key
from dl_rag.db.database import Database
from dl_rag.logging_config import get_logger
from dl_rag.models.api import FeedbackRequest, FeedbackResponse
from dl_rag.repositories.feedback_repository import FeedbackRepository

logger = get_logger(__name__)
router = APIRouter(tags=["feedback"], dependencies=[Depends(require_api_key)])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackRequest,
    db: Database = Depends(get_db),
) -> FeedbackResponse:
    async with db.session() as session:
        await FeedbackRepository(session).add(
            conversation_id=payload.conversation_id,
            message_id=payload.message_id,
            rating=payload.rating.value,
            comment=payload.comment,
            reason=payload.reason,
        )
    logger.info(
        "feedback.recorded",
        conversation_id=payload.conversation_id,
        message_id=payload.message_id,
        rating=payload.rating.value,
    )
    return FeedbackResponse()
