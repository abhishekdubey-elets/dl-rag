"""Conversation memory backed by the :class:`~dl_rag.protocols.Cache`.

Per-conversation state is stored as a single JSON blob under ``conv:{id}``::

    {"summary": str | None, "turns": [{"role": "user"|"assistant", "content": str}, ...]}

Recent turns are kept verbatim; once the turn buffer overflows
``memory_max_turns`` the oldest turns are folded into a rolling summary via the
LLM so long conversations stay bounded without losing early context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dl_rag.logging_config import get_logger

if TYPE_CHECKING:
    from dl_rag.config import Settings
    from dl_rag.protocols import Cache, LLMClient

logger = get_logger(__name__)

_VALID_ROLES = {"user", "assistant"}

_SUMMARY_SYSTEM_PROMPT = (
    "You maintain a compact running summary of an ongoing conversation between a "
    "user and an education-policy analyst assistant. Given the existing summary "
    "and the next batch of turns, return an updated summary that preserves the "
    "durable facts, the user's goals and any commitments or unresolved threads. "
    "Be concise (a short paragraph), neutral, and do not add information that is "
    "not present in the turns."
)


class ConversationMemory:
    """Load/append per-conversation history with rolling summarisation."""

    def __init__(self, cache: Cache, llm: LLMClient, settings: Settings) -> None:
        self._cache = cache
        self._llm = llm
        self._settings = settings

    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"conv:{conversation_id}"

    @staticmethod
    def _sanitize_turns(raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        turns: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in _VALID_ROLES and isinstance(content, str):
                turns.append({"role": role, "content": content})
        return turns

    # ------------------------------------------------------------------ #
    async def load(self, conversation_id: str) -> tuple[str | None, list[dict[str, str]]]:
        """Return ``(summary, recent_turns)``; never raises on a cache miss."""
        try:
            blob = await self._cache.get_json(self._key(conversation_id))
        except Exception as exc:  # noqa: BLE001 - cache failures must not break chat
            logger.warning(
                "memory.load.failed", conversation_id=conversation_id, error=str(exc)
            )
            return None, []

        if not isinstance(blob, dict):
            return None, []

        summary = blob.get("summary")
        summary_value = summary if isinstance(summary, str) and summary.strip() else None
        return summary_value, self._sanitize_turns(blob.get("turns"))

    async def append(
        self, conversation_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Append a user/assistant exchange, summarising overflow as needed."""
        summary, turns = await self.load(conversation_id)
        turns.append({"role": "user", "content": user_message})
        turns.append({"role": "assistant", "content": assistant_message})

        max_turns = max(0, self._settings.memory_max_turns)
        if len(turns) > max_turns:
            overflow = turns[: len(turns) - max_turns]
            turns = turns[len(turns) - max_turns :]
            try:
                summary = await self._summarize(summary, overflow)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully; keep chat alive
                logger.warning(
                    "memory.summarize.failed",
                    conversation_id=conversation_id,
                    error=str(exc),
                )

        blob: dict[str, Any] = {"summary": summary, "turns": turns}
        try:
            await self._cache.set_json(
                self._key(conversation_id),
                blob,
                ttl=self._settings.memory_ttl_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "memory.persist.failed", conversation_id=conversation_id, error=str(exc)
            )

    async def history_for_prompt(
        self, conversation_id: str
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Return ``(summary, turns)`` ready to feed ``build_messages``."""
        return await self.load(conversation_id)

    # ------------------------------------------------------------------ #
    async def _summarize(
        self, current_summary: str | None, overflow_turns: list[dict[str, str]]
    ) -> str:
        """Fold ``overflow_turns`` into ``current_summary`` via the LLM."""
        transcript = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in overflow_turns
        )
        base = current_summary.strip() if current_summary else "(no summary yet)"
        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Existing running summary:\n{base}\n\n"
                    f"New turns to fold in:\n{transcript}\n\n"
                    "Return only the updated running summary."
                ),
            },
        ]
        text, _usage = await self._llm.complete(messages, temperature=0.1, max_tokens=400)
        updated = text.strip()
        return updated or (current_summary or "")
