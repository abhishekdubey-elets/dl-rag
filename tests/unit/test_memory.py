"""Conversation-memory unit tests (fake cache + fake LLM)."""

from __future__ import annotations

from dl_rag.config import Settings
from dl_rag.memory.conversation_memory import ConversationMemory

from tests.conftest import FailingLLM, FakeCache, FakeLLM


def _memory(cache: FakeCache, llm, max_turns: int = 4) -> ConversationMemory:
    settings = Settings(memory_max_turns=max_turns, _env_file=None)
    return ConversationMemory(cache, llm, settings)


class TestConversationMemory:
    async def test_load_missing_conversation(self, fake_cache):
        memory = _memory(fake_cache, FakeLLM())
        summary, turns = await memory.load("nope")
        assert summary is None and turns == []

    async def test_append_and_load_roundtrip(self, fake_cache):
        memory = _memory(fake_cache, FakeLLM())
        await memory.append("c1", "What is SWAYAM?", "SWAYAM is a MOOC platform [1].")
        summary, turns = await memory.history_for_prompt("c1")
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"
        assert "SWAYAM" in turns[0]["content"]

    async def test_overflow_triggers_trim_and_summary(self, fake_cache):
        memory = _memory(fake_cache, FakeLLM("Condensed summary of earlier turns."), max_turns=4)
        for i in range(4):  # 8 turns total > 4
            await memory.append("c2", f"question {i}", f"answer {i}")
        summary, turns = await memory.load("c2")
        assert len(turns) <= 4
        assert summary  # summarisation ran
        # Most recent turns survive verbatim.
        assert turns[-1]["content"] == "answer 3"

    async def test_summarisation_failure_still_trims(self, fake_cache):
        memory = _memory(fake_cache, FailingLLM(), max_turns=2)
        for i in range(3):
            await memory.append("c3", f"q{i}", f"a{i}")
        _summary, turns = await memory.load("c3")
        assert len(turns) <= 2  # trimmed even though the LLM failed
