"""LLM failover + Anthropic adapter tests — hermetic (no SDKs, no network)."""

from __future__ import annotations

import pytest

from dl_rag.exceptions import GenerationError
from dl_rag.generation.anthropic_client import AnthropicLLM, supports_temperature
from dl_rag.generation.llm_client import FallbackLLM, LLMError, is_quota_error
from tests.conftest import FailingLLM, FakeLLM


class FailingThenTokens:
    """Streams a couple of tokens, then dies mid-stream."""

    async def complete(self, messages, **kwargs):
        raise LLMError("boom")

    async def stream(self, messages, **kwargs):
        yield "partial "
        raise LLMError("mid-stream failure")


async def _collect(stream):
    return [chunk async for chunk in stream]


class TestLLMError:
    def test_is_generation_error_with_safe_detail(self):
        err = LLMError("LLM completion failed: Error code: 429 - insufficient_quota")
        assert isinstance(err, GenerationError)
        assert isinstance(err, RuntimeError)  # backwards compatible
        assert err.status_code == 503
        assert err.code == "generation_error"
        assert "insufficient_quota" not in (err.detail or "")  # provider text not leaked
        assert "insufficient_quota" in err.message

    def test_quota_detection(self):
        assert is_quota_error(RuntimeError("429 ... 'code': 'credit_balance_exhausted'"))
        assert is_quota_error(RuntimeError("type: insufficient_quota"))
        assert not is_quota_error(RuntimeError("429 Too Many Requests, slow down"))


class TestFallbackLLM:
    async def test_primary_success_no_fallback(self):
        primary, secondary = FakeLLM("from primary"), FakeLLM("from secondary")
        llm = FallbackLLM(primary, secondary)
        text, usage = await llm.complete([{"role": "user", "content": "q"}])
        assert text == "from primary"
        assert usage["total_tokens"] == 150
        assert secondary.calls == []

    async def test_primary_failure_falls_back(self):
        secondary = FakeLLM("from secondary")
        llm = FallbackLLM(FailingLLM(), secondary)
        text, _ = await llm.complete([{"role": "user", "content": "q"}])
        assert text == "from secondary"
        assert len(secondary.calls) == 1

    async def test_missing_primary_uses_secondary(self):
        secondary = FakeLLM("only provider")
        llm = FallbackLLM(None, secondary)
        text, _ = await llm.complete([{"role": "user", "content": "q"}])
        assert text == "only provider"
        assert llm.providers == ["fallback"]

    async def test_both_failing_raises_llm_error(self):
        class Dead:
            async def complete(self, messages, **kwargs):
                raise LLMError("dead")

        llm = FallbackLLM(Dead(), Dead())
        with pytest.raises(LLMError):
            await llm.complete([{"role": "user", "content": "q"}])

    async def test_no_secondary_reraises(self):
        llm = FallbackLLM(FailingLLM(), None)
        with pytest.raises(LLMError):
            await llm.complete([{"role": "user", "content": "q"}])

    def test_requires_a_provider(self):
        with pytest.raises(ValueError):
            FallbackLLM(None, None)

    async def test_stream_falls_back_before_first_token(self):
        llm = FallbackLLM(FailingLLM(), FakeLLM("a b c"))
        chunks = await _collect(llm.stream([{"role": "user", "content": "q"}]))
        assert "".join(chunks).split() == ["a", "b", "c"]

    async def test_stream_does_not_replay_after_tokens(self):
        llm = FallbackLLM(FailingThenTokens(), FakeLLM("never used"))
        with pytest.raises(LLMError):
            await _collect(llm.stream([{"role": "user", "content": "q"}]))

    async def test_stream_missing_primary(self):
        llm = FallbackLLM(None, FakeLLM("x y"))
        chunks = await _collect(llm.stream([{"role": "user", "content": "q"}]))
        assert "".join(chunks).split() == ["x", "y"]


class _Block:
    def __init__(self, text, type_="text"):
        self.text, self.type = text, type_


class _Usage:
    input_tokens, output_tokens = 120, 30


class _Response:
    content = [_Block("Hello "), _Block("[thinking]", "tool_use"), _Block("world")]
    usage = _Usage()


class TestAnthropicAdapter:
    def test_split_messages_system_and_alternation(self):
        system, turns = AnthropicLLM._split_messages([
            {"role": "system", "content": "You are an analyst."},
            {"role": "system", "content": "Cite sources."},
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "  "},
            {"role": "user", "content": "third"},
        ])
        assert system == "You are an analyst.\n\nCite sources."
        assert turns == [
            {"role": "user", "content": "first\n\nsecond"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "third"},
        ]

    def test_split_messages_leading_assistant_and_json_mode(self):
        system, turns = AnthropicLLM._split_messages(
            [{"role": "assistant", "content": "summary so far"}], json_mode=True
        )
        assert turns[0]["role"] == "user"
        assert turns[1] == {"role": "assistant", "content": "summary so far"}
        assert system is not None and "JSON" in system

    def test_split_messages_empty(self):
        system, turns = AnthropicLLM._split_messages([])
        assert system is None
        assert turns[0]["role"] == "user"

    def test_parse_response_text_blocks_and_usage(self):
        text, usage = AnthropicLLM._parse_response(_Response())
        assert text == "Hello world"
        assert usage == {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}

    def test_build_payload_claude5_omits_temperature(self):
        llm = AnthropicLLM("k", "claude-sonnet-5", temperature=0.1, max_tokens=99)
        payload = llm._build_payload(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
            None, None,
        )
        assert payload == {
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "q"}],
            "max_tokens": 99,
            "system": "sys",
        }
        assert llm._build_payload([{"role": "user", "content": "q"}], 0.7, 5)["max_tokens"] == 5

    def test_build_payload_older_models_keep_temperature(self):
        llm = AnthropicLLM("k", "claude-haiku-4-5-20251001", temperature=0.1, max_tokens=9)
        payload = llm._build_payload([{"role": "user", "content": "q"}], 0.7, None)
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 9

    def test_supports_temperature_heuristic(self):
        assert not supports_temperature("claude-sonnet-5")
        assert not supports_temperature("claude-fable-5-1")
        assert not supports_temperature("claude-opus-5")
        assert supports_temperature("claude-haiku-4-5-20251001")
        assert supports_temperature("claude-3-5-sonnet-latest")


class TestContainerWiring:
    def test_provider_selection(self, settings):
        from dl_rag.api.deps import _build_llm

        # placeholder OpenAI key, no Anthropic key → OpenAI alone (errors surface)
        assert _build_llm(settings).providers == ["openai:gpt-4o-mini"]

        both = settings.model_copy(update={"llm_api_key": "sk-real", "anthropic_api_key": "a"})
        assert _build_llm(both).providers == ["openai:gpt-4o-mini", "anthropic:claude-sonnet-5"]

        only_anthropic = settings.model_copy(update={"anthropic_api_key": "a"})
        assert _build_llm(only_anthropic).providers == ["anthropic:claude-sonnet-5"]

        disabled = both.model_copy(update={"llm_fallback_enabled": False})
        assert _build_llm(disabled).providers == ["openai:gpt-4o-mini"]
