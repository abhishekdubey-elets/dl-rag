"""OpenAI-compatible chat-completion client with retry + streaming.

The concrete :class:`LLMClient` implementation used by the generation layer. It
talks to any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, vLLM,
Together, Groq, ...). The ``openai`` SDK is imported *lazily* so this module can
be compiled and imported even when the SDK is not installed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import tenacity

from dl_rag.logging_config import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM request fails (after exhausting retries)."""


class OpenAICompatibleLLM:
    """Async chat-completion client satisfying the ``LLMClient`` Protocol."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client_obj: Any | None = None

    # ------------------------------------------------------------------ #
    # Lazy openai wiring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _openai() -> Any:
        """Import and return the ``openai`` module lazily."""
        import openai  # noqa: PLC0415 - deliberately lazy so the module compiles without it

        return openai

    def _client(self) -> Any:
        """Return a cached ``AsyncOpenAI`` client, creating it on first use."""
        if self._client_obj is None:
            openai_mod = self._openai()
            self._client_obj = openai_mod.AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client_obj

    def _retrying(self, openai_mod: Any) -> tenacity.AsyncRetrying:
        """Build the retry controller for transient, retryable API errors."""
        return tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception_type(
                (
                    openai_mod.APIConnectionError,
                    openai_mod.RateLimitError,
                    openai_mod.InternalServerError,
                )
            ),
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=self._log_retry,
            reraise=True,
        )

    @staticmethod
    def _log_retry(retry_state: tenacity.RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "llm.retry",
            attempt=retry_state.attempt_number,
            error=str(exc) if exc else None,
        )

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    @staticmethod
    def _usage_dict(response: Any) -> dict[str, int]:
        """Extract prompt/completion/total token counts, tolerating omissions."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    @classmethod
    def _parse_completion(cls, response: Any) -> tuple[str, dict[str, int]]:
        choices = getattr(response, "choices", None) or []
        content = ""
        if choices:
            message = getattr(choices[0], "message", None)
            content = (getattr(message, "content", None) or "") if message else ""
        return content, cls._usage_dict(response)

    # ------------------------------------------------------------------ #
    # Public API (LLMClient Protocol)
    # ------------------------------------------------------------------ #
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Run a single chat completion; return ``(text, usage)``."""
        openai_mod = self._openai()
        client = self._client()
        payload = self._build_payload(messages, temperature, max_tokens, stream=False)
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            # Explicit `await` inside the attempt: tenacity only auto-awaits when
            # the wrapped callable is an `async def`, and some openai SDK versions
            # implement `create` as a sync def returning an awaitable.
            response: Any = None
            async for attempt in self._retrying(openai_mod):
                with attempt:
                    response = await client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001 - normalise every failure to LLMError
            logger.error("llm.complete.failed", model=self._model, error=str(exc))
            raise LLMError(f"LLM completion failed: {exc}") from exc

        content, usage = self._parse_completion(response)
        logger.debug("llm.complete.ok", model=self._model, **usage)
        return content, usage

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding each content delta.

        The initial connection is retried; once the stream is open, mid-stream
        failures are not retried (they surface as :class:`LLMError`).
        """
        openai_mod = self._openai()
        client = self._client()
        payload = self._build_payload(messages, temperature, max_tokens, stream=True)

        try:
            stream_obj: Any = None
            async for attempt in self._retrying(openai_mod):
                with attempt:
                    stream_obj = await client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("llm.stream.connect_failed", model=self._model, error=str(exc))
            raise LLMError(f"LLM streaming failed to start: {exc}") from exc

        try:
            async for chunk in stream_obj:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001 - do not retry mid-stream, but report clearly
            logger.error("llm.stream.failed", model=self._model, error=str(exc))
            raise LLMError(f"LLM streaming failed: {exc}") from exc
