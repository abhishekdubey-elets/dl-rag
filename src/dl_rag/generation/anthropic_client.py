"""Anthropic Messages API client satisfying the ``LLMClient`` Protocol.

Used as the failover provider behind :class:`~dl_rag.generation.llm_client.FallbackLLM`
(or as the primary when no OpenAI key is configured). Accepts the same
OpenAI-style ``messages`` list the prompt layer produces and adapts it to the
Messages API shape: ``system`` roles become the top-level ``system`` prompt,
consecutive same-role turns are merged (the API requires strict alternation),
and ``response_format={"type": "json_object"}`` becomes a system instruction
(there is no JSON mode switch). The ``anthropic`` SDK is imported lazily so
the module compiles without it.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

import tenacity

from dl_rag.generation.llm_client import LLMError
from dl_rag.logging_config import get_logger

logger = get_logger(__name__)

_JSON_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else — no prose, "
    "no markdown fences."
)
# Claude 5 models reject `temperature` ("deprecated for this model"); older
# families (Haiku 4.5, 3.x) still accept it.
_NO_TEMPERATURE_RE = re.compile(r"claude-(?:[a-z]+-)?5(?:[.-]|$)")


def supports_temperature(model: str) -> bool:
    return _NO_TEMPERATURE_RE.search(model or "") is None


class AnthropicLLM:
    """Async Anthropic client with retry + streaming (``LLMClient`` Protocol)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client_obj: Any | None = None

    # ------------------------------------------------------------------ #
    # Lazy SDK wiring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sdk() -> Any:
        import anthropic  # noqa: PLC0415 - deliberately lazy

        return anthropic

    def _client(self) -> Any:
        if self._client_obj is None:
            sdk = self._sdk()
            # tenacity owns retries so behaviour matches the OpenAI client.
            self._client_obj = sdk.AsyncAnthropic(
                api_key=self._api_key, timeout=self._timeout, max_retries=0
            )
        return self._client_obj

    def _retrying(self, sdk: Any) -> tenacity.AsyncRetrying:
        return tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception_type(
                (sdk.APIConnectionError, sdk.RateLimitError, sdk.InternalServerError)
            ),
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=self._log_retry,
            reraise=True,
        )

    @staticmethod
    def _log_retry(retry_state: tenacity.RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning("llm.anthropic.retry", attempt=retry_state.attempt_number,
                       error=str(exc) if exc else None)

    # ------------------------------------------------------------------ #
    # Message adaptation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_messages(
        messages: list[dict[str, str]], *, json_mode: bool = False
    ) -> tuple[str | None, list[dict[str, str]]]:
        """OpenAI-style list → (system prompt, alternating user/assistant turns)."""
        system_parts: list[str] = []
        turns: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "user")
            content = (message.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
                continue
            role = "assistant" if role == "assistant" else "user"
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n\n" + content
            else:
                turns.append({"role": role, "content": content})
        if turns and turns[0]["role"] == "assistant":
            turns.insert(0, {"role": "user", "content": "(conversation resumes)"})
        if not turns:
            turns.append({"role": "user", "content": "Please respond."})
        if json_mode:
            system_parts.append(_JSON_INSTRUCTION)
        system = "\n\n".join(system_parts) if system_parts else None
        return system, turns

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        *,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        system, turns = self._split_messages(messages, json_mode=json_mode)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": turns,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
        }
        if supports_temperature(self._model):
            payload["temperature"] = self._temperature if temperature is None else temperature
        if system:
            payload["system"] = system
        return payload

    @staticmethod
    def _is_temperature_rejection(exc: BaseException) -> bool:
        text = str(exc)
        return "temperature" in text and ("deprecated" in text or "not supported" in text)

    async def _create(self, client: Any, sdk: Any, payload: dict[str, Any]) -> Any:
        """``messages.create`` with retries; drops ``temperature`` once if rejected."""
        try:
            response: Any = None
            async for attempt in self._retrying(sdk):
                with attempt:
                    response = await client.messages.create(**payload)
            return response
        except sdk.BadRequestError as exc:
            if "temperature" not in payload or not self._is_temperature_rejection(exc):
                raise
            logger.info("llm.anthropic.temperature_dropped", model=self._model)
            payload.pop("temperature", None)
            return await client.messages.create(**payload)

    @staticmethod
    def _parse_response(response: Any) -> tuple[str, dict[str, int]]:
        blocks = getattr(response, "content", None) or []
        text = "".join(
            getattr(block, "text", "") or ""
            for block in blocks
            if getattr(block, "type", "text") == "text"
        )
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        completion = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        return text, {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

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
        sdk = self._sdk()
        client = self._client()
        payload = self._build_payload(
            messages, temperature, max_tokens,
            json_mode=bool(response_format and response_format.get("type") == "json_object"),
        )
        try:
            response = await self._create(client, sdk, payload)
        except Exception as exc:  # noqa: BLE001 - normalise every failure to LLMError
            logger.error("llm.anthropic.complete.failed", model=self._model, error=str(exc))
            raise LLMError(f"Anthropic completion failed: {exc}") from exc

        text, usage = self._parse_response(response)
        logger.debug("llm.anthropic.complete.ok", model=self._model, **usage)
        return text, usage

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        sdk = self._sdk()
        client = self._client()
        payload = self._build_payload(messages, temperature, max_tokens)
        async def _open() -> tuple[Any, Any]:
            async for attempt in self._retrying(sdk):
                with attempt:
                    manager = client.messages.stream(**payload)
                    return manager, await manager.__aenter__()
            raise LLMError("Anthropic streaming failed to start")  # pragma: no cover

        try:
            try:
                manager, stream_obj = await _open()
            except sdk.BadRequestError as exc:
                if "temperature" not in payload or not self._is_temperature_rejection(exc):
                    raise
                logger.info("llm.anthropic.temperature_dropped", model=self._model)
                payload.pop("temperature", None)
                manager, stream_obj = await _open()
        except Exception as exc:  # noqa: BLE001
            logger.error("llm.anthropic.stream.connect_failed", model=self._model,
                         error=str(exc))
            raise LLMError(f"Anthropic streaming failed to start: {exc}") from exc

        try:
            async for text in stream_obj.text_stream:
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001 - never retry mid-stream
            logger.error("llm.anthropic.stream.failed", model=self._model, error=str(exc))
            raise LLMError(f"Anthropic streaming failed: {exc}") from exc
        finally:
            await manager.__aexit__(None, None, None)
