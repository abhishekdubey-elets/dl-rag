"""Generation layer: LLM client, prompt assembly, citations and answer building."""

from dl_rag.generation.answer_generator import AnswerGenerator
from dl_rag.generation.citations import (
    build_citations,
    extract_cited_indices,
    to_source_refs,
    used_citations,
)
from dl_rag.generation.anthropic_client import AnthropicLLM
from dl_rag.generation.llm_client import FallbackLLM, LLMError, OpenAICompatibleLLM
from dl_rag.generation.prompts import (
    QUERY_TYPE_INSTRUCTIONS,
    SYSTEM_PROMPT,
    build_messages,
    format_context,
)

__all__ = [
    "QUERY_TYPE_INSTRUCTIONS",
    "SYSTEM_PROMPT",
    "AnswerGenerator",
    "LLMError",
    "OpenAICompatibleLLM",
    "AnthropicLLM",
    "FallbackLLM",
    "build_citations",
    "build_messages",
    "extract_cited_indices",
    "format_context",
    "to_source_refs",
    "used_citations",
]
