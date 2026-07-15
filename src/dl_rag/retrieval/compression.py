"""Token-budgeted context compression.

Given chunks already ordered by relevance, greedily pack as many as fit inside a
token budget. A chunk that would overflow is offered a second chance as a
sentence-trimmed version that keeps only the sentences overlapping the query
terms; if even that will not fit, packing stops. The single most relevant chunk
is always emitted so the generator never receives an empty context.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from dl_rag.logging_config import get_logger
from dl_rag.models.domain import RetrievedChunk
from dl_rag.utils.text import split_sentences
from dl_rag.utils.tokens import count_tokens, truncate_to_tokens

logger = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9]+")

# Minimal stopword set — enough to avoid matching sentences on filler words.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is",
        "are", "was", "were", "be", "by", "with", "as", "at", "it", "its",
        "this", "that", "these", "those", "what", "which", "who", "how",
        "why", "when", "where", "from", "about", "into", "over", "than",
    }
)


class ContextCompressor:
    """Greedily fit relevance-ordered chunks within ``max_tokens``."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    def compress(
        self, query: str, chunks: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Return the prefix of ``chunks`` (some possibly trimmed) that fits."""
        items = list(chunks)
        if not items:
            return []

        query_terms = self._query_terms(query)
        kept: list[RetrievedChunk] = []
        running = 0

        for item in items:
            text = item.chunk.text
            tokens = count_tokens(text)

            if running + tokens <= self.max_tokens:
                kept.append(item)
                running += tokens
                continue

            # Overflow: try a query-focused, sentence-trimmed version.
            trimmed = self._trim_to_query(text, query_terms)
            if trimmed:
                trimmed_tokens = count_tokens(trimmed)
                if running + trimmed_tokens <= self.max_tokens:
                    kept.append(self._with_text(item, trimmed))
                    running += trimmed_tokens
                    continue

            # Cannot fit even trimmed. Guarantee at least the top chunk.
            if not kept:
                fallback = trimmed or text
                fitted = truncate_to_tokens(fallback, self.max_tokens) or fallback
                kept.append(self._with_text(item, fitted))
                running += count_tokens(fitted)
            break

        logger.debug(
            "retrieval.compress.done",
            budget=self.max_tokens,
            used_tokens=running,
            input=len(items),
            kept=len(kept),
        )
        return kept

    # ------------------------------------------------------------------ #
    @staticmethod
    def _query_terms(query: str) -> set[str]:
        return {
            tok
            for tok in _WORD.findall(query.lower())
            if len(tok) > 2 and tok not in _STOPWORDS
        }

    @classmethod
    def _trim_to_query(cls, text: str, query_terms: set[str]) -> str:
        """Keep only sentences overlapping ``query_terms`` (order preserved).

        Returns ``""`` when nothing overlaps, signalling the caller to fall back.
        """
        if not query_terms:
            return ""
        sentences = split_sentences(text)
        if not sentences:
            return ""
        selected = [s for s in sentences if cls._overlaps(s, query_terms)]
        if not selected:
            return ""
        return " ".join(selected)

    @staticmethod
    def _overlaps(sentence: str, query_terms: set[str]) -> bool:
        words = set(_WORD.findall(sentence.lower()))
        return not words.isdisjoint(query_terms)

    @staticmethod
    def _with_text(item: RetrievedChunk, new_text: str) -> RetrievedChunk:
        """Copy ``item`` with the chunk text (and token count) replaced."""
        new_chunk = item.chunk.model_copy(
            update={"text": new_text, "token_count": count_tokens(new_text)}
        )
        return item.model_copy(update={"chunk": new_chunk})
