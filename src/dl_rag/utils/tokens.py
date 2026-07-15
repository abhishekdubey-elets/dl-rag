"""Token counting/truncation with a graceful fallback when tiktoken is absent.

The whole pipeline (chunking, context compression, budgeting) shares this so
token accounting is consistent everywhere.
"""

from __future__ import annotations

from typing import Any

_encoder: Any | None = None
_encoder_ready = False


def _get_encoder() -> Any | None:
    global _encoder, _encoder_ready
    if _encoder_ready:
        return _encoder
    try:  # tiktoken is a declared dependency but keep import lazy + safe.
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - any failure → heuristic fallback
        _encoder = None
    _encoder_ready = True
    return _encoder


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text`` (heuristic if tiktoken missing)."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # ~0.75 words/token for English prose.
    return max(1, round(len(text.split()) / 0.75))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` so it fits within ``max_tokens``."""
    if max_tokens <= 0 or not text:
        return ""
    enc = _get_encoder()
    if enc is not None:
        toks = enc.encode(text, disallowed_special=())
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    approx_words = int(max_tokens * 0.75)
    words = text.split()
    if len(words) <= approx_words:
        return text
    return " ".join(words[:approx_words])
