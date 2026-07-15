"""Lightweight text-normalisation helpers (no heavy NLP dependencies)."""

from __future__ import annotations

import re
import unicodedata

_INLINE_WS = re.compile(r"[ \t\x0b\x0c\r]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


def clean_whitespace(text: str) -> str:
    """Collapse runs of inline whitespace and excess blank lines."""
    if not text:
        return ""
    text = text.replace(" ", " ").replace("​", "")
    text = _INLINE_WS.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "n-a"


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def first_sentences(text: str, n: int) -> str:
    return " ".join(split_sentences(text)[:n])


def extract_years(text: str) -> list[int]:
    """All plausible calendar years (1980–2049) found in ``text``, in order."""
    return [int(m) for m in _YEAR.findall(text or "")]


def truncate_chars(text: str, limit: int, suffix: str = "…") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix
