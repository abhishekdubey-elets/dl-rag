"""Deterministic evaluation metrics.

These are pure functions over answer text / retrieved chunks — no LLM required.
LLM-as-judge metrics (faithfulness, groundedness) live in the Evaluator, which
degrades gracefully when no judge is available.

Convention: every metric returns ``None`` when it cannot be computed for a case
(e.g. no expected URLs supplied), and aggregation ignores ``None``s — a metric
that can't be measured is *unknown*, not zero.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from dl_rag.models.domain import RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _norm_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


def top_k_recall(
    retrieved: Sequence[RetrievedChunk], expected_urls: Sequence[str]
) -> float | None:
    """Fraction of expected source URLs present among retrieved chunks."""
    if not expected_urls:
        return None
    got = {_norm_url(rc.chunk.metadata.url) for rc in retrieved}
    hits = sum(1 for u in expected_urls if _norm_url(u) in got)
    return hits / len(expected_urls)


def citation_precision(answer_text: str, n_sources: int) -> float | None:
    """Fraction of inline ``[n]`` citations that point at a real source index."""
    cited = [int(m) for m in _CITATION_RE.findall(answer_text or "")]
    if not cited:
        return None
    valid = sum(1 for i in cited if 1 <= i <= n_sources)
    return valid / len(cited)


def citation_density(answer_text: str) -> float | None:
    """Citations per non-heading paragraph — a proxy for 'every paragraph cites'."""
    paragraphs = [
        p for p in (answer_text or "").split("\n\n")
        if p.strip() and not p.strip().startswith("#")
    ]
    if not paragraphs:
        return None
    cited_paras = sum(1 for p in paragraphs if _CITATION_RE.search(p))
    return cited_paras / len(paragraphs)


def keyword_coverage(answer_text: str, expected_keywords: Sequence[str]) -> float | None:
    """Fraction of expected facts/keywords that appear in the answer (case-insensitive)."""
    if not expected_keywords:
        return None
    text = (answer_text or "").lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / len(expected_keywords)


def context_keyword_recall(
    retrieved: Sequence[RetrievedChunk], expected_keywords: Sequence[str]
) -> float | None:
    """Fraction of expected keywords present anywhere in the retrieved context.

    A deterministic proxy for context recall: did retrieval even surface the
    material needed to state the expected facts?
    """
    if not expected_keywords:
        return None
    blob = " ".join(rc.chunk.text.lower() for rc in retrieved)
    if not blob:
        return 0.0
    hits = sum(1 for kw in expected_keywords if kw.lower() in blob)
    return hits / len(expected_keywords)


def mean(values: Sequence[float | None]) -> float | None:
    """Mean of the non-None values (None when nothing was measurable)."""
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)
