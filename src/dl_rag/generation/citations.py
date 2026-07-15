"""Citation building and answer-to-source reconciliation.

Turns retrieved chunks into numbered :class:`Citation` objects (aligned with the
numbering produced by :func:`dl_rag.generation.prompts.format_context`), and maps
the ``[n]`` markers the model emitted back onto the sources actually used.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from dl_rag.models.api import SourceRef
from dl_rag.models.domain import Citation, RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")
_FALLBACK_LIMIT = 5


def build_citations(chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    """Build one citation per distinct source, indexed by 1-based position.

    Indices match :func:`format_context` numbering. Chunks that share a URL are
    deduplicated to a single citation: the first occurrence keeps its position
    index and later duplicates map onto it, so the numbering stays stable
    (positions are never renumbered — gaps left by duplicates are expected).
    """
    citations: list[Citation] = []
    seen: dict[str, int] = {}
    for position, retrieved in enumerate(chunks, start=1):
        meta = retrieved.chunk.metadata
        if meta.url in seen:
            continue
        seen[meta.url] = position
        citations.append(
            Citation(
                index=position,
                title=meta.title,
                url=meta.url,
                published_date=meta.published_date,
                content_type=meta.content_type,
                issue=meta.issue,
                author=meta.author,
            )
        )
    return citations


def extract_cited_indices(answer_text: str) -> set[int]:
    """Return the set of citation indices referenced via ``[n]`` in the answer."""
    if not answer_text:
        return set()
    return {int(match) for match in _CITATION_RE.findall(answer_text)}


def used_citations(citations: list[Citation], answer_text: str) -> list[Citation]:
    """Return the citations actually referenced by the answer.

    If the answer cited nothing but citations exist, fall back to the top few
    (<= 5) as best-effort source attribution.
    """
    cited = extract_cited_indices(answer_text)
    used = [citation for citation in citations if citation.index in cited]
    if used:
        return used
    if citations:
        return citations[:_FALLBACK_LIMIT]
    return []


def to_source_refs(citations: Sequence[Citation]) -> list[SourceRef]:
    """Map internal :class:`Citation` objects to wire-level :class:`SourceRef`."""
    return [
        SourceRef(
            index=citation.index,
            title=citation.title,
            url=citation.url,
            date=citation.published_date,
            content_type=citation.content_type,
            issue=citation.issue,
            author=citation.author,
        )
        for citation in citations
    ]
