"""Reciprocal Rank Fusion (RRF) for merging multiple ranked candidate lists.

RRF is score-agnostic: it combines lists using only the *rank* of each item,
which makes it robust when the constituent retrievers (dense ANN, sparse BM25,
KG-expanded passes) produce scores on wildly different scales.
"""

from __future__ import annotations

from dl_rag.models.domain import RetrievedChunk
from dl_rag.models.enums import RetrievalSource


def _merge_sources(
    existing: list[RetrievalSource], incoming: list[RetrievalSource]
) -> list[RetrievalSource]:
    """Order-preserving union of two retrieval-source lists."""
    merged = list(existing)
    for src in incoming:
        if src not in merged:
            merged.append(src)
    return merged


def _min_optional(a: int | None, b: int | None) -> int | None:
    """Minimum of two optional ints, ignoring ``None`` operands."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ``ranked_lists`` into a single ranking via weighted RRF.

    The fused score for a chunk is ``Σ_i weight_i * 1 / (k + rank_i)`` where
    ``rank_i`` is the chunk's 1-based position in list ``i`` (lists in which the
    chunk does not appear contribute nothing). Chunks are identified across
    lists by ``chunk.chunk.id``; their ``sources`` are unioned and the smallest
    observed ``dense_rank`` / ``sparse_rank`` is retained. The fused value is
    written to ``.score`` and the result is returned sorted descending.

    Empty input (or input containing only empty lists) yields ``[]``.
    """
    if not ranked_lists:
        return []

    # Accumulator keyed by chunk id -> (representative chunk, fused score).
    fused_score: dict[str, float] = {}
    representative: dict[str, RetrievedChunk] = {}
    merged_sources: dict[str, list[RetrievalSource]] = {}
    best_dense: dict[str, int | None] = {}
    best_sparse: dict[str, int | None] = {}
    order: list[str] = []  # first-seen order, for stable tie-breaking

    for list_idx, ranked in enumerate(ranked_lists):
        weight = 1.0
        if weights is not None and list_idx < len(weights):
            weight = weights[list_idx]
        for position, item in enumerate(ranked):
            rank = position + 1
            chunk_id = item.chunk.id
            contribution = weight * (1.0 / (k + rank))
            if chunk_id not in fused_score:
                fused_score[chunk_id] = 0.0
                representative[chunk_id] = item
                merged_sources[chunk_id] = list(item.sources)
                best_dense[chunk_id] = item.dense_rank
                best_sparse[chunk_id] = item.sparse_rank
                order.append(chunk_id)
            else:
                merged_sources[chunk_id] = _merge_sources(
                    merged_sources[chunk_id], item.sources
                )
                best_dense[chunk_id] = _min_optional(
                    best_dense[chunk_id], item.dense_rank
                )
                best_sparse[chunk_id] = _min_optional(
                    best_sparse[chunk_id], item.sparse_rank
                )
            fused_score[chunk_id] += contribution

    fused: list[RetrievedChunk] = []
    for chunk_id in order:
        base = representative[chunk_id]
        fused.append(
            RetrievedChunk(
                chunk=base.chunk,
                score=fused_score[chunk_id],
                rerank_score=None,
                sources=merged_sources[chunk_id],
                dense_rank=best_dense[chunk_id],
                sparse_rank=best_sparse[chunk_id],
            )
        )

    fused.sort(key=lambda rc: rc.score, reverse=True)
    return fused
