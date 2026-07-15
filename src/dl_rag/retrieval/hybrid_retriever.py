"""Hybrid retrieval orchestration.

Fuses dense (ANN) and sparse (lexical) retrieval — optionally broadened by a
one-hop knowledge-graph expansion — with Reciprocal Rank Fusion, reranks the
fused head with a cross-encoder, applies a score floor, and optionally compresses
the survivors to a token budget. Every backend call is individually guarded so a
single failing store degrades gracefully instead of failing the whole request.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable
from datetime import date
from typing import Any

from dl_rag.config import Settings
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import QueryAnalysis, RetrievedChunk
from dl_rag.models.enums import RetrievalSource
from dl_rag.protocols import KnowledgeGraph, Reranker, SparseRetriever
from dl_rag.retrieval.compression import ContextCompressor
from dl_rag.retrieval.dense import DenseRetriever
from dl_rag.retrieval.fusion import reciprocal_rank_fusion

logger = get_logger(__name__)

# Weight of freshness (vs. semantic relevance) when a query is recency-sensitive.
_RECENCY_WEIGHT = 0.35


def _force_include_latest(
    kept: list[RetrievedChunk],
    latest: list[RetrievedChunk],
    top_k: int,
    slots: int = 4,
) -> list[RetrievedChunk]:
    """Prepend up to ``slots`` newest on-topic chunks, keeping ``top_k`` total.

    ``latest`` arrives newest-first from the repository. One chunk per document
    (the newest article's lead chunk beats its own tail chunks), duplicates
    already in ``kept`` don't burn a slot.
    """
    kept_ids = {rc.chunk.id for rc in kept}
    seen_docs: set[str] = set()
    forced: list[RetrievedChunk] = []
    for rc in latest:
        if len(forced) >= slots:
            break
        if rc.chunk.id in kept_ids or rc.chunk.document_id in seen_docs:
            continue
        seen_docs.add(rc.chunk.document_id)
        forced.append(rc)
    if not forced:
        return kept[:top_k]
    return (forced + kept)[:top_k]


def _expand_with_entities(query: str, entities: list[str]) -> str:
    """Append canonical entity names that are not already present in the query.

    Bridges abbreviation gaps for lexical search: "next WES event" alone cannot
    match articles that only say "World Education Summit". Canonical names come
    from the analyzer's gazetteer resolution.
    """
    lowered = query.lower()
    extras = [name for name in entities if name.lower() not in lowered]
    if not extras:
        return query
    return f"{query} ({', '.join(extras)})"


def _blend_recency(
    candidates: list[RetrievedChunk], weight: float = _RECENCY_WEIGHT
) -> list[RetrievedChunk]:
    """Re-order reranked candidates by blending relevance with freshness.

    ``final = (1-w) * sigmoid(rerank_score) + w * exp(-age_years)`` — a chunk
    published today scores freshness 1.0, one year old ≈ 0.37, two years ≈ 0.14.
    Used only for queries asking about "next / latest / upcoming", where pure
    semantic similarity systematically prefers older but wordier announcements.
    """
    today = date.today()

    def blended(rc: RetrievedChunk) -> float:
        relevance = 1.0 / (1.0 + math.exp(-rc.final_score))
        published = rc.chunk.metadata.published_date
        if published is None:
            freshness = 0.1  # unknown date → treat as stale, don't reward
        else:
            age_years = max(0.0, (today - published).days / 365.0)
            freshness = math.exp(-age_years)
        return (1.0 - weight) * relevance + weight * freshness

    return sorted(candidates, key=blended, reverse=True)


class HybridRetriever:
    """Coordinate dense + sparse (+ KG) retrieval, fusion, rerank, compression."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        reranker: Reranker,
        settings: Settings,
        knowledge_graph: KnowledgeGraph | None = None,
        compressor: ContextCompressor | None = None,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._settings = settings
        self._kg = knowledge_graph
        self._compressor = compressor

    # ------------------------------------------------------------------ #
    # Filter construction
    # ------------------------------------------------------------------ #
    def build_filters(
        self, analysis: QueryAnalysis, chat_filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Assemble the shared filter dict from analysis + caller-supplied filters.

        Produces the shared schema
        ``{year_from, year_to, content_types, authors, tags, entities}`` with
        None/empty keys omitted. Merge policy:

        * **Years** — intersected: the tightest window implied by the analysis
          time-range and any caller bounds (``max`` of lower bounds, ``min`` of
          upper bounds).
        * **content_types** — the caller's explicit choice wins; when both sides
          are present we keep their intersection, falling back to the caller's
          list if that intersection is empty (never silently drop a user filter).
        * **authors / tags** — caller-only (the analyzer does not infer them).
        * **entities** — caller-only. Analysis-extracted entities are
          deliberately NOT applied as a hard filter: chunk entity tagging is
          high-precision but not exhaustive, so AND-filtering on it silently
          kills recall for lexically-relevant chunks that missed a tag. The
          analyzer's entities still boost recall via the KG-expansion pass.
        """
        caller = self._normalize_chat_filters(chat_filters)

        year_from = _max_opt(analysis.time_range.from_year, caller.get("year_from"))
        year_to = _min_opt(analysis.time_range.to_year, caller.get("year_to"))

        analysis_cts = [ct.value for ct in analysis.content_type_filter]
        content_types = _merge_content_types(analysis_cts, caller.get("content_types"))

        entities = list(caller.get("entities") or [])
        authors = caller.get("authors") or []
        tags = caller.get("tags") or []

        candidate: dict[str, Any] = {
            "year_from": year_from,
            "year_to": year_to,
            "content_types": content_types or None,
            "authors": authors or None,
            "tags": tags or None,
            "entities": entities or None,
        }
        return {key: value for key, value in candidate.items() if value is not None}

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    async def retrieve(
        self, analysis: QueryAnalysis, chat_filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        started = time.perf_counter()
        query = analysis.original_query
        # Canonical-entity expansion: when the analyzer resolved an abbreviation
        # (e.g. "WES" → "World Education Summit"), append canonical names that do
        # not literally appear in the query so lexical search can match articles
        # that only spell the full form. Reranking still uses the original query.
        search_query = _expand_with_entities(query, analysis.entities)
        candidate_k = self._settings.retrieval_candidates

        filters = self.build_filters(analysis, chat_filters)
        search_filters = filters or None

        # --- dense + sparse, concurrently, each independently guarded ------- #
        # Recency-sensitive queries additionally run a "fresh slice" pass — the
        # same query hard-filtered to the last ~18 months. Without it, recent
        # articles drown among hundreds of older lexically-similar chunks and
        # never even reach the reranker for the recency blend to promote.
        recent_filters: dict[str, Any] | None = None
        if analysis.recency_sensitive:
            recent_filters = dict(filters)
            recent_filters["year_from"] = max(
                recent_filters.get("year_from") or 0, date.today().year - 1
            )

        passes: list[Awaitable[list[RetrievedChunk]]] = [
            _guard("dense", self._dense.search(search_query, candidate_k, search_filters)),
            _guard("sparse", self._sparse.search(search_query, candidate_k, search_filters)),
        ]
        if recent_filters is not None:
            passes.append(
                _guard(
                    "dense_recent",
                    self._dense.search(search_query, candidate_k, recent_filters),
                )
            )
            passes.append(
                _guard(
                    "sparse_recent",
                    self._sparse.search(search_query, candidate_k, recent_filters),
                )
            )
        results = await asyncio.gather(*passes)
        dense_results, sparse_results = results[0], results[1]

        ranked_lists: list[list[RetrievedChunk]] = [dense_results, sparse_results]
        weights: list[float] = [
            self._settings.vector_weight,
            self._settings.sparse_weight,
        ]
        for fresh in results[2:]:
            if fresh:
                ranked_lists.append(fresh)
                weights.append(self._settings.vector_weight)

        # --- optional one-hop KG expansion --------------------------------- #
        # Design choice: we run an *extra dense pass* whose query text is
        # augmented with the expanded entity names, rather than injecting those
        # names into filters["entities"]. Filtering would AND-restrict results to
        # chunks tagged with the (often sparsely-tagged) related entities and can
        # over-prune; query augmentation instead *broadens* semantic recall toward
        # the neighbourhood while letting RRF weigh the extra evidence.
        kg_results = await self._maybe_kg_expand(analysis, query, candidate_k,
                                                  search_filters)
        if kg_results:
            ranked_lists.append(kg_results)
            weights.append(self._settings.vector_weight)

        # --- fusion --------------------------------------------------------- #
        fused = reciprocal_rank_fusion(
            ranked_lists, k=self._settings.rrf_k, weights=weights
        )

        # --- rerank fused head --------------------------------------------- #
        # For recency-sensitive queries ("next", "latest", "upcoming") keep ALL
        # reranked candidates so the recency blend below can promote fresh
        # articles that pure semantic relevance would cut from the top-k.
        rerank_input = fused[: min(len(fused), candidate_k)]
        if analysis.recency_sensitive and len(results) > 2:
            # Guarantee fresh-slice hits a rerank slot: RRF ordering favours
            # multi-list consensus, which systematically buries chunks that only
            # the fresh passes surfaced. Append their heads past the RRF cut.
            in_input = {rc.chunk.id for rc in rerank_input}
            for fresh_list in results[2:]:
                for rc in fresh_list[:12]:
                    if rc.chunk.id not in in_input:
                        rerank_input.append(rc)
                        in_input.add(rc.chunk.id)
        rerank_k = (
            len(rerank_input) if analysis.recency_sensitive
            else self._settings.final_top_k
        )
        reranked = await self._reranker.rerank(query, rerank_input, rerank_k)

        if analysis.recency_sensitive and reranked:
            reranked = _blend_recency(reranked)[: self._settings.final_top_k]

        # --- score floor (never empty if anything survived rerank) --------- #
        min_score = self._settings.min_rerank_score
        kept = [rc for rc in reranked if rc.final_score >= min_score]
        if not kept and reranked:
            kept = reranked[:1]

        # --- guaranteed entity/latest slice --------------------------------- #
        # Two query shapes systematically defeat similarity ranking:
        #   1. recency ("next / latest / upcoming X") — argmax-by-date;
        #   2. entity + single-year ("WES 2026", "speakers at WES 2026") —
        #      edition-scoped: lexically-loud lookalikes (other summits that
        #      year) outrank the entity's own articles.
        # For both, force-include the newest chunks that phrase-match the
        # resolved entity (within the active filters, e.g. the year window).
        single_year = (
            analysis.time_range.from_year is not None
            and analysis.time_range.from_year == analysis.time_range.to_year
        )
        if analysis.recency_sensitive or (analysis.entities and single_year):
            phrase = analysis.entities[0] if analysis.entities else None
            latest_hits = await self._latest_slice(
                search_query, search_filters, phrase
            )
            if latest_hits:
                kept = _force_include_latest(
                    kept, latest_hits, self._settings.final_top_k
                )

        # --- optional compression ------------------------------------------ #
        final = self._compressor.compress(query, kept) if self._compressor else kept

        logger.info(
            "retrieval.done",
            query_type=analysis.query_type.value,
            dense=len(dense_results),
            sparse=len(sparse_results),
            kg=len(kg_results),
            fused=len(fused),
            reranked=len(reranked),
            final=len(final),
            filtered=search_filters is not None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return final

    # ------------------------------------------------------------------ #
    async def _latest_slice(
        self,
        query: str,
        filters: dict[str, Any] | None,
        phrase: str | None = None,
    ) -> list[RetrievedChunk]:
        """Newest on-topic chunks via the sparse backend's ``latest`` (optional)."""
        latest_fn = getattr(self._sparse, "latest", None)
        if latest_fn is None:
            return []
        try:
            return await latest_fn(query, 6, filters, phrase=phrase)
        except Exception as exc:  # noqa: BLE001 - never fail retrieval on this
            logger.warning("retrieval.latest_slice.failed", error=str(exc))
            return []

    # ------------------------------------------------------------------ #
    async def _maybe_kg_expand(
        self,
        analysis: QueryAnalysis,
        query: str,
        candidate_k: int,
        search_filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Run a KG-broadened extra dense pass; ``[]`` when disabled/unavailable."""
        if not (
            self._settings.kg_expansion_enabled
            and self._kg is not None
            and analysis.entities
        ):
            return []
        try:
            expanded = await self._kg.expand(analysis.entities, hops=1)
            new_names = [n for n in expanded if n not in set(analysis.entities)]
            if not new_names:
                return []
            augmented_query = f"{query} {' '.join(new_names)}".strip()
            results = await self._dense.search(
                augmented_query, candidate_k, search_filters
            )
            for item in results:
                if RetrievalSource.KG not in item.sources:
                    item.sources.append(RetrievalSource.KG)
            logger.debug(
                "retrieval.kg.expanded",
                seed=len(analysis.entities),
                expanded=len(new_names),
                results=len(results),
            )
            return results
        except Exception as exc:  # noqa: BLE001 - KG is best-effort; never fatal.
            logger.warning("retrieval.kg.failed", error=str(exc))
            return []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_chat_filters(
        chat_filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Coerce a caller filter dict (or pydantic model) to normalized parts."""
        if chat_filters is None:
            return {}
        if hasattr(chat_filters, "model_dump"):
            raw: dict[str, Any] = chat_filters.model_dump()  # type: ignore[attr-defined]
        elif isinstance(chat_filters, dict):
            raw = chat_filters
        else:
            return {}
        return {
            "year_from": _as_opt_int(raw.get("year_from")),
            "year_to": _as_opt_int(raw.get("year_to")),
            "content_types": _as_str_list(raw.get("content_types")),
            "authors": _as_str_list(raw.get("authors")),
            "tags": _as_str_list(raw.get("tags")),
            "entities": _as_str_list(raw.get("entities")),
        }


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
async def _guard(
    backend: str, coro: Awaitable[list[RetrievedChunk]]
) -> list[RetrievedChunk]:
    """Await a retrieval coroutine, returning ``[]`` (and logging) on failure."""
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 - isolate one backend's failure.
        logger.warning("retrieval.backend.failed", backend=backend, error=str(exc))
        return []


def _max_opt(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _min_opt(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _merge_content_types(
    analysis_cts: list[str], caller_cts: list[str] | None
) -> list[str]:
    if not caller_cts:
        return list(analysis_cts)
    if not analysis_cts:
        return list(caller_cts)
    caller_set = set(caller_cts)
    intersection = [ct for ct in analysis_cts if ct in caller_set]
    return intersection or list(caller_cts)


def _as_opt_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    items = [value] if isinstance(value, str) else list(value)
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(getattr(item, "value", item)).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
