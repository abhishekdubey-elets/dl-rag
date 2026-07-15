"""Fusion, compression, noop-rerank, and hybrid orchestration unit tests."""

from __future__ import annotations

from typing import Any

import pytest

from dl_rag.config import Settings
from dl_rag.models.domain import QueryAnalysis, TimeRange
from dl_rag.models.enums import ContentType, QueryType, RetrievalSource
from dl_rag.retrieval.compression import ContextCompressor
from dl_rag.retrieval.dense import DenseRetriever
from dl_rag.retrieval.fusion import reciprocal_rank_fusion
from dl_rag.retrieval.hybrid_retriever import HybridRetriever
from dl_rag.retrieval.reranker import NoopReranker

from tests.conftest import make_retrieved


class TestRRF:
    def test_overlapping_chunk_wins(self):
        a = [make_retrieved(1, source=RetrievalSource.DENSE),
             make_retrieved(2, source=RetrievalSource.DENSE)]
        b = [make_retrieved(2, source=RetrievalSource.SPARSE),
             make_retrieved(3, source=RetrievalSource.SPARSE)]
        fused = reciprocal_rank_fusion([a, b], k=60)
        assert fused[0].chunk.chunk_index == 2  # appears in both lists
        srcs = set(fused[0].sources)
        assert RetrievalSource.DENSE in srcs and RetrievalSource.SPARSE in srcs
        assert len(fused) == 3

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []


class TestCompression:
    def test_budget_respected(self):
        chunks = [make_retrieved(i, text="word " * 200) for i in range(5)]
        kept = ContextCompressor(max_tokens=450).compress("NEP", chunks)
        assert 1 <= len(kept) < 5

    def test_always_keeps_top_one(self):
        chunks = [make_retrieved(0, text="word " * 500)]
        kept = ContextCompressor(max_tokens=10).compress("NEP", chunks)
        assert len(kept) == 1


class TestNoopReranker:
    async def test_sorts_and_truncates(self):
        cands = [make_retrieved(i, score=float(i), rerank=None) for i in range(5)]
        out = await NoopReranker().rerank("q", cands, top_k=3)
        assert len(out) == 3
        assert out[0].chunk.chunk_index == 4  # highest score first
        assert all(rc.rerank_score is not None for rc in out)


# --------------------------------------------------------------------------- #
# Hybrid retriever with fake backends
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    dimension = 8

    async def embed_documents(self, texts):
        return [[0.1] * 8 for _ in texts]

    async def embed_query(self, text):
        return [0.1] * 8


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.calls: list[dict[str, Any]] = []

    async def ensure_collection(self, dimension):  # pragma: no cover
        pass

    async def upsert(self, chunks):  # pragma: no cover
        return len(chunks)

    async def search(self, vector, top_k, filters=None):
        self.calls.append({"top_k": top_k, "filters": filters})
        return list(self.results)

    async def delete_by_document(self, document_id):  # pragma: no cover
        pass

    async def count(self):  # pragma: no cover
        return 0


class FakeSparse:
    def __init__(self, results=None, fail=False):
        self.results = results or []
        self.fail = fail

    async def search(self, query, top_k, filters=None):
        if self.fail:
            raise RuntimeError("sparse backend down")
        return list(self.results)


class FakeKG:
    async def add_entities(self, entities):  # pragma: no cover
        pass

    async def add_relations(self, relations):  # pragma: no cover
        pass

    async def expand(self, entity_names, hops=1):
        return ["SWAYAM Prabha"]

    async def neighbors(self, entity_name):  # pragma: no cover
        return []


def _analysis(**overrides) -> QueryAnalysis:
    base = dict(
        original_query="How has NEP evolved since 2020?",
        normalized_query="how has nep evolved since 2020?",
        query_type=QueryType.TIMELINE,
        entities=["NEP 2020"],
        time_range=TimeRange(from_year=2020),
    )
    base.update(overrides)
    return QueryAnalysis(**base)


@pytest.fixture
def hybrid(settings: Settings):
    settings = settings.model_copy(update={"final_top_k": 3, "retrieval_candidates": 10})
    dense_results = [make_retrieved(i, source=RetrievalSource.DENSE) for i in range(4)]
    sparse_results = [make_retrieved(i + 2, source=RetrievalSource.SPARSE) for i in range(4)]
    store = FakeVectorStore(dense_results)
    retriever = HybridRetriever(
        DenseRetriever(FakeEmbedder(), store),
        FakeSparse(sparse_results),
        NoopReranker(),
        settings,
        knowledge_graph=FakeKG(),
        compressor=None,
    )
    return retriever, store


class TestHybridRetriever:
    async def test_end_to_end_fusion(self, hybrid):
        retriever, _ = hybrid
        out = await retriever.retrieve(_analysis())
        assert 1 <= len(out) <= 3
        # chunk 2 and 3 appear in both dense and sparse → should rank on top
        top_ids = {rc.chunk.chunk_index for rc in out[:2]}
        assert top_ids & {2, 3}

    async def test_survives_backend_failure(self, settings: Settings):
        dense_results = [make_retrieved(i) for i in range(3)]
        retriever = HybridRetriever(
            DenseRetriever(FakeEmbedder(), FakeVectorStore(dense_results)),
            FakeSparse(fail=True),
            NoopReranker(),
            settings,
        )
        out = await retriever.retrieve(_analysis())
        assert len(out) >= 1  # dense results still flow through

    def test_build_filters_merges(self, hybrid):
        retriever, _ = hybrid
        analysis = _analysis(
            time_range=TimeRange(from_year=2018, to_year=2022),
            content_type_filter=[ContentType.INTERVIEW],
        )
        filters = retriever.build_filters(analysis, {"year_from": 2019})
        assert filters["year_from"] == 2019  # intersect: max of lower bounds
        assert filters["year_to"] == 2022
        assert ContentType.INTERVIEW.value in filters["content_types"]

    def test_build_filters_empty(self, hybrid):
        retriever, _ = hybrid
        analysis = _analysis(entities=[], time_range=TimeRange())
        assert retriever.build_filters(analysis) == {}


class TestRecencyBlend:
    def test_fresh_chunk_promoted_for_recency_queries(self):
        from datetime import date, timedelta

        from dl_rag.retrieval.hybrid_retriever import _blend_recency

        old = make_retrieved(0, rerank=2.0, year=2024)
        old.chunk.metadata.published_date = date.today() - timedelta(days=730)
        fresh = make_retrieved(1, rerank=1.0, year=2026)
        fresh.chunk.metadata.published_date = date.today() - timedelta(days=20)

        # Old chunk is more "relevant" (higher rerank) but blend favours fresh.
        ordered = _blend_recency([old, fresh])
        assert ordered[0].chunk.chunk_index == 1

    def test_unknown_date_not_rewarded(self):
        from datetime import date, timedelta

        from dl_rag.retrieval.hybrid_retriever import _blend_recency

        dated = make_retrieved(0, rerank=1.0)
        dated.chunk.metadata.published_date = date.today() - timedelta(days=10)
        undated = make_retrieved(1, rerank=1.0)
        undated.chunk.metadata.published_date = None

        ordered = _blend_recency([undated, dated])
        assert ordered[0].chunk.chunk_index == 0

    async def test_entity_single_year_forces_latest_slice(self, settings: Settings):
        """'WES 2026'-style queries force the entity's newest chunks into context."""

        class SparseWithLatest(FakeSparse):
            def __init__(self):
                super().__init__([])
                self.latest_calls = []

            async def latest(self, query, top_k, filters=None, phrase=None):
                self.latest_calls.append(phrase)
                return [make_retrieved(9, year=2026, url="https://x/wes-2026/")]

        settings = settings.model_copy(update={"final_top_k": 3})
        sparse = SparseWithLatest()
        dense_results = [make_retrieved(i, year=2020) for i in range(3)]
        retriever = HybridRetriever(
            DenseRetriever(FakeEmbedder(), FakeVectorStore(dense_results)),
            sparse,
            NoopReranker(),
            settings,
        )
        analysis = _analysis(
            original_query="Who are the speakers at WES 2026?",
            entities=["World Education Summit"],
            time_range=TimeRange(from_year=2026, to_year=2026),
        )
        out = await retriever.retrieve(analysis)
        assert sparse.latest_calls == ["World Education Summit"]
        assert out[0].chunk.chunk_index == 9  # forced entity chunk leads

    async def test_analyzer_flags_recency(self, settings: Settings):
        from dl_rag.retrieval.query_understanding import HeuristicQueryAnalyzer

        analyzer = HeuristicQueryAnalyzer(settings)
        assert (await analyzer.analyze("When is the next WES event?")).recency_sensitive
        assert (await analyzer.analyze("Latest UGC guidelines")).recency_sensitive
        assert not (
            await analyzer.analyze("NEP implementation in Karnataka in 2021")
        ).recency_sensitive
