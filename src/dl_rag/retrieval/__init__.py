"""Retrieval orchestration layer.

Assembles the query-understanding, dense/sparse retrieval, fusion, reranking,
context-compression, and hybrid-orchestration components. All classes depend
only on the Protocols in :mod:`dl_rag.protocols`; concrete infrastructure is
supplied by the integrator via dependency injection.
"""

from __future__ import annotations

from dl_rag.retrieval.compression import ContextCompressor
from dl_rag.retrieval.dense import DenseRetriever
from dl_rag.retrieval.fusion import reciprocal_rank_fusion
from dl_rag.retrieval.hybrid_retriever import HybridRetriever
from dl_rag.retrieval.query_understanding import HeuristicQueryAnalyzer
from dl_rag.retrieval.reranker import CrossEncoderReranker, NoopReranker

__all__ = [
    "ContextCompressor",
    "CrossEncoderReranker",
    "DenseRetriever",
    "HeuristicQueryAnalyzer",
    "HybridRetriever",
    "NoopReranker",
    "reciprocal_rank_fusion",
]
