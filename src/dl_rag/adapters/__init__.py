"""Adapters wiring repositories to the retrieval/KG protocols."""

from __future__ import annotations

from dl_rag.adapters.knowledge_graph import PostgresKnowledgeGraph
from dl_rag.adapters.sparse_retriever import PostgresFTSRetriever

__all__ = ["PostgresFTSRetriever", "PostgresKnowledgeGraph"]
