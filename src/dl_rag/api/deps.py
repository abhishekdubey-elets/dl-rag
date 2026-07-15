"""Composition root / dependency-injection container.

``build_container`` constructs every singleton once; ``main.py`` stashes it on
``app.state.container`` during the lifespan. The ``get_*`` callables are the
FastAPI dependencies the routers depend on — nothing is imported by concrete
type inside a router.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from dl_rag.adapters.knowledge_graph import PostgresKnowledgeGraph
from dl_rag.adapters.sparse_retriever import PostgresFTSRetriever
from dl_rag.cache.redis_cache import RedisCache
from dl_rag.config import Settings
from dl_rag.db.database import Database
from dl_rag.embeddings.embedder import SentenceTransformerEmbedder
from dl_rag.generation.answer_generator import AnswerGenerator
from dl_rag.generation.llm_client import OpenAICompatibleLLM
from dl_rag.ingestion.chunking.semantic_chunker import SemanticChunker
from dl_rag.ingestion.crawler.wordpress import WordPressCrawler
from dl_rag.ingestion.entities.extractor import EntityExtractor
from dl_rag.ingestion.pipeline import IngestionPipeline
from dl_rag.memory.conversation_memory import ConversationMemory
from dl_rag.retrieval.compression import ContextCompressor
from dl_rag.retrieval.dense import DenseRetriever
from dl_rag.retrieval.hybrid_retriever import HybridRetriever
from dl_rag.retrieval.query_understanding import HeuristicQueryAnalyzer
from dl_rag.retrieval.reranker import CrossEncoderReranker
from dl_rag.services.admin_service import AdminService
from dl_rag.services.chat_service import ChatService
from dl_rag.services.ingestion_service import IngestionService
from dl_rag.vectorstore.qdrant_store import QdrantVectorStore


@dataclass(slots=True)
class Container:
    settings: Settings
    db: Database
    cache: RedisCache
    vector_store: QdrantVectorStore
    embedder: SentenceTransformerEmbedder
    llm: OpenAICompatibleLLM
    knowledge_graph: PostgresKnowledgeGraph
    analyzer: HeuristicQueryAnalyzer
    retriever: HybridRetriever
    generator: AnswerGenerator
    memory: ConversationMemory
    pipeline: IngestionPipeline
    chat_service: ChatService
    ingestion_service: IngestionService
    admin_service: AdminService


def build_container(settings: Settings) -> Container:
    db = Database(settings.postgres_dsn)
    cache = RedisCache(settings.redis_url)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
        settings.embedding_query_prefix,
    )
    vector_store = QdrantVectorStore(
        settings.qdrant_url, settings.qdrant_collection, settings.qdrant_api_key
    )
    llm = OpenAICompatibleLLM(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        settings.llm_temperature,
        settings.llm_max_tokens,
        settings.llm_timeout_seconds,
    )

    # Retrieval wiring (all against protocols).
    sparse = PostgresFTSRetriever(db.sessionmaker)
    knowledge_graph = PostgresKnowledgeGraph(db.sessionmaker)
    dense = DenseRetriever(embedder, vector_store)
    reranker = CrossEncoderReranker(settings.reranker_model, settings.reranker_device)
    compressor = ContextCompressor(settings.context_max_tokens)
    retriever = HybridRetriever(
        dense, sparse, reranker, settings,
        knowledge_graph=knowledge_graph, compressor=compressor,
    )

    analyzer = HeuristicQueryAnalyzer(settings)
    generator = AnswerGenerator(llm, settings)
    memory = ConversationMemory(cache, llm, settings)

    # Ingestion wiring.
    pipeline = IngestionPipeline(
        settings=settings,
        db=db,
        vector_store=vector_store,
        embedder=embedder,
        knowledge_graph=knowledge_graph,
        crawler=WordPressCrawler(settings),
        chunker=SemanticChunker(settings),
        entity_extractor=EntityExtractor(settings),
    )

    chat_service = ChatService(
        analyzer=analyzer, retriever=retriever, generator=generator,
        memory=memory, db=db, settings=settings,
    )
    ingestion_service = IngestionService(pipeline, db, settings)
    admin_service = AdminService(db, vector_store)

    return Container(
        settings=settings,
        db=db,
        cache=cache,
        vector_store=vector_store,
        embedder=embedder,
        llm=llm,
        knowledge_graph=knowledge_graph,
        analyzer=analyzer,
        retriever=retriever,
        generator=generator,
        memory=memory,
        pipeline=pipeline,
        chat_service=chat_service,
        ingestion_service=ingestion_service,
        admin_service=admin_service,
    )


# --- FastAPI dependency accessors ---------------------------------------- #
def get_container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


def get_chat_service(request: Request) -> ChatService:
    return get_container(request).chat_service


def get_ingestion_service(request: Request) -> IngestionService:
    return get_container(request).ingestion_service


def get_admin_service(request: Request) -> AdminService:
    return get_container(request).admin_service


def get_db(request: Request) -> Database:
    return get_container(request).db
