"""Centralised, typed application configuration.

All settings are read from environment variables (or a local ``.env`` file) and
validated once at process start. Import :func:`get_settings` everywhere — it is
cached so the environment is parsed a single time.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    """Strongly-typed settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "digitalLEARNING RAG API"
    environment: Environment = "development"
    debug: bool = True
    log_level: str = "INFO"
    log_json: bool = False

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    api_prefix: str = "/api"
    # Serve the built-in demo chat page at "/" (disable for API-only deployments).
    serve_ui: bool = True
    # Comma-separated origins allowed to call the API from browsers ("" = no CORS).
    cors_origins: str = ""

    # --- Security ---
    api_keys: str = "dev-local-key"
    require_auth: bool = False
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # --- PostgreSQL ---
    postgres_dsn: str = "postgresql+asyncpg://dl:dl@localhost:5432/dl_rag"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "dl_chunks"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # --- Reranker ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"

    # --- Retrieval ---
    retrieval_candidates: int = 40
    final_top_k: int = 8
    rrf_k: int = 60
    vector_weight: float = 1.0
    sparse_weight: float = 1.0
    kg_expansion_enabled: bool = True
    # Floor on raw cross-encoder logits. ms-marco models emit NEGATIVE logits
    # even for relevant pairs, so the floor is disabled by default; only set a
    # real value after calibrating against your reranker's score distribution.
    min_rerank_score: float = -1_000_000.0
    context_max_tokens: int = 6000

    # --- LLM (OpenAI-compatible) ---
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-replace-me"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1500
    llm_timeout_seconds: int = 60

    # --- Crawler ---
    crawler_base_url: str = "https://digitallearning.eletsonline.com"
    crawler_user_agent: str = (
        "dLEARNING-RAG-Indexer/0.1 (+https://digitallearning.in)"
    )
    crawler_concurrency: int = 4
    crawler_delay_seconds: float = 1.0
    crawler_respect_robots: bool = True
    crawler_max_pages: int = 0
    crawler_timeout_seconds: int = 30

    # --- YouTube ingestion ---
    youtube_channel_url: str = "https://www.youtube.com/user/eletsvideos"
    # Optional Google YouTube Data API v3 key — used for the catalog when set;
    # otherwise yt-dlp lists the channel keylessly.
    youtube_api_key: str | None = None
    # Optional keyed transcript service (used before the keyless library when
    # set). URL template with {video_id}; key sent via the named header.
    transcript_api_url: str | None = None
    transcript_api_key: str | None = None
    transcript_api_key_header: str = "x-api-key"
    transcript_languages: str = "en,en-IN,en-GB,hi"
    youtube_max_videos: int = 500

    # --- Supabase transcript source (optional import) ---
    supabase_db_host: str | None = None
    supabase_db_port: int = 5432
    supabase_db_name: str = "postgres"
    supabase_db_user: str | None = None
    supabase_db_password: str | None = None

    # --- Chunking ---
    chunk_max_tokens: int = 600
    chunk_overlap_ratio: float = 0.15
    chunk_min_tokens: int = 64

    # --- Conversation memory ---
    memory_max_turns: int = 20
    memory_ttl_seconds: int = 604_800

    # --- Caching ---
    cache_ttl_seconds: int = 3600
    answer_cache_enabled: bool = True
    embedding_cache_enabled: bool = True

    # ------------------------------------------------------------------ #
    @field_validator("api_keys")
    @classmethod
    def _strip_keys(cls, v: str) -> str:
        return v.strip()

    @field_validator(
        "qdrant_api_key", "youtube_api_key", "transcript_api_url",
        "transcript_api_key", mode="before",
    )
    @classmethod
    def _empty_key_is_none(cls, v: str | None) -> str | None:
        return v or None

    @property
    def api_key_set(self) -> set[str]:
        """Parsed set of accepted API keys."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def chunk_overlap_tokens(self) -> int:
        return int(self.chunk_max_tokens * self.chunk_overlap_ratio)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, cached :class:`Settings` instance."""
    return Settings()
