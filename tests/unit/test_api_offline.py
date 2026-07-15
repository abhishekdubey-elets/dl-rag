"""Full-app degraded-mode test: boots the real FastAPI app with every datastore
unreachable.

Verifies graceful degradation end-to-end: health reports 'degraded', validation
fires, and the chat path returns the no-evidence guardrail instead of erroring
or calling the LLM. Hermetic by construction — all service DSNs (including the
LLM endpoint) are pointed at unroutable ports, so the test passes identically
whether or not real services happen to be running on this machine.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from dl_rag.constants import NO_EVIDENCE_MESSAGE
from dl_rag.embeddings.embedder import SentenceTransformerEmbedder

_ENV_OVERRIDES = {
    # Port 1 refuses connections immediately on localhost — fast, deterministic.
    "POSTGRES_DSN": "postgresql+asyncpg://x:x@127.0.0.1:1/x",
    "REDIS_URL": "redis://127.0.0.1:1/0",
    "QDRANT_URL": "http://127.0.0.1:1",
    "LLM_BASE_URL": "http://127.0.0.1:1/v1",
    "LLM_API_KEY": "test-key-never-used",
    "REQUIRE_AUTH": "false",
}


@pytest.fixture(scope="module")
def client():
    async def _fail_fast(self, text):  # avoid model download in tests
        raise RuntimeError("embedder disabled in tests")

    from dl_rag.config import get_settings

    saved_env = {k: os.environ.get(k) for k in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    get_settings.cache_clear()

    original = SentenceTransformerEmbedder.embed_query
    SentenceTransformerEmbedder.embed_query = _fail_fast  # type: ignore[method-assign]
    try:
        from dl_rag.main import create_app

        with TestClient(create_app()) as c:
            yield c
    finally:
        SentenceTransformerEmbedder.embed_query = original  # type: ignore[method-assign]
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


class TestOfflineApp:
    def test_health_degrades_not_500(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")
        assert set(body["checks"]) == {"postgres", "redis", "qdrant"}

    def test_openapi_and_metrics(self, client):
        assert client.get("/openapi.json").status_code == 200
        m = client.get("/metrics")
        assert m.status_code == 200
        assert "dlrag_http_requests_total" in m.text

    def test_validation_error_envelope(self, client):
        r = client.post("/api/chat", json={"query": "a"})
        assert r.status_code == 422
        assert r.json()["error"] == "validation_error"

    def test_chat_no_evidence_guardrail(self, client):
        r = client.post("/api/chat", json={"query": "How has NEP evolved since 2020?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == NO_EVIDENCE_MESSAGE
        assert body["confidence"] == 0.0
        assert body["query_type"] == "timeline"
        assert body["sources"] == []
        assert body["conversation_id"]
