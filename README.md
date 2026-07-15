# digitalLEARNING RAG API

A production-grade **Retrieval-Augmented Generation** service over the
[digitalLEARNING](https://digitallearning.eletsonline.com/) archive — 20+ years
(since 2005) of magazines, interviews, news, rankings, and policy coverage on
Indian education and edtech.

It answers natural-language questions like an **education-policy analyst**, not a
search box: grounded, structured, and cited back to the original articles.

> _"How has NEP evolved since 2020?"_ → a chronological brief with year-by-year
> developments, differing viewpoints where they exist, inline `[n]` citations,
> and clickable links to the source articles — ending with **Key Takeaways**.

---

## Why it's more than "vector search"

| Concern | What this does |
| --- | --- |
| Recall | **Hybrid retrieval**: dense (Qdrant) + sparse Postgres FTS + metadata filtering + knowledge-graph expansion, fused with Reciprocal Rank Fusion |
| Precision | **Cross-encoder reranking** (`bge-reranker` / MS-MARCO) over the top candidates, then token-budgeted **context compression** |
| Understanding | **Query classification** (timeline / comparison / trend / definition / ranking / interview / …) drives *different* retrieval + prompt strategies |
| Grounding | Answers cite only supplied sources; a low-evidence guardrail refuses to fabricate |
| Structure | Analyst-style sections: Executive Summary → Key Findings → Evidence → Historical Context → Current Situation → Future Outlook → Key Takeaways |
| Memory | Redis-backed conversation memory: last 20 turns verbatim + a rolling LLM summary |

---

## Architecture

```
                         ┌──────────────────────────── FastAPI ────────────────────────────┐
  POST /api/chat  ─────► │  auth → rate-limit → request-ctx → ChatService                   │
  (SSE optional)         │                                                                  │
                         │   1. QueryAnalyzer      intent + entities + time-range           │
                         │   2. HybridRetriever    ┌── dense (Qdrant) ─┐                     │
                         │                         ├── sparse (PG FTS) ─┼─ RRF ─ rerank ─ compress
                         │                         └── KG expansion  ──┘                     │
                         │   3. AnswerGenerator    analyst prompt → LLM (OpenAI-compatible)  │
                         │   4. Citations          [n] markers → clickable sources          │
                         │   5. ConversationMemory (Redis)                                   │
                         └──────────────────────────────────────────────────────────────────┘

  Ingestion (offline):  WordPress crawler → extract → clean markdown → semantic chunk
                        → entity/keyword/relation extraction → embed → Postgres + Qdrant + KG
```

**Stores**
- **PostgreSQL** — documents, chunks (+ generated `tsvector` for FTS), entities, relations, feedback, query logs, crawl jobs.
- **Qdrant** — dense vectors with payload filters (year / content-type / author / tags).
- **Redis** — conversation memory, answer/embedding cache, rate-limit windows.

**Design**: hexagonal / ports-and-adapters. Services depend on `typing.Protocol`
interfaces (`src/dl_rag/protocols.py`); concrete adapters are injected by the DI
container (`src/dl_rag/api/deps.py`). SOLID, fully async, typed, no monolith files.

---

## Project layout

```
src/dl_rag/
├── config.py              # typed Settings (pydantic-settings)
├── constants.py           # domain gazetteer, relation triggers, intent hints
├── logging_config.py      # structlog
├── protocols.py           # Embedder / VectorStore / SparseRetriever / Reranker / LLMClient / Cache / KnowledgeGraph
├── exceptions.py
├── models/                # enums, domain models, API schemas
├── utils/                 # token counting, text helpers
├── db/                    # async SQLAlchemy engine + ORM
├── repositories/          # repository pattern over Postgres
├── cache/                 # Redis adapter
├── vectorstore/           # Qdrant adapter
├── embeddings/            # sentence-transformers adapter
├── adapters/              # PostgresFTSRetriever, PostgresKnowledgeGraph
├── retrieval/             # query understanding, dense, fusion, reranker, compression, hybrid
├── generation/            # llm_client, prompts, citations, answer_generator
├── memory/                # conversation memory
├── ingestion/             # crawler, extractors, chunking, entities, ocr
├── knowledge_graph/       # KG builder
├── services/              # chat / ingestion / admin orchestration
├── evaluation/            # faithfulness / recall / groundedness / citation-precision
├── api/                   # deps, security, middleware, errors, routers
└── main.py                # FastAPI app + lifespan
```

---

## Quickstart

### Option A — Docker (everything)

```bash
cp .env.example .env
#  edit .env → set LLM_API_KEY (and LLM_BASE_URL/LLM_MODEL if not OpenAI)
docker compose up --build
# API docs:      http://localhost:8000/docs
# health/metrics: http://localhost:8000/health , /metrics
```

### Option B — Local (Poetry)

```bash
# 1. start the datastores only
docker compose up -d postgres redis qdrant
# 2. install + run the API
poetry install --with dev
cp .env.example .env         # POSTGRES_DSN/REDIS_URL/QDRANT_URL already point at localhost
poetry run uvicorn dl_rag.main:app --reload
```

> First start downloads the embedding + reranker models (hundreds of MB). Set
> `EMBEDDING_MODEL` / `RERANKER_MODEL` to the small dev defaults for speed, or the
> `bge-large` / `bge-reranker-large` pair for best quality.

---

## Ingesting the archive

```bash
# discover + crawl + chunk + embed + index (see --help for scope flags)
poetry run dl-ingest --max-pages 200                 # a first slice
poetry run dl-ingest --content-type interview        # only interviews
poetry run dl-ingest --full                          # the whole archive (long-running)

# or trigger over HTTP:
curl -X POST localhost:8000/api/index \
  -H 'Content-Type: application/json' \
  -d '{"full_crawl": true, "max_pages": 500}'
```

Ingestion is **idempotent** (content-hash de-duplication) and safe to re-run.
Please crawl your own property responsibly — concurrency and a politeness delay
are configurable (`CRAWLER_*`), and `robots.txt` is honoured by default.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/chat` | Ask a question. `stream: true` → SSE token stream. |
| POST | `/api/index` | Trigger crawl + ingestion (async job). |
| POST | `/api/reindex` | Re-embed / rebuild the index. |
| GET | `/api/document/{id}` | Fetch a source document + metadata. |
| POST | `/api/feedback` | Thumbs up/down + comment on an answer. |
| GET | `/api/admin/stats` | Index size, docs by type/year, failures. |
| GET | `/api/admin/insights` | Popular questions, latency, citation frequency, feedback. |
| GET | `/health` | Liveness + dependency checks. |
| GET | `/metrics` | Prometheus metrics. |

### `POST /api/chat`

```jsonc
// request
{ "query": "How has NEP evolved since 2020?", "conversation_id": "abc", "stream": false }
```

```jsonc
// response
{
  "answer": "**Executive Summary** … competency-based learning from 2021 [1][3] …",
  "sources": [
    { "index": 1, "title": "NEP Implementation in Karnataka", "url": "https://…",
      "date": "2022-06-14", "content_type": "policy", "issue": null }
  ],
  "confidence": 0.87,
  "confidence_band": "high",
  "query_type": "timeline",
  "retrieved_documents": 8,
  "conversation_id": "abc",
  "message_id": "…",
  "latency_ms": 1420,
  "token_usage": { "prompt_tokens": 3100, "completion_tokens": 640 }
}
```

Streaming (`stream: true`) emits SSE events: `token` deltas, then a final
`done` event carrying `sources`, `confidence`, and usage.

Auth: send `X-API-Key: <key>` when `REQUIRE_AUTH=true`.

---

## Retrieval, by query type

| Query type | Retrieval tweak | Answer shape |
| --- | --- | --- |
| Timeline | pull across years, sort chronologically | year-by-year sections |
| Comparison | retrieve each subject separately | comparison **table** + synthesis |
| Trend | bucket eras 2005–2010 … 2024–present | trajectory narrative |
| Interview | bias to `content_type = interview` | quotes + attribution |
| Ranking / Statistics | prioritise ranking/figure-bearing chunks | ordered list / figures |
| Definition | tight top-k | crisp definition + context |

---

## Evaluation

```bash
poetry run python -m dl_rag.evaluation.run --dataset eval/questions.json
# tiers: --retrieval-only (no LLM), --no-judge (generate but skip LLM-as-judge)
```

Reports **answer faithfulness**, **context recall**, **groundedness**,
**citation precision**, **top-k recall**, **latency**, and **token usage** — an
LLM-as-judge harness with deterministic metrics where possible.

---

## Testing & quality

```bash
make test          # unit tests
make test-int      # integration tests (needs live postgres/redis/qdrant)
make lint          # ruff
make typecheck     # mypy --strict
make compile       # fast syntax check
```

CI (GitHub Actions) runs lint + compile on every push and the full suite against
service containers.

---

## Configuration

All configuration is environment-driven and validated at startup — see
[`.env.example`](.env.example) for the full annotated list. Key knobs:

- `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` — any OpenAI-compatible endpoint
  (OpenAI, Azure, vLLM, Together, Groq, Ollama …).
- `EMBEDDING_MODEL`, `RERANKER_MODEL` — quality/speed trade-off.
- `RETRIEVAL_CANDIDATES` (40) → `FINAL_TOP_K` (8), `RRF_K`, weights.
- `REQUIRE_AUTH`, `RATE_LIMIT_*`.

---

## Status & roadmap

This repository is a complete, runnable foundation. Subsystems that are wired
end-to-end vs. those that benefit from live-data iteration are called out in
[`docs/STATUS.md`](docs/STATUS.md). The crawler is written against the
digitalLEARNING WordPress structure (WP REST API + Yoast sitemaps) but selectors
should be tuned against a live sample before a full archive crawl.
