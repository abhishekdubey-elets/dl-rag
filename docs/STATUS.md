# Implementation Status

An honest, subsystem-by-subsystem account of what is **fully wired and verified**
versus what **needs live-data iteration** before a production launch. Nothing in
this repo is a stub — every module is a real implementation — but some code paths
can only be hardened against the live site / live traffic.

_Last updated: 2026-07-11._

## Verification performed

- `python -m compileall` clean across the whole package.
- Full application graph imports and boots (FastAPI lifespan) with **zero**
  external services — degrades gracefully, never 500s.
- **67 unit tests passing** (query understanding, chunking, fusion, compression,
  citations, answer generation, memory, extraction, entities, metrics, auth,
  offline full-app behaviour), **3 integration tests** (Postgres FTS roundtrip,
  Redis cache, Qdrant vectors) that auto-skip locally and run in CI with services.
- Offline end-to-end check: `/api/chat` with all datastores down returns the
  no-evidence guardrail with `confidence: 0` — the system cannot hallucinate an
  answer it has no sources for.

## Subsystem status

| Subsystem | State | Notes |
| --- | --- | --- |
| Config / logging / DI / errors | ✅ Solid | Typed settings, structlog, container in `api/deps.py`. |
| API surface (chat, index, document, feedback, admin, health, metrics) | ✅ Solid | All routes registered; SSE streaming via `sse-starlette`. |
| Auth + rate limiting | ✅ Solid | API-key header; Redis fixed-window limiter (fail-open by design). |
| Postgres layer (ORM, repositories, FTS) | ✅ Solid | Generated `tsvector` + GIN; verified DDL. Integration-tested in CI. |
| Qdrant vector store | ✅ Solid | Payload indexes for year/type/author filters; uuid5 point ids. |
| Redis cache + conversation memory | ✅ Solid | 20-turn window + rolling LLM summary; guarded end-to-end. |
| Hybrid retrieval (RRF fusion, filters, KG expansion) | ✅ Solid | Backend failures degrade to remaining backends; unit-tested. |
| Cross-encoder reranking | ✅ Code-complete | `NoopReranker` unit-tested; the real `CrossEncoderReranker` needs the model downloaded at first run (`RERANKER_MODEL`). |
| Embeddings | ✅ Code-complete | Lazy-loaded sentence-transformers; first run downloads the model. |
| Query understanding | ✅ Solid | Deterministic heuristic classifier + gazetteer entities + time-range parsing; 12 unit tests. Optional LLM hook reserved. |
| Answer generation + citations + guardrails | ✅ Solid | Analyst prompt per query type; citation dedup; confidence bands; no-evidence refusal (LLM is provably not called without sources). |
| Ingestion pipeline (orchestrator) | ✅ Solid | Idempotent per-document: replace chunks + vectors, upsert doc, feed KG. |
| **WordPress crawler** | ✅ Validated live | Verified against the live site 2026-07-11: URL discovery works (REST API + sitemaps) and extraction was clean on sampled pages (title/type/author/date/markdown all correct). Caveat: sampled pages were recent; very old (2005-era) pages may still need selector checks if extraction-quality issues show up in `pages_failed`. |
| PDF / OCR | ⚠️ Optional deps | pypdf + pdfplumber wired; OCR fallback requires `tesseract` + `poppler` binaries (`poetry install --with ocr`, uncomment Dockerfile lines). |
| Knowledge graph | ✅ Solid for scope | Gazetteer + trigger-phrase relation extraction (high precision, modest recall). Statistical NER (spaCy) is optional and off unless installed. |
| Evaluation harness | ✅ Solid | Deterministic metrics always; LLM-as-judge (faithfulness/groundedness) when an LLM is reachable. Seed dataset in `eval/questions.json` — extend with gold URLs after first ingest. |
| Docker / compose / CI | ✅ Written | Compose stack + healthchecks; CI lints, compiles, tests with service containers. `poetry.lock` should be generated (`poetry lock`) on first install. |
| Alembic migrations | ⚠️ Recommended next | Schema is created via `Database.create_all()` (fine for first deploy). Generate an Alembic baseline before the schema evolves in production. |

## Live-run fixes (2026-07-11)

First full live run (real crawl + real queries) surfaced and fixed four integration
bugs that offline testing could not catch:

1. **qdrant-client ≥1.13** removed the classic `.search()` — `QdrantVectorStore`
   now uses `query_points()` with a legacy fallback.
2. **FTS AND-semantics**: `websearch_to_tsquery` requires *every* term, so
   natural-language questions matched nothing. `fts_search` now falls back to an
   OR-relaxed tsquery when the strict pass returns zero rows.
3. **Entity over-filtering**: analyzer-extracted entities were applied as a hard
   filter, killing recall for untagged-but-relevant chunks. Entities are now a
   caller-only filter; the KG expansion pass provides the entity boost instead.
4. **tenacity × openai SDK**: `AsyncRetrying(fn, ...)` didn't await the SDK's
   awaitable-returning `create`, yielding empty answers. Replaced with the
   explicit `async for attempt` pattern.

Observed live: hybrid retrieval returns dense=40/sparse=40/kg=40 → rerank → 8;
answers score 94–99 % confidence with correct per-type structure (institution,
timeline) and accurate citations. Warm end-to-end latency ≈ 13 s on CPU
(cross-encoder rerank + LLM generation dominate; see targets below).

## Recency-aware answering (2026-07-13)

User feedback ("not satisfied — WES answered from stale coverage / refused") drove a
second hardening round. "When is the next X?" is an **argmax-by-date** question, and
similarity ranking systematically buries the newest coverage. The shipped design:

1. **Graded evidence policy** (prompts): answer from what the archive documents and
   state its boundary; refuse only on genuinely irrelevant sources.
2. **Temporal grounding**: current date injected into the prompt; every source header
   carries a code-computed relative age ("2 years ago") so the LLM never does date math.
3. **Recency signals in retrieval** (`recency_sensitive` on QueryAnalysis, set for
   next/latest/upcoming/current queries):
   - canonical-entity query expansion ("WES" → "World Education Summit");
   - an extra dense+sparse "fresh slice" pass filtered to the last ~18 months;
   - rerank window widened so fresh-pass hits are always scored;
   - relevance×freshness blend ordering (`0.65·sigmoid(rerank) + 0.35·exp(−age)`);
   - a **guaranteed latest slice**: newest chunks phrase-matching the query entity are
     force-included into the final context (one per document, title-matches first).
4. **Index hygiene**: 67 utility/listing pages (video galleries, About, Subscribe…)
   purged — they carried crawl-date freshness and polluted recency ranking; the crawler
   denylist now skips them. FTS gained an OR-relaxed top-up (strict AND starves on
   natural-language questions), and `MIN_RERANK_SCORE` defaults to disabled (ms-marco
   logits are negative even for relevant pairs).

Verified live: "When is the next WES event?" → WES 2026 at the Hyatt Regency, New
Delhi, at 96.8 % confidence, citing the two dedicated announcement articles, with past
editions correctly framed in past tense.

Extension (2026-07-13): the guaranteed slice also fires for **entity + single-year**
queries ("speakers at WES 2026") — edition-scoped questions where lexically-loud
lookalikes (other 2026 summits) otherwise outrank the entity's own articles. Verified:
the speakers query now grounds on the WES-2026 announcements and answers honestly that
named speakers aren't yet published, instead of refusing.

**Editorial finding surfaced by the system**: the archive's two WES-2026 announcements
disagree on the event date — 15–16 September 2026 (article of 2026-06-22) vs 12–13
October 2026 (article of 2026-06-29). Answers currently follow the newer source; the
site content should be reconciled.

**KG speaker extension (2026-07-15, shipped)**: spaCy NER + a deterministic honorific
person detector ("Prof./Dr./Shri X" — sm-NER misses many Indian names), a person-gated
`spoke_at` predicate (PERSON subject, venue-like object; no places/role-words/mangled
spans; newline-bounded sentences to stop headline contamination), and the
`dl-kg-extract` CLI that rebuilds the graph from stored documents (~30 min, no
re-crawl). Full-archive rebuild: **88,724 entities / 12,420 relations / 166 spoke_at
edges** across 17,812 documents; document entity tags refreshed (WES now tagged).
Verified: "Who has spoken at WES over the years?" → timeline answer naming
Dr. APJ Abdul Kalam (2011 inaugural) et al., 98.5 % confidence. Re-run the CLI after
gazetteer/trigger changes.

## YouTube video ingestion (2026-07-15, shipped)

Elets' channel (`youtube.com/user/eletsvideos`, discovered from the site's own links)
is ingested as first-class VIDEO documents: title + description + **transcript** as the
searchable body, flowing through the standard pipeline (chunking, embedding, KG,
citations → clickable watch URLs). Components:

- `ingestion/youtube/` — catalog (yt-dlp keyless, or YouTube Data API v3 when
  `YOUTUBE_API_KEY` is set), transcripts (keyed provider via `TRANSCRIPT_API_URL/KEY`
  first when configured, keyless youtube-transcript-api ≥1.x fallback), document
  builder. `dl-ingest-youtube` CLI with `--match/--skip-existing/--max-videos`.
- Query routing: video-intent words ("videos", "watch", "recording", "youtube")
  narrow retrieval to `content_type=video`; a video-specific prompt renders answers
  as a linked list (title, date, one-liner, [n] citation) instead of the analyst
  sections. Combines with the entity+year guarantee ("wes 2023 videos").

Verified live: "give me wes 2026 videos link" and "give me wes 2023 videos link" →
titled, dated YouTube links. Coverage: **991 videos indexed (2016–2026)** from the
1,019 WES-titled uploads found across the channel's 8,000+ videos. Re-run
`dl-ingest-youtube --skip-existing` periodically (or wire it into the reindex job)
to pick up new uploads.

**Transcripts via the user's Supabase (2026-07-15)**: `public.chunks`
(source='youtube') in the user's Supabase holds 41k timestamped transcript segments
for 5,889 Elets videos. `dl-import-supabase` stitches them per video and re-ingests
matched documents (text only — their embeddings are a different model, so we
re-embed locally). Result: **527/991 videos now carry full transcripts** (505
imported in one run, 0 failures; the remainder have no transcript source anywhere).
Spoken content is now searchable and feeds the knowledge graph (entities + spoke_at
edges from talks). The youtube-transcript.io key (no credits) is no longer needed
for matched videos. Index size after import: ~29.4k chunks.

## Performance targets (from the spec)

The <2s average / <700ms first-token targets are **achievable but must be measured
after ingest** — they depend on hardware, model choices, and index size:

- Use `bge-small` embeddings + MiniLM reranker on CPU for latency; upgrade to
  `bge-large` + `bge-reranker-large` on GPU for quality.
- Prometheus histograms (`dlrag_chat_latency_seconds`,
  `dlrag_retrieval_latency_seconds`, `dlrag_llm_latency_seconds`) are already in
  place to verify the budget per stage.
- 100 concurrent users: run multiple uvicorn workers behind a load balancer;
  Postgres/Qdrant/Redis all pool connections. Load-test with the eval harness
  before committing to the SLO.

## Recommended launch sequence

1. `docker compose up -d postgres redis qdrant` → `poetry install --with dev`.
2. `poetry run dl-crawl --limit 20 --fetch` — eyeball extraction quality on live pages; adjust selectors if the theme differs.
3. `poetry run dl-ingest --max-pages 200` — first slice; check `/api/admin/stats`.
4. Ask real questions; inspect citations; tune `RETRIEVAL_*` knobs.
5. `python -m dl_rag.evaluation.run --retrieval-only` → then full eval with judge.
6. Full crawl (`--full`), then enable `REQUIRE_AUTH=true` and deploy.
