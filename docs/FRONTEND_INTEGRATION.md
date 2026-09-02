# digitalLEARNING Research API — Frontend Integration Guide

Ask questions in natural language about the digitalLEARNING archive (2005–present:
articles, magazines, interviews, policy coverage, and 991 event videos with
transcripts). Answers come back as **markdown with `[n]` citations** plus a
structured `sources[]` array of clickable links.

- **Base URL**: `https://wash.eletsonline.com`
- **Interactive explorer**: `https://wash.eletsonline.com/docs` ← try every endpoint live
- **Auth**: every `/api/*` call needs the header `X-API-Key: <key>` (obtain from the backend team — see *Security* below)
- **Content type**: `application/json`

---

## 1. Quick start

```bash
curl -X POST https://wash.eletsonline.com/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"query": "How has NEP evolved since 2020?"}'
```

---

## 2. 🔒 Security — read this first

**Never put the API key in browser JavaScript.** Anyone can read it via view-source
or DevTools and burn your OpenAI quota.

**Do this** — proxy through your own backend, which holds the key server-side:

```
Browser  ──►  your site's /api/ask  ──►  https://wash.eletsonline.com/api/chat
(no key)      (adds X-API-Key)            (key never leaves your server)
```

Example — Next.js route handler (`app/api/ask/route.ts`):

```ts
export async function POST(req: Request) {
  const { query, conversation_id } = await req.json();
  const res = await fetch("https://wash.eletsonline.com/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.DL_RAG_API_KEY!,   // server-side env var only
    },
    body: JSON.stringify({ query, conversation_id }),
  });
  return new Response(await res.text(), {
    status: res.status,
    headers: { "Content-Type": "application/json" },
  });
}
```

PHP/WordPress equivalent:

```php
$res = wp_remote_post('https://wash.eletsonline.com/api/chat', [
  'headers' => [
    'Content-Type' => 'application/json',
    'X-API-Key'    => getenv('DL_RAG_API_KEY'),
  ],
  'body'    => wp_json_encode(['query' => $query]),
  'timeout' => 60,
]);
```

> Direct browser → API calls are only acceptable behind an authenticated internal
> dashboard, and require the backend team to add your origin to `CORS_ORIGINS`.

---

## 3. `POST /api/chat` — the main endpoint

### Request

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `query` | string | ✅ | 2–2000 chars |
| `conversation_id` | string | — | Omit on the first message; reuse the returned id to keep context (last 20 turns) |
| `stream` | boolean | — | `true` → SSE token stream (§4). Default `false` |
| `filters` | object | — | See below |
| `top_k` | int (1–20) | — | Sources to ground on. Default 8 |

`filters` object:

| Field | Type | Example |
| --- | --- | --- |
| `year_from` / `year_to` | int | `2020`, `2024` |
| `content_types` | string[] | `["interview","video"]` |
| `authors` | string[] | `["Ravi Kumar"]` |
| `tags` | string[] | `["NEP"]` |

Valid `content_types`: `news`, `interview`, `video`, `policy`, `magazine_issue`,
`magazine_article`, `ranking`, `higher_education`, `school`, `government_news`,
`corporate`, `feature`, `opinion`, `special_story`, `cover_story`, `other`.

```jsonc
{
  "query": "Compare CBSE and State Board reforms",
  "conversation_id": "8f1c2e...",
  "stream": false,
  "filters": { "year_from": 2020, "content_types": ["policy", "news"] },
  "top_k": 8
}
```

### Response `200`

```jsonc
{
  "answer": "## Executive Summary\nNEP 2020 introduced … competency-based learning [1][3].",
  "sources": [
    {
      "index": 1,
      "title": "NEP Implementation in Karnataka",
      "url": "https://digitallearning.eletsonline.com/2022/06/…",
      "date": "2022-06-14",
      "content_type": "policy",
      "category": "Policy Matters",
      "issue": null,
      "author": "Ravi Kumar"
    }
  ],
  "confidence": 0.94,
  "confidence_band": "high",          // high | medium | low
  "query_type": "comparison",          // see §6
  "retrieved_documents": 8,
  "conversation_id": "8f1c2e…",        // send back on the next turn
  "message_id": "a7b3…",               // needed for feedback
  "latency_ms": 1420,
  "token_usage": { "prompt_tokens": 3100, "completion_tokens": 640, "total_tokens": 3740 }
}
```

### Rendering notes

- **`answer` is markdown** — render it with a markdown component (`react-markdown`,
  `marked`, …). It may contain `##` headings, **bold**, bullet lists, and comparison
  **tables** (enable GFM tables).
- **Citations** are inline `[1]`, `[2]`, sometimes `[1][3]`. Match `[n]` to the source
  where `source.index === n` and link it to `source.url`. A simple regex replace turns
  them into superscript links.
- **Always show the sources list** — grounded citations are the point of the product.
- `confidence_band` is a good UI signal (green/amber/grey chip).
- **No-evidence case**: when the archive can't support an answer you get
  `confidence: 0`, empty `sources`, and the exact text
  *"I could not find enough supporting evidence in the digitalLEARNING archive to answer this confidently."*
  Render it as an informational state, not an error.

---

## 4. Streaming (`stream: true`)

Returns **Server-Sent Events**. Recommended for chat UIs — first tokens appear in
~1–2 s instead of waiting for the full answer.

| Event | `data` | Meaning |
| --- | --- | --- |
| `meta` | JSON `{conversation_id, message_id, query_type, retrieved_documents}` | Fires once, before generation |
| `token` | raw text (not JSON) | Append to the answer as it arrives |
| `done` | JSON — the full `ChatResponse` (§3) | Final answer + sources + confidence |
| `error` | JSON `{message}` | Generation failed |

```ts
const res = await fetch("/api/ask-stream", {          // your proxy route
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query, stream: true }),
});

const reader = res.body!.getReader();
const decoder = new TextDecoder();
let buffer = "", answer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  const frames = buffer.split("\n\n");
  buffer = frames.pop() ?? "";
  for (const frame of frames) {
    const event = /^event:\s*(.*)$/m.exec(frame)?.[1];
    const data  = /^data:\s*([\s\S]*)$/m.exec(frame)?.[1] ?? "";
    if (event === "token") { answer += data; setAnswer(answer); }
    if (event === "done")  { const full = JSON.parse(data); setSources(full.sources); }
    if (event === "error") { showError(JSON.parse(data).message); }
  }
}
```

> Your proxy must stream too — return the upstream `res.body` directly and don't
> buffer it (in Next.js, avoid `await res.text()` for this route).

---

## 5. Other endpoints

### `POST /api/feedback` — thumbs up/down
```jsonc
{ "conversation_id": "8f1c2e…", "message_id": "a7b3…", "rating": "up", "comment": "optional" }
```
`rating` is `"up"` or `"down"`. → `{ "accepted": true, "message": "Thanks — feedback recorded." }`

### `GET /api/document/{id}?include_content=true` — full source document
Returns title, url, author, dates, tags, entities, keywords, `chunk_count`, and
(optionally) the full markdown body. Use it for a "view source" panel.

### `GET /health` — no auth
```json
{"status":"ok","version":"0.1.0","checks":{"postgres":"ok","redis":"ok","qdrant":"ok"}}
```

---

## 6. `query_type` values

The API auto-detects intent and shapes the answer accordingly — useful for UI hints:

`timeline` (year-by-year) · `comparison` (renders a table) · `trend` (era buckets) ·
`definition` · `policy` · `institution` · `person` · `interview` · `magazine` ·
`event` · `ranking` · `recommendation` · `statistics` · `summarization` · `general`

Video queries ("give me WES 2023 videos") return a linked list with
`content_type: "video"` sources pointing at YouTube.

---

## 7. Errors & limits

All errors share one shape:

```json
{ "error": "validation_error", "detail": "Request payload failed validation.",
  "request_id": "406c46b4…", "extra": {} }
```

| Status | `error` | Cause |
| --- | --- | --- |
| 401 | `authentication_error` | Missing/invalid `X-API-Key` |
| 422 | `validation_error` | Bad payload (e.g. `query` under 2 chars) |
| 429 | `rate_limited` | Over the limit — honour the `Retry-After` header |
| 404 | `not_found` | Unknown document/job id |
| 503 | `retrieval_error` / `generation_error` | Downstream hiccup — safe to retry once |

**Rate limit**: 60 requests per 60 s per API key. **Timeouts**: set your client to
**60 s** — answers usually land in ~10–20 s (CPU reranking + LLM), streaming feels
instant. Always include `request_id` when reporting a bug to the backend team.

---

## 8. Suggested UX

- Seed the input with example prompts: *"How has NEP evolved since 2020?"*,
  *"Compare CBSE and State Board reforms"*, *"Show interviews featuring AI in education"*,
  *"Give me WES 2023 video links"*.
- Show a "Sources (N)" section under every answer — that's the trust signal.
- Wire the thumbs up/down to `/api/feedback`; the backend dashboard reads it.
- Keep `conversation_id` in component state so follow-ups ("what about 2024?") work.
