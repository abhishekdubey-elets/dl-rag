# digitalLEARNING Research API — CMS & Frontend Integration Guide

_Version 1.1 · 2026-09-03 · for the digitalLEARNING (WordPress) team_

The Research API answers natural-language questions about the digitalLEARNING
archive — 18,800+ documents since 2005 (articles, magazine issues, interviews,
policy coverage, rankings) plus ~990 Elets event videos, many with full
transcripts. Every answer is grounded: markdown text with inline `[n]` citations
and a structured `sources[]` list of clickable article / video links.

| | |
| --- | --- |
| **Base URL** | `https://wash.eletsonline.com` |
| **Interactive explorer** | `https://wash.eletsonline.com/docs` (try every endpoint live) · OpenAPI JSON at `/openapi.json` |
| **Auth** | every `/api/*` call needs `X-API-Key: <key>` (ask the backend team; never ship it to browsers) |
| **Content type** | `application/json` (responses: JSON, or `text/event-stream` when `stream: true`) |
| **Freshness** | new site articles + education-vertical channel videos are indexed automatically every 24 h; a publish hook (§3.6) indexes an article within seconds |

---

## 1. Quick start

```bash
curl -X POST https://wash.eletsonline.com/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"query": "How has NEP evolved since 2020?"}'
```

You get back an `answer` (markdown), `sources[]`, a `confidence` score, and a
`conversation_id` to send with the next question so follow-ups keep context.

---

## 2. Security & architecture (read before writing code)

**The API key must live on the server, never in page JavaScript.** Anything in a
browser is public (view-source, DevTools) and would let anyone drain the LLM quota.

```
Visitor's browser ──► your WordPress endpoint  ──► https://wash.eletsonline.com/api/chat
   (no key)            /wp-json/dl-ai/v1/ask          (X-API-Key added server-side)
                       nonce · per-visitor limit
                       optional response cache
```

Rules of the road:

- The CMS proxy adds the key, enforces a **per-visitor rate limit** (the API's own
  limit is 60 requests / minute for the whole key — one abusive visitor must not
  exhaust it for everyone), and forwards only the fields listed in §4.
- Send a WordPress **nonce** with widget requests so the endpoint can't be scripted
  from other sites.
- Direct browser → API calls are only acceptable for an authenticated internal
  dashboard, and require the backend team to add that origin to `CORS_ORIGINS`
  (the API allows `GET, POST` with `Content-Type` and `X-API-Key` headers).
- Log the `request_id` from every error — it lets the backend team find the exact
  server-side trace.

---

## 3. WordPress integration recipe

Everything below is a single **must-use plugin** file
(`wp-content/mu-plugins/dl-ai-assistant.php`) plus one JS file and a shortcode.
Adapt names freely.

### 3.1 Configuration

Put the key in `wp-config.php` (never in the database or theme files):

```php
define('DL_RAG_API_BASE', 'https://wash.eletsonline.com');
define('DL_RAG_API_KEY',  'paste-key-here');
```

### 3.2 Proxy endpoint — `/wp-json/dl-ai/v1/ask`

```php
<?php
/**
 * Plugin Name: digitalLEARNING AI Assistant (API proxy + widget)
 */
add_action('rest_api_init', function () {
    register_rest_route('dl-ai/v1', '/ask', [
        'methods'             => 'POST',
        'permission_callback' => 'dl_ai_permission',
        'callback'            => 'dl_ai_ask',
        'args' => [
            'query'           => ['required' => true, 'type' => 'string'],
            'conversation_id' => ['type' => 'string'],
            'filters'         => ['type' => 'object'],
        ],
    ]);
    register_rest_route('dl-ai/v1', '/feedback', [
        'methods'             => 'POST',
        'permission_callback' => 'dl_ai_permission',
        'callback'            => 'dl_ai_feedback',
    ]);
});

function dl_ai_permission(WP_REST_Request $req) {
    // Same-site only: the widget sends the REST nonce (works for logged-out visitors too).
    if (!wp_verify_nonce($req->get_header('X-WP-Nonce'), 'wp_rest')) {
        return new WP_Error('forbidden', 'Bad nonce', ['status' => 403]);
    }
    // Per-visitor limit: 10 questions per 5 minutes, keyed by IP.
    $key   = 'dl_ai_rl_' . md5($_SERVER['REMOTE_ADDR'] ?? 'na');
    $count = (int) get_transient($key);
    if ($count >= 10) {
        return new WP_Error('rate_limited', 'Please wait a few minutes.', ['status' => 429]);
    }
    set_transient($key, $count + 1, 5 * MINUTE_IN_SECONDS);
    return true;
}

function dl_ai_ask(WP_REST_Request $req) {
    $query = trim((string) $req['query']);
    if (mb_strlen($query) < 2 || mb_strlen($query) > 2000) {
        return new WP_Error('validation', 'Question must be 2–2000 characters.', ['status' => 422]);
    }
    $payload = array_filter([
        'query'           => $query,
        'conversation_id' => $req['conversation_id'] ?: null,
        'filters'         => $req['filters'] ?: null,
    ]);

    // Cache identical first-turn questions for 10 minutes (huge win for FAQ-style traffic).
    $cache_key = empty($payload['conversation_id']) ? 'dl_ai_' . md5(wp_json_encode($payload)) : null;
    if ($cache_key && ($hit = get_transient($cache_key))) {
        return rest_ensure_response($hit);
    }

    $res = wp_remote_post(DL_RAG_API_BASE . '/api/chat', [
        'timeout' => 90,
        'headers' => ['Content-Type' => 'application/json', 'X-API-Key' => DL_RAG_API_KEY],
        'body'    => wp_json_encode($payload),
    ]);
    if (is_wp_error($res)) {
        return new WP_Error('upstream', 'The assistant is unreachable right now.', ['status' => 502]);
    }
    $status = wp_remote_retrieve_response_code($res);
    $body   = json_decode(wp_remote_retrieve_body($res), true);
    if ($status !== 200) {
        // Pass through the API's error envelope + request_id (see §6) with a friendly detail.
        return new WP_Error($body['error'] ?? 'upstream_error',
            $body['detail'] ?? 'The assistant could not answer right now.',
            ['status' => $status, 'request_id' => $body['request_id'] ?? null]);
    }
    if ($cache_key && ($body['confidence'] ?? 0) > 0) {
        set_transient($cache_key, $body, 10 * MINUTE_IN_SECONDS);
    }
    return rest_ensure_response($body);
}

function dl_ai_feedback(WP_REST_Request $req) {
    $res = wp_remote_post(DL_RAG_API_BASE . '/api/feedback', [
        'timeout' => 15,
        'headers' => ['Content-Type' => 'application/json', 'X-API-Key' => DL_RAG_API_KEY],
        'body'    => wp_json_encode([
            'conversation_id' => (string) $req['conversation_id'],
            'message_id'      => (string) $req['message_id'],
            'rating'          => $req['rating'] === 'down' ? 'down' : 'up',
            'comment'         => mb_substr((string) ($req['comment'] ?? ''), 0, 2000) ?: null,
        ]),
    ]);
    return rest_ensure_response(['accepted' => !is_wp_error($res)]);
}
```

### 3.3 Shortcode + script

```php
add_shortcode('dl_ask', function ($atts) {
    $atts = shortcode_atts(['placeholder' => 'Ask about Indian education, policy, WES…'], $atts);
    wp_enqueue_script('dl-ai-widget', plugins_url('dl-ai-widget.js', __FILE__), [], '1.1', true);
    wp_enqueue_script('marked', 'https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js', [], '12.0.2', true);
    wp_localize_script('dl-ai-widget', 'DL_AI', [
        'endpoint' => esc_url_raw(rest_url('dl-ai/v1/ask')),
        'feedback' => esc_url_raw(rest_url('dl-ai/v1/feedback')),
        'nonce'    => wp_create_nonce('wp_rest'),
    ]);
    return '<div class="dl-ask" data-placeholder="' . esc_attr($atts['placeholder']) . '"></div>';
});
```

Drop `[dl_ask]` into any page/post (or `echo do_shortcode('[dl_ask]')` in a template).

### 3.4 Widget script (`dl-ai-widget.js`, vanilla, no framework)

```js
(function () {
  const root = document.querySelector('.dl-ask');
  if (!root) return;
  root.innerHTML = `
    <form class="dl-ask__form">
      <input name="q" required minlength="2" maxlength="2000" placeholder="${root.dataset.placeholder}">
      <button type="submit">Ask</button>
    </form>
    <div class="dl-ask__thread"></div>`;
  const form = root.querySelector('form');
  const thread = root.querySelector('.dl-ask__thread');
  let conversationId = null;   // keep it: follow-ups like "what about 2024?" need it

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = form.q.value.trim();
    if (!query) return;
    form.q.value = '';
    thread.insertAdjacentHTML('beforeend', `<div class="dl-msg dl-msg--user">${esc(query)}</div>`);
    const box = document.createElement('div');
    box.className = 'dl-msg dl-msg--bot dl-msg--loading';
    box.textContent = 'Searching the archive…';
    thread.append(box);

    try {
      const res = await fetch(DL_AI.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-WP-Nonce': DL_AI.nonce },
        body: JSON.stringify({ query, conversation_id: conversationId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || data.detail || 'Something went wrong.');
      conversationId = data.conversation_id;
      box.classList.remove('dl-msg--loading');
      box.innerHTML = render(data);
      wireFeedback(box, data);
    } catch (err) {
      box.classList.remove('dl-msg--loading');
      box.textContent = err.message;
    }
  });

  function render(d) {
    const byIndex = Object.fromEntries(d.sources.map(s => [s.index, s]));
    // [1][3] → superscript links to the matching source
    const html = marked.parse(d.answer).replace(/\[(\d+)\]/g, (m, n) =>
      byIndex[n] ? `<sup><a href="${byIndex[n].url}" target="_blank" rel="noopener">[${n}]</a></sup>` : m);
    const sources = d.sources.map(s => `
      <li>
        ${s.content_type === 'video' ? `<img src="https://img.youtube.com/vi/${ytId(s.url)}/mqdefault.jpg" alt="" width="120">` : ''}
        <a href="${s.url}" target="_blank" rel="noopener">[${s.index}] ${esc(s.title)}</a>
        <small>${s.date || ''} · ${s.content_type.replace('_', ' ')}${s.author ? ' · ' + esc(s.author) : ''}</small>
      </li>`).join('');
    const band = d.confidence_band;   // high | medium | low → colour a chip
    return `
      <div class="dl-answer">${html}</div>
      ${d.sources.length ? `<details open class="dl-sources"><summary>Sources (${d.sources.length})
        <span class="dl-chip dl-chip--${band}">${band} confidence</span></summary><ul>${sources}</ul></details>` : ''}
      <div class="dl-feedback">Was this helpful?
        <button data-rating="up" aria-label="Helpful">👍</button>
        <button data-rating="down" aria-label="Not helpful">👎</button></div>`;
  }

  function wireFeedback(box, d) {
    box.querySelectorAll('.dl-feedback button').forEach(b => b.addEventListener('click', () => {
      fetch(DL_AI.feedback, { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-WP-Nonce': DL_AI.nonce },
        body: JSON.stringify({ conversation_id: d.conversation_id, message_id: d.message_id, rating: b.dataset.rating }) });
      box.querySelector('.dl-feedback').textContent = 'Thanks for the feedback.';
    }));
  }
  const ytId = (u) => (u.match(/[?&]v=([\w-]{11})/) || [])[1] || '';
  const esc = (s) => s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
})();
```

### 3.5 Streaming (optional, for a "typing" effect)

`wp_remote_post` buffers the whole response, so the PHP proxy above delivers the
answer in one piece (typically 10–20 s; show a skeleton/loader). If you want tokens
to appear as they're generated, run a tiny Node/edge proxy that forwards
`stream: true` and pipes the body through untouched; the client then reads
Server-Sent Events:

| Event | `data` | Meaning |
| --- | --- | --- |
| `meta` | JSON `{conversation_id, message_id, query_type, retrieved_documents}` | once, before generation |
| `token` | raw text | append to the answer |
| `done` | JSON — the full response object of §4.1 | final answer + sources + confidence |
| `error` | JSON `{message}` | `retrieval_failed` / `generation_failed` |

```js
const res = await fetch('/ask-stream', { method: 'POST', headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({ query, stream: true }) });
const reader = res.body.getReader(), dec = new TextDecoder();
let buf = '', answer = '';
for (;;) {
  const { done, value } = await reader.read(); if (done) break;
  buf += dec.decode(value, { stream: true });
  const frames = buf.split('\n\n'); buf = frames.pop() ?? '';
  for (const f of frames) {
    const ev = /^event:\s*(.*)$/m.exec(f)?.[1], data = /^data:\s*([\s\S]*)$/m.exec(f)?.[1] ?? '';
    if (ev === 'token') { answer += data; paint(answer); }
    if (ev === 'done')  { finish(JSON.parse(data)); }
    if (ev === 'error') { fail(JSON.parse(data).message); }
  }
}
```

### 3.6 Instant indexing when editors publish

New and edited articles are picked up automatically once a day. To make an article
answerable within seconds of publishing, call the index endpoint from a publish hook:

```php
add_action('transition_post_status', function ($new, $old, $post) {
    if ($new !== 'publish' || $post->post_type !== 'post') return;
    wp_remote_post(DL_RAG_API_BASE . '/api/index', [
        'timeout'  => 5,
        'blocking' => false,                       // fire-and-forget
        'headers'  => ['Content-Type' => 'application/json', 'X-API-Key' => DL_RAG_API_KEY],
        'body'     => wp_json_encode(['urls' => [get_permalink($post)]]),
    ]);
}, 10, 3);
```

The API crawls that URL, chunks + embeds it, updates the knowledge graph, and
returns `202 {job_id}`; poll `GET /api/index/{job_id}` if you want to display
status. Re-publishing an edited post re-indexes it (content-hash de-duplicated).
Channel videos need no hook — new education-vertical uploads are detected daily.

---

## 4. API reference

### 4.1 `POST /api/chat`

Request fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `query` | string | ✅ | 2–2000 characters |
| `conversation_id` | string | — | omit on the first turn; send the returned id afterwards (last 20 turns are remembered for 7 days) |
| `stream` | boolean | — | `true` → SSE (§3.5). Default `false` |
| `filters` | object | — | narrows retrieval, see below |
| `top_k` | int 1–20 | — | number of sources to ground on (default 8) |

`filters`:

| Field | Type | Example |
| --- | --- | --- |
| `year_from`, `year_to` | int | `2020`, `2024` |
| `content_types` | string[] | `["interview", "video"]` |
| `authors` | string[] | `["Ravi Kumar"]` |
| `tags` | string[] | `["NEP"]` |

Content types you can filter on: `news`, `interview`, `video`, `policy`,
`magazine_issue`, `magazine_article`, `ranking`, `higher_education`, `school`,
`government_news`, `corporate`, `feature`, `opinion`, `special_story`,
`cover_story`, `other`.

Response `200`:

```jsonc
{
  "answer": "## Executive Summary\nNEP 2020 introduced … competency-based learning [1][3].",
  "sources": [
    { "index": 1, "title": "NEP Implementation in Karnataka",
      "url": "https://digitallearning.eletsonline.com/2022/06/…", "date": "2022-06-14",
      "content_type": "policy", "category": "Policy Matters", "issue": null, "author": "Ravi Kumar" },
    { "index": 5, "title": "Interview | Peter Mugambi at 35th Elets World Education Summit 2026 – Dubai",
      "url": "https://www.youtube.com/watch?v=…", "date": "2026-03-02",
      "content_type": "video", "category": "Videos", "issue": null, "author": "elets Insights" }
  ],
  "confidence": 0.94,
  "confidence_band": "high",           // high | medium | low
  "query_type": "comparison",           // see §4.6
  "retrieved_documents": 8,
  "conversation_id": "8f1c2e…",         // send back on the next turn
  "message_id": "a7b3…",                // needed for feedback
  "latency_ms": 1420,
  "token_usage": { "prompt_tokens": 3100, "completion_tokens": 640, "total_tokens": 3740 }
}
```

`sources[]` only lists documents the answer actually cites; indices can have gaps.

### 4.2 `POST /api/feedback`

```jsonc
{ "conversation_id": "8f1c2e…", "message_id": "a7b3…", "rating": "up", "comment": "optional, ≤2000 chars", "reason": "optional" }
```
`rating` ∈ `up | down` → `{ "accepted": true, "message": "Thanks — feedback recorded." }`

### 4.3 `GET /api/document/{id}?include_content=true`

Full source record: `title, url, subtitle, author, published_date, content_type,
category, tags, issue_name, issue_year, entities, keywords, word_count,
chunk_count`, and (with the flag) `content_markdown`. Use it for a "view source"
side panel. Document ids are stable hashes of the canonical URL.

### 4.4 `POST /api/index` · `GET /api/index/{job_id}`

`{"urls": ["https://digitallearning.eletsonline.com/2026/09/…/"]}` → `202
{ "job_id", "status": "pending", "accepted_urls": 1 }`. Job status:
`pending | running | completed | failed` with `pages_indexed`, `chunks_created`,
`error`. Intended for the publish hook (§3.6) — not for visitor traffic.

### 4.5 `GET /health` (no auth)

`{"status":"ok","version":"0.1.0","checks":{"postgres":"ok","redis":"ok","qdrant":"ok"}}` —
degraded dependencies show `"error"` while still returning 200, so check the fields.

### 4.6 `query_type` values

The API detects intent and shapes the answer — handy for UI hints:

`timeline` (year-by-year) · `comparison` (includes a table) · `trend` · `definition`
· `policy` · `institution` · `person` · `interview` · `magazine` · `event` ·
`ranking` · `recommendation` · `statistics` · `summarization` · `general`

Video-intent questions ("give me WES 2023 videos", "recording of the Dubai panel")
return a linked list of `content_type: "video"` sources; "what did speakers say at…"
questions draw on video transcripts alongside interview articles.

---

## 5. Rendering guide

- **`answer` is markdown**: headings (`##`), bold, bullet lists, and GFM tables
  (comparisons). Use `marked` / `react-markdown` with tables enabled and sanitise the
  HTML if your markdown library doesn't.
- **Citations** are `[n]` markers, sometimes stacked `[1][3]`. Map `n` to the source
  whose `index === n` and link to its `url` (regex in §3.4). Never strip them —
  grounded citations are the product.
- **Always show the sources list** under the answer. For `content_type: "video"`
  sources the `url` is a YouTube watch link — show the thumbnail
  (`https://img.youtube.com/vi/<id>/mqdefault.jpg`) and open in a new tab.
- **Confidence**: `confidence_band` → chip colour (high = green, medium = amber,
  low = grey). Don't hide low-confidence answers; label them.
- **No-evidence case**: `confidence: 0`, empty `sources`, and the exact text
  *"I could not find enough supporting evidence in the digitalLEARNING archive to
  answer this confidently."* Render as an informational state and suggest example
  prompts — it is not an error.
- **Dates** are ISO `YYYY-MM-DD` (may be `null` for undated pages).
- Keep `conversation_id` in component state for the whole session so follow-ups
  resolve ("what about 2024?", "any videos on that?").

---

## 6. Errors & limits

All errors share one envelope (also mirrored by the WordPress proxy above):

```json
{ "error": "validation_error", "detail": "Request payload failed validation.",
  "request_id": "406c46b4…", "extra": {} }
```

| Status | `error` | Cause / what to do |
| --- | --- | --- |
| 401 | `authentication_error` | missing or wrong `X-API-Key` (server config problem, not the visitor's) |
| 404 | `not_found` | unknown document / job id |
| 422 | `validation_error` | bad payload — e.g. `query` under 2 chars; `extra.errors` lists the fields |
| 429 | `rate_limited` | over 60 requests / minute for the key — honour `Retry-After` (seconds); apply your own per-visitor limit so this never trips |
| 503 | `generation_error` | the language model was unavailable (the API fails over between two providers, so this is rare); safe to retry once after a few seconds |
| 503 | `retrieval_error` / `dependency_unavailable` | search backend hiccup; retry once |
| 500 | `internal_error` | unexpected — report the `request_id` |

Every response carries `X-Request-ID`; you may also send your own to correlate logs.

**Timeouts**: allow **90 s** on the server-side call (answers usually land in
10–20 s; long comparison answers can take 40 s). **Payload**: ≤ 2000-character
questions. **Memory**: conversations expire after 7 days of inactivity.

---

## 7. UX recommendations

- Seed the box with prompts that show the range: *"How has NEP evolved since
  2020?"* · *"Compare CBSE and State Board reforms"* · *"What did speakers say at
  the World Education Summit 2026 in Dubai?"* · *"Give me WES 2023 video links"* ·
  *"Who has spoken at WES over the years?"*
- Place the widget on the homepage search area and on article pages ("Ask about
  this topic"), pre-filling the query with the article's primary tag.
- Show "Sources (N)" expanded by default — readers click through, which is
  traffic back to the archive.
- Wire 👍/👎 to `/api/feedback`; the editorial team can review it in the backend
  insights dashboard.
- Cache identical first-turn questions for ~10 minutes at the CMS (done in §3.2) —
  popular questions cost nothing after the first visitor.
- For a long answer, render progressively (streaming, §3.5) or show a skeleton
  with the message "Reading the archive…" — a blank 15-second wait feels broken.

---

## 8. Operational facts for the CMS team

| Topic | Fact |
| --- | --- |
| Index refresh | automatic daily run (~12:30 UTC / 18:00 IST): new + edited articles from the site, new education-vertical uploads from the `@eletsvideos` channel, transcript backfill |
| Instant refresh | publish hook → `POST /api/index` (§3.6) |
| Coverage today | 18,800+ documents, ~29,800 searchable passages, ~990 videos (transcripts for a growing subset) |
| Knowledge graph | 95k entities / 12.9k relations power "who / where / with whom" questions |
| Provider failover | answers come from OpenAI with automatic fallback to Anthropic — no CMS change needed if one provider is down |
| Status | `GET /health`; backend metrics at `/metrics` (Prometheus, internal) |
| Support | quote the `request_id` and the exact question when reporting an odd answer |
