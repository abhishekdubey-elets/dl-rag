"""GET / — a minimal, self-contained chat UI for the archive assistant.

Single inline HTML page (no external assets, no build step) that talks to
``POST /api/chat``. Meant for local demos and smoke-testing the full loop;
production frontends should be built separately against the same API.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>digitalLEARNING Research Assistant</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --text: #1a2233; --muted: #67718a;
    --accent: #0f62fe; --accent-soft: #e8f0fe; --border: #e2e6ee;
    --user: #0f62fe; --user-text: #ffffff; --ok: #1a7f37; --warn: #b58105; --low: #b3261e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1420; --panel: #161d2e; --text: #e8ecf5; --muted: #93a0bd;
      --accent: #6ea8ff; --accent-soft: #1d2a45; --border: #26314d;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
         background: var(--bg); color: var(--text); display: flex; flex-direction: column;
         height: 100vh; }
  header { padding: 14px 22px; background: var(--panel); border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 12px; }
  header .logo { width: 34px; height: 34px; border-radius: 8px; background: var(--accent);
                 color: #fff; display: grid; place-items: center; font-weight: 700; }
  header h1 { font-size: 16px; margin: 0; }
  header p { margin: 0; font-size: 12.5px; color: var(--muted); }
  header .status { margin-left: auto; font-size: 12px; color: var(--muted); }
  header .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                margin-right: 5px; background: var(--muted); }
  #chat { flex: 1; overflow-y: auto; padding: 24px 0; }
  .wrap { max-width: 860px; margin: 0 auto; padding: 0 20px; }
  .msg { margin-bottom: 18px; display: flex; }
  .msg.user { justify-content: flex-end; }
  .bubble { max-width: 78%; padding: 12px 16px; border-radius: 14px; white-space: pre-wrap;
            overflow-wrap: break-word; }
  .user .bubble { background: var(--user); color: var(--user-text); border-bottom-right-radius: 4px; }
  .bot .bubble { background: var(--panel); border: 1px solid var(--border);
                 border-bottom-left-radius: 4px; width: 100%; max-width: 100%; }
  .bot .bubble h1,.bot .bubble h2,.bot .bubble h3,.bot .bubble h4 { margin: 12px 0 6px; font-size: 15px; }
  .bot .bubble table { border-collapse: collapse; margin: 8px 0; max-width: 100%; display: block; overflow-x: auto; }
  .bot .bubble td,.bot .bubble th { border: 1px solid var(--border); padding: 5px 9px; font-size: 13.5px; }
  .meta { margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--border);
          font-size: 12px; color: var(--muted); display: flex; gap: 10px; flex-wrap: wrap; }
  .chip { background: var(--accent-soft); color: var(--accent); border-radius: 20px;
          padding: 2px 10px; font-weight: 600; }
  .conf-high { color: var(--ok); } .conf-medium { color: var(--warn); } .conf-low { color: var(--low); }
  .sources { margin-top: 8px; }
  .sources a { display: block; color: var(--accent); text-decoration: none; font-size: 13px;
               margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sources a:hover { text-decoration: underline; }
  .hint { text-align: center; color: var(--muted); margin: 40px auto; max-width: 620px; }
  .hint .ex { display: inline-block; margin: 4px; padding: 7px 13px; border: 1px solid var(--border);
              border-radius: 20px; background: var(--panel); cursor: pointer; font-size: 13px; }
  .hint .ex:hover { border-color: var(--accent); color: var(--accent); }
  footer { padding: 14px 0 20px; background: linear-gradient(transparent, var(--bg) 40%); }
  form { display: flex; gap: 10px; }
  input[type=text] { flex: 1; padding: 13px 16px; border-radius: 12px; border: 1px solid var(--border);
          background: var(--panel); color: var(--text); font-size: 15px; outline: none; }
  input[type=text]:focus { border-color: var(--accent); }
  button { padding: 0 22px; border: 0; border-radius: 12px; background: var(--accent);
           color: #fff; font-weight: 600; font-size: 15px; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .typing { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<header>
  <div class="logo">dL</div>
  <div>
    <h1>digitalLEARNING Research Assistant</h1>
    <p>20+ years of education coverage · grounded, cited answers</p>
  </div>
  <div class="status" id="status"><span class="dot"></span>checking…</div>
</header>

<div id="chat"><div class="wrap" id="stream">
  <div class="hint" id="hint">
    <p><strong>Ask the archive.</strong> Answers are grounded in digitalLEARNING articles and cite their sources.</p>
    <span class="ex">How has NEP evolved since 2020?</span>
    <span class="ex">What is SWAYAM?</span>
    <span class="ex">Compare CBSE and State Board reforms</span>
    <span class="ex">Show interviews featuring AI in education</span>
  </div>
</div></div>

<footer><div class="wrap">
  <form id="form">
    <input type="text" id="q" placeholder="Ask about Indian education, policy, edtech…" autocomplete="off" autofocus>
    <button id="send" type="submit">Ask</button>
  </form>
</div></footer>

<script>
const stream = document.getElementById('stream');
const form = document.getElementById('form');
const q = document.getElementById('q');
const send = document.getElementById('send');
const hint = document.getElementById('hint');
let conversationId = null;

// --- health badge -----------------------------------------------------------
fetch('/health').then(r => r.json()).then(h => {
  const el = document.getElementById('status');
  const up = Object.values(h.checks).filter(v => v === 'ok').length;
  const total = Object.keys(h.checks).length;
  const color = h.status === 'ok' ? 'var(--ok)' : (up > 0 ? 'var(--warn)' : 'var(--low)');
  el.innerHTML = `<span class="dot" style="background:${color}"></span>${h.status} · ${up}/${total} services`;
  el.title = JSON.stringify(h.checks);
}).catch(() => {});

// --- tiny markdown renderer (headings, bold, tables, lists) ------------------
function esc(s) { return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function md(src) {
  const lines = esc(src).split('\\n');
  let out = [], table = [];
  const flush = () => {
    if (!table.length) return;
    const rows = table.filter(r => !/^\\s*\\|?[\\s:|-]+\\|?\\s*$/.test(r));
    out.push('<table>' + rows.map(r =>
      '<tr>' + r.replace(/^\\||\\|$/g, '').split('|').map(c => '<td>' + c.trim() + '</td>').join('') + '</tr>'
    ).join('') + '</table>');
    table = [];
  };
  for (const line of lines) {
    if (line.includes('|') && line.trim().startsWith('|')) { table.push(line); continue; }
    flush();
    let l = line;
    if (/^#{1,6}\\s/.test(l)) { out.push('<h4>' + l.replace(/^#{1,6}\\s*/, '') + '</h4>'); continue; }
    l = l.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    l = l.replace(/^[-*]\\s+/, '• ');
    out.push(l);
  }
  flush();
  return out.join('\\n');
}

function bubble(cls, html) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = '<div class="bubble">' + html + '</div>';
  stream.appendChild(div);
  div.scrollIntoView({behavior: 'smooth', block: 'end'});
  return div.querySelector('.bubble');
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text) return;
  if (hint) hint.remove();
  q.value = ''; send.disabled = true;
  bubble('user', esc(text));
  const b = bubble('bot', '<span class="typing">Searching the archive…</span>');
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: text, conversation_id: conversationId, stream: false})
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      b.innerHTML = '<span style="color:var(--low)">Error: ' + esc(err.detail || res.statusText) + '</span>';
      return;
    }
    const data = await res.json();
    conversationId = data.conversation_id;
    let html = md(data.answer);
    html += '<div class="meta">'
         + '<span class="chip">' + esc(data.query_type) + '</span>'
         + '<span class="conf-' + data.confidence_band + '">confidence ' + (data.confidence * 100).toFixed(0) + '% (' + data.confidence_band + ')</span>'
         + '<span>' + data.retrieved_documents + ' docs</span>'
         + '<span>' + data.latency_ms + ' ms</span>'
         + '</div>';
    if (data.sources && data.sources.length) {
      html += '<div class="sources"><strong style="font-size:12px;color:var(--muted)">SOURCES</strong>'
           + data.sources.map(s =>
              '<a href="' + s.url + '" target="_blank" rel="noopener">[' + s.index + '] '
              + esc(s.title) + (s.date ? ' (' + s.date + ')' : '') + '</a>').join('')
           + '</div>';
    }
    b.innerHTML = html;
  } catch (err) {
    b.innerHTML = '<span style="color:var(--low)">Network error: ' + esc(String(err)) + '</span>';
  } finally {
    send.disabled = false; q.focus();
    b.scrollIntoView({behavior: 'smooth', block: 'end'});
  }
});

document.querySelectorAll('.ex').forEach(el =>
  el.addEventListener('click', () => { q.value = el.textContent; form.requestSubmit(); }));
</script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home() -> HTMLResponse:
    return HTMLResponse(content=_PAGE)
