# Deploying to a Hostinger VPS

End-to-end runbook: fresh Hostinger VPS → live HTTPS chatbot with your already-built
index. Commands marked **[local]** run on your Windows machine (PowerShell);
everything else runs on the VPS over SSH.

---

## 0. Sizing & purchase

In hPanel → **VPS**, pick a plan:

| Plan | Specs | Verdict |
| --- | --- | --- |
| KVM 2 | 2 vCPU / 8 GB RAM / 100 GB NVMe | **Minimum.** Works; reranker+LLM answers ~20–30 s warm. |
| KVM 4 | 4 vCPU / 16 GB RAM / 200 GB NVMe | **Recommended.** Headroom for reranking, ingest jobs, and growth. |

RAM budget at runtime: API process (torch + embedder + reranker) ≈ 2.5–3 GB,
Postgres ≈ 0.5–1 GB, Qdrant ≈ 0.5 GB, Redis ≈ 0.2 GB, OS ≈ 1 GB.

During setup choose **Ubuntu 24.04 LTS** (plain, not a template) and add your SSH
public key (hPanel → VPS → Settings → SSH keys). On Windows, create one if needed:

```powershell
# [local]
ssh-keygen -t ed25519            # accept defaults; public key at ~\.ssh\id_ed25519.pub
Get-Content ~\.ssh\id_ed25519.pub  # paste this into hPanel
```

---

## 1. First login & base hardening

```bash
# [local]  (IP shown in hPanel)
ssh root@YOUR_VPS_IP
```

```bash
# --- on the VPS ---
apt update && apt upgrade -y

# Non-root user
adduser deploy                       # choose a strong password
usermod -aG sudo deploy
rsync -a ~/.ssh /home/deploy/ && chown -R deploy:deploy /home/deploy/.ssh

# Firewall: SSH + web only
apt install -y ufw
ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443/tcp
ufw --force enable

# Disable root + password SSH logins
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/; s/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh

# Optional but recommended
apt install -y fail2ban
```

Reconnect as `deploy` from here on: `ssh deploy@YOUR_VPS_IP`.

---

## 2. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker                        # or log out/in
docker --version && docker compose version
```

---

## 3. Get the code onto the VPS

**Option A — private Git repo (recommended; makes updates trivial):**

```powershell
# [local] one-time: publish the project
cd C:\Users\Abhishek\Desktop\dl
git init -b main
git add -A && git commit -m "digitalLEARNING RAG"
gh repo create dl-rag --private --source . --push     # or create on github.com and push
```

```bash
# VPS
git clone https://github.com/YOURUSER/dl-rag.git ~/dl && cd ~/dl
```

**Option B — direct copy (no Git):**

```powershell
# [local]
scp -r C:\Users\Abhishek\Desktop\dl deploy@YOUR_VPS_IP:~/dl   # excludes nothing — slower
```

> `.env` is git-ignored — it must be created on the VPS either way (next step).

---

## 4. Production configuration

```bash
cd ~/dl
cp .env.example .env
nano .env
```

Set at minimum:

```ini
ENVIRONMENT=production
DEBUG=false
LOG_JSON=true

REQUIRE_AUTH=true
API_KEYS=<generate: openssl rand -hex 24>       # comma-separate several if needed
RATE_LIMIT_REQUESTS=60

LLM_API_KEY=<your OpenAI key — use a FRESH one, rotate the old>
LLM_MODEL=gpt-4o-mini

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5           # must match the index you migrate!
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_CANDIDATES=40                          # drop to 20 on KVM 2 if answers feel slow
```

Then edit the domain into the proxy config:

```bash
nano deploy/Caddyfile        # replace ask.example.com with your real subdomain
```

**DNS**: in your DNS panel add an **A record** — e.g. `ask.digitallearning.in → YOUR_VPS_IP`.
Wait for it to resolve (`dig +short ask.digitallearning.in`) before first boot, so
Caddy can obtain the TLS certificate.

---

## 5. First boot

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
# First build is slow (torch download). Watch:
docker compose logs -f api
```

When up: `https://ask.yourdomain.com` serves the chat UI (health badge will show
3/3), `/docs` the API docs. The first chat query downloads the embedding/reranker
models into the persistent `models_cache` volume (one-time, ~1 min).

Because `REQUIRE_AUTH=true`, API calls need the header `X-API-Key: <your key>`.

**API-only deployment** (no demo chat page): set in `.env`

```ini
SERVE_UI=false          # "/" is gone; the API and /docs remain
REQUIRE_AUTH=true       # every /api/* call needs X-API-Key
CORS_ORIGINS=https://digitallearning.eletsonline.com   # only if a browser frontend will call the API directly
```

If you keep the demo UI instead: it calls the API without a key, so either leave
`REQUIRE_AUTH=false` and gate the site with the proxy's basic-auth, or extend the
UI to send the key.

---

## 6. Load the index — migrate (fast) or re-ingest (clean)

### Option A — migrate your local data (recommended, ~minutes)

```powershell
# [local] dump Postgres (documents, chunks, KG, logs)
docker exec dl-postgres-1 pg_dump -U dl -d dl_rag -Fc -f /tmp/dl_rag.dump
docker cp dl-postgres-1:/tmp/dl_rag.dump .\dl_rag.dump

# [local] snapshot Qdrant vectors
curl -X POST http://localhost:6333/collections/dl_chunks/snapshots
# note the snapshot name it returns, then download:
curl -o dl_chunks.snapshot "http://localhost:6333/collections/dl_chunks/snapshots/<SNAPSHOT_NAME>"

# [local] ship both to the VPS
scp .\dl_rag.dump .\dl_chunks.snapshot deploy@YOUR_VPS_IP:~/dl/
```

```bash
# VPS — restore Postgres
docker compose cp dl_rag.dump postgres:/tmp/
docker compose exec postgres pg_restore -U dl -d dl_rag --clean --if-exists /tmp/dl_rag.dump

# VPS — restore Qdrant snapshot
docker compose cp dl_chunks.snapshot qdrant:/tmp/
docker compose exec qdrant sh -c 'curl -s -X POST "http://localhost:6333/collections/dl_chunks/snapshots/upload?priority=snapshot" -H "Content-Type: multipart/form-data" -F "snapshot=@/tmp/dl_chunks.snapshot"'

# sanity
curl -s https://ask.yourdomain.com/api/admin/stats -H "X-API-Key: $KEY" | head
```

### Option B — re-ingest on the server (no file shipping; ~3–4 h total)

```bash
docker compose exec api dl-ingest --full                       # articles (~2.5 h)
docker compose exec api dl-ingest-youtube --match "world education summit|\bwes\b" --max-videos 8000
docker compose exec api dl-import-supabase                     # needs SUPABASE_DB_* in .env
docker compose exec api dl-kg-extract --rebuild                # knowledge graph (~30 min)
```

---

## 7. Verify

```bash
KEY=<your api key>
curl -s https://ask.yourdomain.com/health
curl -s -X POST https://ask.yourdomain.com/api/chat \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"query":"When is the next WES event?"}' | head -c 600
```

Open `https://ask.yourdomain.com` in a browser and ask a question end-to-end.

---

## 8. Operations

**Backups (cron on the VPS):**

```bash
crontab -e     # add:
0 3 * * * cd ~/dl && docker compose exec -T postgres pg_dump -U dl -d dl_rag -Fc > ~/backups/dl_rag_$(date +\%F).dump && find ~/backups -mtime +14 -delete
```

**Fresh content** (new articles/videos appear on the site continuously):

```bash
0 4 * * 0 cd ~/dl && docker compose exec -T api dl-ingest --since $(date -d '8 days ago' +\%F)
30 4 * * 0 cd ~/dl && docker compose exec -T api dl-ingest-youtube --skip-existing --max-videos 300
```

**Updates** (code changes):

```bash
cd ~/dl && git pull && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

**Monitoring**: `/metrics` is Prometheus-format; point a Grafana Cloud free agent or
a self-hosted Uptime-Kuma container at `/health`.

**Logs**: `docker compose logs -f api` (JSON in prod; ship to Loki/CloudWatch later
if needed).

---

## 9. Post-deploy checklist

- [ ] `REQUIRE_AUTH=true` (or Caddy basic_auth) confirmed — `/api/chat` rejects keyless calls
- [ ] Old OpenAI key rotated; new one only in the VPS `.env`
- [ ] Supabase DB password rotated (it transited chat); only needed on the VPS if you re-run imports
- [ ] Firewall: only 22/80/443 open (`sudo ufw status`)
- [ ] Backups cron installed and a restore tested once
- [ ] `docker compose ps` — all five services healthy after a reboot (`sudo reboot` test)

## Performance tuning on small VPSes

- `RETRIEVAL_CANDIDATES=20` halves reranker CPU time with modest quality cost.
- `stream: true` in the chat request makes answers feel immediate (first tokens < 2 s).
- If latency still hurts: move reranking off (`NoopReranker` via a config toggle is a
  small code change) or upgrade the plan — the cross-encoder is the CPU hog.
