# STRIKECAST — Deploy guide

Architecture:
- **Frontend** → Vercel (static, free, edge CDN)
- **Backend** → Railway (FastAPI in Docker, $5/mo after free trial)

Routing: Vercel rewrites `/api/*` → Railway. Browser only ever talks to the Vercel domain; CORS is not exercised in normal traffic. CORS middleware is still on in the backend so direct calls work (curl, dev, alt frontends).

---

## 0 · Pre-flight (one-time, local)

You need a recent `train.parquet` + `odds_close.parquet` baked into the image as a baseline. Refresh locally:

```powershell
python -m src.refresh
```

That writes today's ESPN data into `data/raw/odds/` and rebuilds `odds_close.parquet` + `features_market.parquet`. The image picks these up via the `COPY` lines in `Dockerfile`.

> ⚡ **Auto-refresh on visit** is enabled in production. When someone hits `/api/games?start=<today>`, the backend automatically fetches fresh ESPN odds (~1–2s on cold cache, throttled to once every 120s afterwards). The background loop is OFF (`STRIKECAST_DISABLE_REFRESH=1`) so the container stays idle when no one's browsing.
>
> **The only thing that still requires a redeploy:** new games whose features aren't in `train.parquet` (lineups, pitcher stats…). Those need a local `python -m src.features.build && python -m src.train` + push. In practice: rebuild and push once a day.

---

## 1 · Backend → Railway

### 1a. Push code to GitHub
```powershell
git init                      # if not already
git add Dockerfile .dockerignore railway.json requirements.txt src/ data/processed/*.parquet data/models/*.pkl
git commit -m "deploy: initial Railway backend"
git remote add origin https://github.com/<you>/strikecast.git
git push -u origin main
```

> Note: parquets and `.pkl` are normally `.gitignore`'d. Either force-add them (`git add -f ...`) or move them to a Railway volume later. For first deploy, force-add is simplest — total is ~8 MB.

### 1b. Create Railway service
1. Go to [railway.com](https://railway.com) → New Project → Deploy from GitHub repo
2. Pick this repo. Railway auto-detects `Dockerfile` + `railway.json`.
3. Add env vars (Settings → Variables):
   - `STRIKECAST_DISABLE_REFRESH=1` (already in Dockerfile, but pin it here too)
   - `STRIKECAST_CORS_ORIGINS=https://<your-vercel-domain>.vercel.app` *(set after step 2 gives you the domain — temporarily use `*`)*
4. Settings → Networking → **Generate Domain**. You'll get e.g. `strikecast-production.up.railway.app`.
5. Wait for build (~3 min). Check `/api/health` returns `{"ok": true, ...}`.

### 1c. Smoke test
```powershell
curl https://strikecast-production.up.railway.app/api/health
curl "https://strikecast-production.up.railway.app/api/games?start=2025-08-15&end=2025-08-15" | head -200
```

---

## 2 · Frontend → Vercel

### 2a. Point `vercel.json` at your Railway domain
Edit `vercel/vercel.json` line 5 — replace `REPLACE_WITH_RAILWAY_DOMAIN` with the host from step 1b.4 (no `https://`, just the host — actually keep the `https://`, see file).

### 2b. Deploy
Option A — Vercel CLI (fastest):
```powershell
npm i -g vercel
cd vercel
vercel              # first time: link / create project
vercel --prod       # promote to production
```

Option B — GitHub integration:
1. vercel.com → Add New Project → Import this repo
2. Root Directory: `vercel`
3. Framework Preset: **Other** (it's a static dir)
4. Deploy

You'll get `https://strikecast.vercel.app` (or whatever name).

### 2c. Lock down CORS
Go back to Railway → set `STRIKECAST_CORS_ORIGINS=https://strikecast.vercel.app` (your actual domain). Click **Redeploy**.

---

## 3 · Daily update workflow

Whenever you want fresh predictions for today's slate:

```powershell
python -m src.refresh                 # 30s — updates today's odds JSON + parquet
git add data/processed/odds_close.parquet data/processed/features_market.parquet
git commit -m "data: refresh $(Get-Date -Format yyyy-MM-dd)"
git push
```

Railway auto-rebuilds & redeploys (~3 min). Vercel doesn't need to be touched — it just proxies.

To automate: set up a GitHub Action with a daily cron that runs `python -m src.refresh` + commits. Worth doing once Railway is stable.

---

## 4 · Custom domain (optional)

- Buy `strikecast.com` (Cloudflare ~$10/yr, no markup)
- Vercel → Project → Settings → Domains → add `strikecast.com`
- Vercel gives you the DNS records → paste into Cloudflare
- Optionally point `api.strikecast.com` → Railway with a CNAME, then update `vercel.json` rewrites

---

## Cost summary

| service | tier | monthly |
|---|---|---|
| Vercel | Hobby | $0 |
| Railway | Hobby ($5 trial) | ~$5 (always-on container, 512 MB RAM) |
| Domain | optional | ~$1/mo amortized |
| **Total** | | **$0–6/mo** |

Free alternative: deploy backend to **Fly.io** (3 shared-cpu-1x VMs free, 256 MB each — enough for STRIKECAST). Same `Dockerfile` works. Replace step 1 with `fly launch && fly deploy`.
