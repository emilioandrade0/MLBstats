# STRIKECAST — Deploy guide (24/7 en el cloud)

## Arquitectura final

```
                    ┌─────────────────────────────────┐
                    │   GitHub Actions (gratis)       │
                    │   cron cada 2h Mar-Nov:         │
                    │   src.refresh → src.model.      │
                    │   value_picks → git push        │
                    └───────────────┬─────────────────┘
                                    │ push → auto-deploy
                                    ▼
  ┌─────────────────┐         ┌────────────────────┐
  │  Vercel (free)  │────────▶│   Railway ($5/mo)  │
  │  StatsMLB       │  /api   │   FastAPI Docker   │
  │  Next.js UI     │◀────────│   uvicorn :8000    │
  └─────────────────┘         └────────────────────┘
        ▲                              ▲
        │                              │
    tu browser                    parquets bakeados
    /móvil                       en la imagen Docker
```

**Costo total: ~$5/mes** (Railway Hobby). Tu laptop puede quedarse apagada.

**Los 40 GB de raw se quedan en tu laptop** como archivo. El cron de GitHub Actions descarga solo los raw del día (~5 MB) en el runner y los desecha; los parquets sí se commitean.

---

## 0 · Pre-flight (una sola vez, local)

Asegúrate de tener los archivos que Railway va a bakear en la imagen:

```powershell
cd C:\Users\andra\Desktop\STRIKECAST
python -m src.refresh --days-forward 7
python -m src.model.value_picks --summary
```

Esto deja fresh:
- `data/processed/odds_close.parquet`
- `data/processed/features_market.parquet`
- `data/processed/value_picks.parquet`
- `data/models/value_first.pkl`, `value_win.pkl`

---

## 1 · Backend → Railway

### 1a. Subir código a GitHub

```powershell
cd C:\Users\andra\Desktop\STRIKECAST
git init                          # si aún no está
git add .gitignore Dockerfile .dockerignore railway.json requirements.txt src/ .github/
# Parquets/modelos slim que embarcan en Docker:
git add data/processed/train.parquet `
        data/processed/games.parquet `
        data/processed/games_xref.parquet `
        data/processed/odds_close.parquet `
        data/processed/features_market.parquet `
        data/processed/features_pitcher_season.parquet `
        data/processed/value_picks.parquet
git add data/models/*.pkl
git commit -m "deploy: initial Railway backend"
git remote add origin https://github.com/<tu-usuario>/strikecast.git
git push -u origin main
```

### 1b. Crear servicio Railway

1. [railway.com](https://railway.com) → New Project → Deploy from GitHub repo
2. Selecciona el repo. Railway detecta `Dockerfile` + `railway.json` automáticamente.
3. Settings → Variables → agrega:

   | Variable | Valor |
   |---|---|
   | `STRIKECAST_DISABLE_REFRESH` | `1` |
   | `STRIKECAST_ONREQ_TTL_SEC` | `120` |
   | `STRIKECAST_CORS_ORIGINS` | `*` (temporal, se ajusta en 2c) |
   | `ODDS_API_KEY` | `f5be61575edbd4a77f398d40df2d8c83` |

4. Settings → Networking → **Generate Domain**. Anota el URL, ej: `strikecast-production.up.railway.app`.
5. Wait ~3 min para el build. Verifica:
   ```powershell
   curl https://<tu-railway>.up.railway.app/api/health
   ```
   Debe devolver `{"ok": true, ...}`.

### 1c. Smoke test del Value endpoint

```powershell
curl "https://<tu-railway>.up.railway.app/api/value_picks?start=2026-08-02&end=2026-08-02"
```

Debe devolver `{updated_at, picks: [...]}` con al menos algunos tier `VALOR-FUERTE`.

---

## 2 · Frontend → Vercel

El frontend real es **`StatsMLB/`** (Next.js). La carpeta `vercel/` es el static viejo (index.html) — se puede ignorar.

### 2a. Deploy

**Opción A — Vercel CLI (rápido):**
```powershell
npm i -g vercel
cd C:\Users\andra\Desktop\STRIKECAST\StatsMLB
vercel                # primera vez: linkea el proyecto
vercel --prod         # promueve a producción
```

**Opción B — Integración GitHub:**
1. vercel.com → Add New Project → Import repo
2. Root Directory: `StatsMLB`
3. Framework Preset: **Next.js** (auto-detectado)
4. Deploy

### 2b. Configurar la variable de backend

En Vercel → Project → Settings → Environment Variables:

| Variable | Valor |
|---|---|
| `STRIKECAST_BASE_URL` | `https://<tu-railway>.up.railway.app` |
| `TELEGRAM_BOT_TOKEN` | (el tuyo, si usas Telegram picks) |
| `TELEGRAM_CHAT_ID` | (el tuyo) |

Redeploy tras agregarlas para que las tome.

### 2c. Cerrar CORS

En Railway → Variables → cambia `STRIKECAST_CORS_ORIGINS`:
```
https://<tu-vercel>.vercel.app
```
Redeploy Railway.

---

## 3 · Automatización 24/7 (GitHub Actions)

Ya está configurado en `.github/workflows/refresh-odds.yml`. Cada 2 horas durante temporada MLB (Mar-Nov):

1. Descarga odds frescas de ESPN (`src.refresh --days-forward 7`)
2. Reentrenar Value Model (`src.model.value_picks`)
3. Auto-commit `odds_close.parquet` + `features_market.parquet` + `value_picks.parquet`
4. Push → Railway auto-rebuild ~3 min → Vercel proxy ve la data fresca

**Nada más que hacer.** Tu laptop puede quedarse apagada. La app queda viva 24/7.

Para triggear manualmente en cualquier momento:
GitHub → tu repo → Actions → refresh-odds → Run workflow.

---

## 4 · Dominio custom (opcional)

- Compra `strikecast.com` en Cloudflare (~$10/año)
- Vercel → Project → Settings → Domains → añade `strikecast.com`
- Pega los DNS records que te da Vercel en Cloudflare
- Opcional: `api.strikecast.com` → CNAME al Railway; actualiza `STRIKECAST_BASE_URL` en Vercel

---

## Cost summary

| Servicio | Tier | $/mes |
|---|---|---:|
| Vercel Hobby | Frontend Next.js | $0 |
| Railway Hobby | Backend Docker 512 MB RAM | ~$5 |
| GitHub Actions | 2000 min/mes gratis (usamos ~50 min/mes) | $0 |
| Dominio (opcional) | Cloudflare Registry | ~$1/mes amortizado |
| **Total** | | **$0-6/mo** |

---

## Alternativa 100% gratis (sin Railway)

Si quieres cero costos: **Fly.io** ofrece 3 VMs gratis de 256 MB RAM cada una — suficiente para STRIKECAST. Mismo `Dockerfile`.

```powershell
curl -L https://fly.io/install.sh | sh
fly launch --dockerfile Dockerfile
fly deploy
```

Reemplaza el URL de Railway en `STRIKECAST_BASE_URL` de Vercel.

---

## Troubleshooting

### El endpoint `/api/value_picks` devuelve 503 en Railway
Falta el parquet dentro de la imagen. Verifica que hiciste `git add data/processed/value_picks.parquet` antes del push, y que `.gitignore` NO lo excluye (línea `!data/processed/value_picks.parquet`).

### GitHub Actions no pushea cambios
Verifica que el repo tenga permisos `contents: write` en Actions (Settings → Actions → General → Workflow permissions → **Read and write**).

### Vercel no puede hablar con Railway
- Confirma que `STRIKECAST_BASE_URL` en Vercel apunta al dominio Railway con `https://`.
- Confirma que Railway `STRIKECAST_CORS_ORIGINS` incluye el dominio Vercel exacto.
- Tests: `curl https://<railway>/api/health` desde tu terminal (bypasses Vercel).

### El modelo Value da NEUTRO en todos los games
El parquet vino sin odds. Espera al siguiente ciclo de cron (~2h) o dispara manualmente el workflow en GitHub → Actions.
