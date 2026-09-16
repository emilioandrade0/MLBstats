"""Live ESPN moneyline snapshot vía el endpoint /summary (pickcenter).

El endpoint `sports.core.api.espn.com/events/{id}/competitions/{id}/odds`
que usa `src.ingest` está devolviendo count:0 para juegos en el pregame
window. Pero el `site.web.api.espn.com/apis/site/v2/sports/baseball/mlb/
summary?event={id}` sí trae `pickcenter` con moneyline actualizado.

Este script:
  1. Query scoreboard para today ± window
  2. Para cada event.id, query summary y extrae pickcenter
  3. Escribe a data/processed/espn_odds_timeseries.parquet con hora UTC del capture

Es totalmente auto-contenido — no necesita que `src.refresh` haya corrido.
"""
from __future__ import annotations

import json
import ssl
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "espn_odds_timeseries.parquet"

WINDOW_BACK_DAYS = 1
WINDOW_FORWARD_DAYS = 2
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def http_get_json(url: str, timeout: int = 20) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        print(f"  [warn] GET {url[:80]}...: {exc}")
        return None


def scoreboard_event_ids(date_str: str) -> list[str]:
    url = f"https://site.web.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date_str}"
    data = http_get_json(url)
    if not data:
        return []
    return [str(ev["id"]) for ev in data.get("events") or [] if ev.get("id")]


def event_pickcenter(event_id: str) -> list[dict]:
    url = f"https://site.web.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={event_id}"
    data = http_get_json(url)
    if not data:
        return []
    return data.get("pickcenter") or []


def main() -> None:
    snapshot_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    today = datetime.now(timezone.utc).date()
    dates = [
        (today + timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(-WINDOW_BACK_DAYS, WINDOW_FORWARD_DAYS + 1)
    ]

    event_ids: list[str] = []
    for d in dates:
        ids = scoreboard_event_ids(d)
        print(f"[scoreboard {d}] {len(ids)} eventos")
        event_ids.extend(ids)

    rows: list[dict] = []
    for i, event_id in enumerate(event_ids, 1):
        pc = event_pickcenter(event_id)
        if not pc:
            continue
        for entry in pc:
            provider = entry.get("provider") or {}
            home = entry.get("homeTeamOdds") or {}
            away = entry.get("awayTeamOdds") or {}
            home_ml = home.get("moneyLine")
            away_ml = away.get("moneyLine")
            if home_ml is None or away_ml is None:
                continue
            try:
                rows.append({
                    "snapshot_ts": snapshot_ts,
                    "espn_event_id": str(event_id),
                    "provider_id": str(provider.get("id")) if provider.get("id") is not None else None,
                    "provider_name": provider.get("name"),
                    "home_ml": float(home_ml),
                    "away_ml": float(away_ml),
                })
            except (TypeError, ValueError):
                continue
        if i % 10 == 0:
            print(f"  procesados {i}/{len(event_ids)} eventos, {len(rows)} filas hasta ahora")

    if not rows:
        print(f"[snapshot_espn_live] 0 filas capturadas de {len(event_ids)} eventos")
        return

    new_df = pd.DataFrame(rows)
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["snapshot_ts", "espn_event_id", "provider_name"], keep="last"
        )
    else:
        combined = new_df

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT, index=False)
    print(
        f"[snapshot_espn_live] +{len(new_df):,} filas ({snapshot_ts}); "
        f"total {len(combined):,} filas en {combined['snapshot_ts'].nunique()} snapshots"
    )


if __name__ == "__main__":
    main()
