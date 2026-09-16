"""Capture ESPN per-provider moneylines with the current UTC timestamp.

Each run of ``update_data.bat`` refreshes ``data/raw/odds/{year}/*.json``
with fresh ESPN scoreboard data — the ``moneyLine`` field on each item is
what ESPN reports at the moment we fetched.  This script reads those files
and appends every provider's price to
``data/processed/espn_odds_timeseries.parquet`` tagged with the current UTC
timestamp, so we build our own hourly line-movement history without paying
The Odds API.

Runs idempotently: same event+provider+timestamp doesn't get duplicated.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ODDS_RAW = ROOT / "data" / "raw" / "odds"
OUT = ROOT / "data" / "processed" / "espn_odds_timeseries.parquet"

# Ventana activa: solo capturamos eventos entre [today-2d, today+14d].
# Fuera de esa ventana los momios están congelados (post-game) o aún no salen.
WINDOW_BACK_DAYS = 7
WINDOW_FORWARD_DAYS = 14


def collect(year_dir: Path, snapshot_ts: str) -> list[dict]:
    rows: list[dict] = []
    for jf in year_dir.glob("*.json"):
        event_id = jf.stem
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data.get("items") or []:
            provider = item.get("provider") or {}
            home = item.get("homeTeamOdds") or {}
            away = item.get("awayTeamOdds") or {}
            home_ml = home.get("moneyLine")
            away_ml = away.get("moneyLine")
            if home_ml is None or away_ml is None:
                continue
            try:
                home_ml_f = float(home_ml)
                away_ml_f = float(away_ml)
            except (TypeError, ValueError):
                continue
            rows.append({
                "snapshot_ts": snapshot_ts,
                "espn_event_id": str(event_id),
                "provider_id": str(provider.get("id")) if provider.get("id") is not None else None,
                "provider_name": provider.get("name"),
                "home_ml": home_ml_f,
                "away_ml": away_ml_f,
            })
    return rows


def main() -> None:
    snapshot_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    today = datetime.now(timezone.utc).date()
    window_start = today - timedelta(days=WINDOW_BACK_DAYS)
    window_end = today + timedelta(days=WINDOW_FORWARD_DAYS)

    # Cross-referenciamos event_id → game_date desde games.parquet para filtrar.
    try:
        xref = pd.read_parquet(
            ROOT / "data" / "processed" / "games_xref.parquet",
            columns=["espn_event_id", "game_pk", "_merge"],
        )
        xref = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
        xref["espn_event_id"] = xref["espn_event_id"].astype(str)
        games = pd.read_parquet(
            ROOT / "data" / "processed" / "games.parquet",
            columns=["game_pk", "game_date"],
        )
        games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
        merged = xref.merge(games, on="game_pk", how="inner")
        active_event_ids = set(
            merged.loc[
                (merged["game_date"] >= window_start) & (merged["game_date"] <= window_end),
                "espn_event_id",
            ].tolist()
        )
    except Exception as exc:
        print(f"[snapshot_espn_odds] no pude cargar xref/games: {exc}. Capturo todo.")
        active_event_ids = None

    rows: list[dict] = []
    for year_dir in sorted(ODDS_RAW.glob("*")):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year_rows = collect(year_dir, snapshot_ts)
        if active_event_ids is not None:
            year_rows = [r for r in year_rows if r["espn_event_id"] in active_event_ids]
        rows.extend(year_rows)

    if not rows:
        print("[snapshot_espn_odds] no rows extracted from ESPN raw JSONs")
        return

    new_df = pd.DataFrame(rows)
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["snapshot_ts", "espn_event_id", "provider_name"],
            keep="last",
        )
    else:
        combined = new_df

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT, index=False)
    print(
        f"[snapshot_espn_odds] +{len(new_df):,} rows ({snapshot_ts}); "
        f"total {len(combined):,} rows across {combined['snapshot_ts'].nunique()} snapshots"
    )


if __name__ == "__main__":
    main()
