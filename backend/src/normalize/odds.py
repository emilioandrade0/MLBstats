"""Build odds.parquet — one row per (espn_event_id, provider) with current odds snapshot.

The ESPN core odds endpoint returns items[*] = { provider, details, overUnder,
spread, awayTeamOdds: { moneyLine, ... }, homeTeamOdds: { moneyLine, ... }, ... }.
"""
from __future__ import annotations

from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from .paths import RAW, PROCESSED, SEASONS


def _item_rows(event_id: str, season: int, payload: dict) -> list[dict]:
    out: list[dict] = []
    for item in payload.get("items") or []:
        provider = (item.get("provider") or {})
        away = item.get("awayTeamOdds") or {}
        home = item.get("homeTeamOdds") or {}
        out.append({
            "espn_event_id": event_id,
            "season": season,
            "provider_id": provider.get("id"),
            "provider_name": provider.get("name"),
            "details": item.get("details"),
            "over_under": item.get("overUnder"),
            "spread": item.get("spread"),
            "over_odds": item.get("overOdds"),
            "under_odds": item.get("underOdds"),
            "away_moneyline": (away.get("moneyLine") if isinstance(away.get("moneyLine"), (int, float)) else None),
            "home_moneyline": (home.get("moneyLine") if isinstance(home.get("moneyLine"), (int, float)) else None),
            "away_spread_odds": away.get("spreadOdds"),
            "home_spread_odds": home.get("spreadOdds"),
            "away_favorite": away.get("favorite"),
            "home_favorite": home.get("favorite"),
            "away_underdog": away.get("underdog"),
            "home_underdog": home.get("underdog"),
        })
    return out


def build() -> Path:
    rows: list[dict] = []
    for season in SEASONS:
        d = RAW / "odds" / str(season)
        files = sorted(d.glob("*.json"))
        for p in tqdm(files, desc=f"odds {season}"):
            try:
                payload = orjson.loads(p.read_bytes())
            except Exception:
                continue
            rows.extend(_item_rows(p.stem, season, payload))
    df = pd.DataFrame(rows)
    out = PROCESSED / "odds.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")
    return out


if __name__ == "__main__":
    build()
