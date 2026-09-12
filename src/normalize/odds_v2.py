"""Re-parse ESPN odds using explicit close.moneyLine + per-provider breakout.

Output: data/processed/odds_close.parquet
Schema:
  espn_event_id, season,
  provider_id, provider_name,
  home_ml_open, home_ml_close, home_ml_current,
  away_ml_open, away_ml_close, away_ml_current,
  spread_open, spread_close,
  total_open, total_close,
  moneyline_winner (post-game outcome flag — NOT to be used as feature),
"""
from __future__ import annotations

from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from .paths import RAW, PROCESSED, SEASONS


def _ml_from_dict(d: dict | None) -> float | None:
    """Extract american moneyline from the structured {moneyLine: {american: '-130', value: 1.769}} dict."""
    if not isinstance(d, dict):
        return None
    ml = d.get("moneyLine") or {}
    if isinstance(ml, (int, float)):
        return float(ml)
    if isinstance(ml, dict):
        # 'american' is the canonical str; some games are just numeric value.
        a = ml.get("american")
        if isinstance(a, str):
            try:
                return float(a.replace("+", ""))
            except ValueError:
                return None
        if isinstance(a, (int, float)):
            return float(a)
    return None


def _spread(d: dict | None) -> float | None:
    if not isinstance(d, dict):
        return None
    s = d.get("spread") or d.get("pointSpread") or {}
    if isinstance(s, dict):
        a = s.get("american") or s.get("alternateDisplayValue")
        if isinstance(a, str):
            try:
                return float(a.replace("+", ""))
            except ValueError:
                return None
    return None


def _total(d: dict | None) -> float | None:
    if not isinstance(d, dict):
        return None
    t = d.get("total") or {}
    if isinstance(t, dict):
        a = t.get("alternateDisplayValue") or t.get("american")
        if isinstance(a, str):
            try:
                v = float(a.replace("+", ""))
            except ValueError:
                return None
            # Sanity: MLB totals live in ~5-15 runs. Some providers (ESPN BET, older
            # payloads) put the OVER/UNDER *price* here instead of the line.
            # Reject anything outside the plausible range.
            if 3.0 <= v <= 20.0:
                return v
    return None


def _total_from_item(item: dict, side: str) -> float | None:
    """Preferred close/open total: use item-level overUnder / initialOverUnder
    (always a numeric line, correct across providers). Fall back to _total()
    on the nested open/close dict."""
    if side == "close":
        v = item.get("overUnder")
    else:  # open
        v = item.get("initialOverUnder")
    if isinstance(v, (int, float)) and 3.0 <= float(v) <= 20.0:
        return float(v)
    return _total(item.get(side))


def _provider_rows(event_id: str, season: int, payload: dict) -> list[dict]:
    out: list[dict] = []
    for item in payload.get("items") or []:
        prov = item.get("provider") or {}
        home = item.get("homeTeamOdds") or {}
        away = item.get("awayTeamOdds") or {}
        out.append({
            "espn_event_id": event_id,
            "season": season,
            "provider_id": prov.get("id"),
            "provider_name": prov.get("name"),
            "home_ml_top": (home.get("moneyLine") if isinstance(home.get("moneyLine"), (int, float)) else None),
            "away_ml_top": (away.get("moneyLine") if isinstance(away.get("moneyLine"), (int, float)) else None),
            "home_ml_open": _ml_from_dict(home.get("open")),
            "home_ml_close": _ml_from_dict(home.get("close")),
            "home_ml_current": _ml_from_dict(home.get("current")),
            "away_ml_open": _ml_from_dict(away.get("open")),
            "away_ml_close": _ml_from_dict(away.get("close")),
            "away_ml_current": _ml_from_dict(away.get("current")),
            "spread_close": _spread(home.get("close")),
            "spread_open": _spread(home.get("open")),
            "total_open": _total_from_item(item, "open"),
            "total_close": _total_from_item(item, "close"),
            "moneyline_winner_home": item.get("moneylineWinner"),  # post-game; keep for sanity but never feature
        })
    return out


def build(merge: bool = False, quiet: bool = False) -> Path:
    """Rebuild odds_close.parquet from RAW/odds/*.json.

    merge=True: keep existing rows for events not present in the current
    RAW dir, replacing only those that ARE present. Required in production
    (Docker image ships odds_close.parquet but NOT historical RAW JSONs,
    so a full overwrite would wipe history).
    """
    rows: list[dict] = []
    for season in SEASONS:
        d = RAW / "odds" / str(season)
        files = sorted(d.glob("*.json"))
        iterator = files if quiet else tqdm(files, desc=f"odds_v2 {season}")
        for p in iterator:
            try:
                payload = orjson.loads(p.read_bytes())
            except Exception:
                continue
            rows.extend(_provider_rows(p.stem, season, payload))
    new_df = pd.DataFrame(rows)
    out = PROCESSED / "odds_close.parquet"

    if merge and out.exists() and not new_df.empty:
        existing = pd.read_parquet(out)
        refreshed_ids = set(new_df["espn_event_id"].dropna().astype(str))
        keep_mask = ~existing["espn_event_id"].astype(str).isin(refreshed_ids)
        combined = pd.concat([existing[keep_mask], new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(out, index=False)
    if not quiet:
        print(f"wrote {out} ({len(combined):,} rows; {len(new_df):,} refreshed)")
        if not new_df.empty:
            print("Provider counts (refreshed):")
            print(new_df["provider_name"].value_counts().head(10).to_string())
    return out


if __name__ == "__main__":
    build()
