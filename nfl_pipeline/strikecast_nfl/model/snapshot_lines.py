"""Capture a snapshot of current betting lines for every upcoming game.

Runs on-demand (or from iniciar_app.bat). Appends one row per (game_id,
source) with a timestamp, so over time you accumulate a time-series of
line movement per game.

Sources captured:
  - nflverse   : lines from data/raw/nfl_data_py/schedules_{season}.parquet
                 (nflreadpy — updated by community daily)
  - pinnacle   : from data/processed/odds_api_latest.parquet if present
  - espn       : from data/raw/espn/odds/{season}/{event_id}.json if present

Output (appended):
  data/processed/line_snapshots.parquet

Idempotent per minute — if the same (game_id, source, lines) tuple was
captured within the last 5 minutes, we skip to avoid noise on rapid restarts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import orjson
import pandas as pd

from ..paths import PROCESSED, RAW

OUT = PROCESSED / "line_snapshots.parquet"


def _from_nflverse() -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    upcoming = games[games["status"].isin(["upcoming", "live"])]
    if upcoming.empty:
        return pd.DataFrame()
    keep = ["game_id", "season", "week", "kickoff_utc", "home_team", "away_team",
            "spread_line", "total_line", "home_moneyline", "away_moneyline"]
    df = upcoming[[c for c in keep if c in upcoming.columns]].copy()
    df["source"] = "nflverse"
    return df


def _from_pinnacle() -> pd.DataFrame:
    p = PROCESSED / "odds_api_latest.parquet"
    if not p.exists():
        return pd.DataFrame()
    odds = pd.read_parquet(p)
    pin = odds[odds["bookmaker"] == "pinnacle"].copy()
    if pin.empty:
        return pd.DataFrame()
    games = pd.read_parquet(PROCESSED / "games.parquet",
                            columns=["game_id", "home_team", "away_team", "kickoff_utc",
                                     "season", "week"])
    # Match by (home_team, away_team) + closest kickoff time
    pin["commence_ts"] = pd.to_datetime(pin["commence_time"], utc=True, errors="coerce")
    games["kickoff_ts"] = pd.to_datetime(games["kickoff_utc"], utc=True, errors="coerce")
    merged = pin.merge(games, on=["home_team", "away_team"], how="left")
    merged["dt"] = (merged["commence_ts"] - merged["kickoff_ts"]).dt.total_seconds().abs()
    merged = merged[merged["dt"] < 12 * 3600]  # within 12h of scheduled kickoff
    if merged.empty:
        return pd.DataFrame()
    rename = {
        "home_spread": "spread_line",
        "total_point": "total_line",
        "home_ml": "home_moneyline",
        "away_ml": "away_moneyline",
    }
    for k, v in rename.items():
        if k in merged.columns:
            merged[v] = merged[k]
    keep = ["game_id", "season", "week", "kickoff_utc", "home_team", "away_team",
            "spread_line", "total_line", "home_moneyline", "away_moneyline"]
    df = merged[[c for c in keep if c in merged.columns]].copy()
    df["source"] = "pinnacle"
    return df


def _from_espn() -> pd.DataFrame:
    espn_dir = RAW / "espn" / "odds"
    if not espn_dir.exists():
        return pd.DataFrame()
    games = pd.read_parquet(PROCESSED / "games.parquet",
                            columns=["game_id", "season", "week", "kickoff_utc",
                                     "home_team", "away_team"])
    rows: list[dict] = []
    # Use latest season subdir
    for season_dir in sorted(espn_dir.iterdir()):
        if not season_dir.is_dir():
            continue
        for f in season_dir.glob("*.json"):
            try:
                payload = orjson.loads(f.read_bytes())
            except Exception:
                continue
            items = payload.get("items") or []
            if not items:
                continue
            # Use the first (or highest-priority) provider's snapshot
            item = items[0]
            rows.append({
                "espn_event_id": f.stem,
                "spread_line": item.get("spread"),
                "total_line": item.get("overUnder"),
                "home_moneyline": (item.get("homeTeamOdds") or {}).get("moneyLine"),
                "away_moneyline": (item.get("awayTeamOdds") or {}).get("moneyLine"),
            })
    if not rows:
        return pd.DataFrame()
    # We don't have a game_id mapping from ESPN events; skip if not resolvable.
    # (Left as todo: build xref via team + kickoff match if desired.)
    return pd.DataFrame()  # currently unmapped -> return empty


def _existing_recent(minutes: int = 5) -> set[tuple[str, str]]:
    """Return (game_id, source) tuples captured in the last N minutes."""
    if not OUT.exists():
        return set()
    df = pd.read_parquet(OUT, columns=["game_id", "source", "snapshot_at"])
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"], utc=True, errors="coerce")
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    recent = df[df["snapshot_at"] >= cutoff]
    return set(zip(recent["game_id"].astype(str), recent["source"].astype(str)))


def capture() -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    parts = [_from_nflverse(), _from_pinnacle()]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if df.empty:
        print("No lines to capture (no upcoming games?).")
        return df
    df["snapshot_at"] = now

    # Skip duplicates from last 5 min
    recent = _existing_recent(minutes=5)
    if recent:
        before = len(df)
        df = df[~df.apply(lambda r: (r["game_id"], r["source"]) in recent, axis=1)]
        skipped = before - len(df)
        if skipped:
            print(f"  Skipped {skipped} snapshots taken within last 5 min.")

    if df.empty:
        print("Nothing new to append.")
        return df

    # Append
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        combined = pd.concat([prev, df], ignore_index=True)
    else:
        combined = df
    combined.to_parquet(OUT, index=False)
    print(f"Appended {len(df):,} snapshots -> {OUT} (total rows: {len(combined):,})")
    by_source = df["source"].value_counts().to_dict()
    for src, n in by_source.items():
        print(f"  {src}: {n}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    capture()


if __name__ == "__main__":
    main()
