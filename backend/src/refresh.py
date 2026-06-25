"""Smart incremental refresh.

Re-fetches only data that can have changed since last ingest:
  - Today's games (ESPN odds shift through the day, scores roll in live)
  - Optionally yesterday (for late results that hadn't closed)
  - Future scheduled games for line movement

Past days are SKIPPED entirely because:
  - Their raw JSONs already exist on disk
  - The data is immutable (final scores, closing odds)
  - The ingest scripts already `exists()`-check before re-downloading

What gets rebuilt after the raw refresh:
  - odds_close.parquet            (fast, parses all odds JSONs)
  - features_market.parquet       (depends on odds_close)

What does NOT get rebuilt automatically (heavy, done manually):
  - games/team_box/player_box/plays/pitches  — only when major data lands
  - features_team/features_pitcher/features_lineup — same
  - train.parquet + the model ensemble itself

Usage:
    python -m src.refresh                       # quick today-only ESPN refresh
    python -m src.refresh --days-back 1         # include yesterday for late results
    python -m src.refresh --statsapi            # also refresh StatsAPI feed/live
    python -m src.refresh --include-savant      # also refresh current week Statcast
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta

from .ingest import run as run_espn
from .ingest_statsapi import run as run_statsapi
from .ingest_savant import run as run_savant
from .normalize.odds_v2 import build as build_odds_close
from .features.market_close import build as build_features_market


async def refresh_async(
    days_back: int = 0,
    days_forward: int = 0,
    statsapi: bool = False,
    include_savant: bool = False,
    verbose: bool = True,
) -> dict:
    """Smart refresh; returns summary of what was rebuilt."""
    today = date.today()
    start = today - timedelta(days=days_back)
    end = today + timedelta(days=days_forward)

    if verbose:
        print(f"[refresh] window {start} → {end}")
        print("[refresh] ESPN ingest…")
    # ESPN ingest is rate-limit-friendly and reuses _fetch_live_feed semantics:
    # scoreboard + summary + odds + probabilities per event.
    await run_espn(start, end, concurrency=8, force=True)

    if statsapi:
        if verbose:
            print("[refresh] StatsAPI ingest…")
        await run_statsapi(start, end, concurrency=8, force=True)

    if include_savant:
        # Statcast lags a few hours. Refresh the trailing 7 days, no force.
        sav_start = today - timedelta(days=7)
        if verbose:
            print(f"[refresh] Savant {sav_start} → {today}…")
        await run_savant(sav_start, today, concurrency=3, force=False)

    # Rebuild derivative parquets (light)
    if verbose:
        print("[refresh] rebuild odds_close.parquet")
    await asyncio.to_thread(build_odds_close)
    if verbose:
        print("[refresh] rebuild features_market.parquet")
    await asyncio.to_thread(build_features_market)

    return {
        "window": [start.isoformat(), end.isoformat()],
        "statsapi": statsapi,
        "savant": include_savant,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days-back", type=int, default=0,
                   help="Include past N days in the refresh window (default 0 = today only).")
    p.add_argument("--days-forward", type=int, default=0,
                   help="Include future N days (for line movement on scheduled games).")
    p.add_argument("--statsapi", action="store_true",
                   help="Also refresh StatsAPI feed/live JSONs (heavier, normally not needed).")
    p.add_argument("--include-savant", action="store_true",
                   help="Also refresh trailing-7-days Statcast (heavy).")
    a = p.parse_args()
    result = asyncio.run(refresh_async(
        days_back=a.days_back, days_forward=a.days_forward,
        statsapi=a.statsapi, include_savant=a.include_savant,
    ))
    print("\nrefresh result:", result)


if __name__ == "__main__":
    main()
