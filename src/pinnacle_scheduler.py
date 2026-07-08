"""Pinnacle odds scheduler — pull sharp odds ~1h before each game, credit-aware.

One bulk call to the-odds-api returns ALL games (~3 credits), so fetching
per-game is not cheaper. Instead we TIME the pulls: a pull ~1h before a game
covers every game starting within the next COVER window, so clustered games
share a single call. Typical MLB day → ~4-6 pulls (~12-18 credits).

Greedy schedule:
  - Sort today's game start times.
  - For the earliest uncovered game, schedule a pull at (start - LEAD_MIN).
  - That pull covers every game starting within COVER_MIN after it.
  - Repeat for the next uncovered game.

After each scheduled time the script runs:
  python -m src.ingest_odds_api      (pull Pinnacle + soft books)
  python -m src.normalize.pinnacle_value   (rebuild sharp-value table)

Run it once in the morning and leave it open; it sleeps between pulls and
exits after the last game. Set ODDS_API_KEY in the environment first.

  python -m src.pinnacle_scheduler [--lead 60] [--cover 90] [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
PY = sys.executable


def _todays_starts() -> list[pd.Timestamp]:
    g = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "first_pitch_utc"],
    )
    g["game_date"] = pd.to_datetime(g["game_date"])
    today = pd.Timestamp.now().normalize()
    tg = g[g["game_date"] == today].copy()
    tg["fp"] = pd.to_datetime(tg["first_pitch_utc"], utc=True, errors="coerce")
    tg = tg.dropna(subset=["fp"])
    return sorted(pd.Timestamp(t) for t in tg["fp"].unique())


def _schedule(starts: list[pd.Timestamp], lead_min: int, cover_min: int) -> list[pd.Timestamp]:
    """Greedy: one pull per cluster of games within COVER of each other."""
    pulls: list[pd.Timestamp] = []
    covered_until: pd.Timestamp | None = None
    for s in starts:
        if covered_until is not None and s <= covered_until:
            continue
        pull_at = s - pd.Timedelta(minutes=lead_min)
        pulls.append(pull_at)
        # A pull's odds stay fresh for games starting within COVER of the PULL
        # (not of the triggering game), so no game's odds are older than COVER.
        covered_until = pull_at + pd.Timedelta(minutes=cover_min)
    return pulls


def _run_pull() -> None:
    print(f"[{datetime.now():%H:%M:%S}] pulling Pinnacle + soft books...")
    r1 = subprocess.run([PY, "-m", "src.ingest_odds_api"], cwd=".")
    if r1.returncode != 0:
        print("  ingest failed (quota? key?) — skipping normalize")
        return
    subprocess.run([PY, "-m", "src.normalize.pinnacle_value"], cwd=".")
    print(f"[{datetime.now():%H:%M:%S}] sharp-value table updated.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=60, help="minutes before game start to pull")
    ap.add_argument("--cover", type=int, default=90, help="minutes one pull stays fresh for later games")
    ap.add_argument("--dry-run", action="store_true", help="print schedule, don't pull")
    args = ap.parse_args()

    starts = _todays_starts()
    if not starts:
        print("No games with start times today. Run update_data.bat first.")
        return
    pulls = _schedule(starts, args.lead, args.cover)
    now = pd.Timestamp.now(tz="UTC")

    print(f"Today: {len(starts)} game start times → {len(pulls)} scheduled pulls "
          f"(~{len(pulls) * 3} credits)")
    for p in pulls:
        et = p.tz_convert("America/New_York")
        status = "past" if p < now else "pending"
        print(f"  pull {et:%I:%M %p ET}  [{status}]")

    if args.dry_run:
        return

    future = [p for p in pulls if p > now]
    if not future:
        print("\nAll scheduled pulls are in the past. Doing one immediate pull for freshness.")
        _run_pull()
        return

    print(f"\nWaiting for {len(future)} upcoming pulls. Leave this window open.\n")
    for p in future:
        wait_s = (p - pd.Timestamp.now(tz="UTC")).total_seconds()
        if wait_s > 0:
            et = p.tz_convert("America/New_York")
            print(f"[{datetime.now():%H:%M:%S}] sleeping until {et:%I:%M %p ET} "
                  f"({wait_s/60:.0f} min)...")
            time.sleep(wait_s)
        _run_pull()
    print(f"\n[{datetime.now():%H:%M:%S}] Done — all games have started. Exiting.")


if __name__ == "__main__":
    main()
