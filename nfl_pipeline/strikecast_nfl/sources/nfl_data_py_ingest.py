"""Historical NFL ingest via nflreadpy (modern polars-based successor of nfl_data_py).

Produces:
  data/raw/nfl_data_py/schedules_{season}.parquet  — one row per game with
      spread_line, total_line, home_moneyline, away_moneyline, weather, roof, etc.
  data/raw/nfl_data_py/pbp_{season}.parquet        — full play-by-play
  data/raw/nfl_data_py/rosters_{season}.parquet    — weekly rosters

Walk-forward safety: schedules are frozen at ingest time; PBP is per-season and
never modified retroactively. Downstream features MUST filter by as_of_date.

Usage:
    python -m strikecast_nfl.sources.nfl_data_py_ingest --seasons 2006-2025
    python -m strikecast_nfl.sources.nfl_data_py_ingest --seasons 2024,2025
"""
from __future__ import annotations

import argparse
from typing import Iterable

import nflreadpy as nfl
import pandas as pd

from ..paths import RAW

OUT = RAW / "nfl_data_py"


def _parse_seasons(spec: str) -> list[int]:
    seasons: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            seasons.update(range(int(a), int(b) + 1))
        elif part:
            seasons.add(int(part))
    return sorted(seasons)


def _to_pandas(df) -> pd.DataFrame:
    """nflreadpy returns polars; convert at the boundary."""
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


def ingest_schedules(seasons: Iterable[int]) -> pd.DataFrame:
    """Schedules include closing spread_line, total_line, home/away moneyline."""
    seasons = list(seasons)
    df = _to_pandas(nfl.load_schedules(seasons=seasons))
    OUT.mkdir(parents=True, exist_ok=True)
    for season, group in df.groupby("season"):
        out = OUT / f"schedules_{season}.parquet"
        group.to_parquet(out, index=False)
        print(f"  schedules {season}: {len(group):>4} games -> {out.name}")
    return df


def ingest_pbp(seasons: Iterable[int]) -> None:
    """Play-by-play — one file per season."""
    OUT.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        out = OUT / f"pbp_{season}.parquet"
        if out.exists():
            print(f"  pbp {season}: already present, skipping")
            continue
        df = _to_pandas(nfl.load_pbp(seasons=[season]))
        df.to_parquet(out, index=False)
        print(f"  pbp {season}: {len(df):>6} plays -> {out.name}")


def ingest_rosters(seasons: Iterable[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        out = OUT / f"rosters_{season}.parquet"
        if out.exists():
            continue
        try:
            df = _to_pandas(nfl.load_rosters_weekly(seasons=[season]))
        except Exception as e:
            print(f"  rosters {season}: skipped ({e})")
            continue
        df.to_parquet(out, index=False)
        print(f"  rosters {season}: {len(df):>5} rows -> {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2006-2025",
                    help="e.g. '2006-2025' or '2024,2025'")
    ap.add_argument("--skip-pbp", action="store_true", help="Skip play-by-play (fast).")
    ap.add_argument("--skip-rosters", action="store_true")
    args = ap.parse_args()

    seasons = _parse_seasons(args.seasons)
    print(f"Seasons: {seasons[0]}..{seasons[-1]}  ({len(seasons)} total)")

    print("\n[1/3] Schedules (closing lines)")
    ingest_schedules(seasons)

    if not args.skip_pbp:
        print("\n[2/3] Play-by-play")
        ingest_pbp(seasons)

    if not args.skip_rosters:
        print("\n[3/3] Weekly rosters")
        ingest_rosters(seasons)

    print("\nDone.")


if __name__ == "__main__":
    main()
