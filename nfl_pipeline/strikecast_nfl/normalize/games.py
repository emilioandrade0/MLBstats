"""Unify schedules across seasons into one canonical games.parquet.

Schema (canonical):
  game_id            str    e.g. "2024_01_KC_BAL"
  season             int
  week               int    1..22 (regular + playoffs)
  game_type          str    REG | WC | DIV | CON | SB
  gameday            date
  gametime           str    HH:MM (kickoff, UTC-4 approx from source)
  kickoff_utc        datetime UTC
  home_team          str    3-letter code
  away_team          str
  home_score         int|NA
  away_score         int|NA
  result             int|NA (home_score - away_score)
  total              int|NA (home_score + away_score)
  overtime           bool
  spread_line        float  closing spread (home perspective, negative = home fav)
  total_line         float  closing total
  home_moneyline     int|NA
  away_moneyline     int|NA
  roof               str    dome | outdoors | closed | open
  surface            str
  temp               float|NA
  wind               float|NA
  stadium            str
  status             str    upcoming | live | final
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..paths import PROCESSED, RAW

SRC = RAW / "nfl_data_py"


KEEP = [
    "game_id", "season", "week", "game_type", "gameday", "gametime",
    "home_team", "away_team", "home_score", "away_score", "result", "total", "overtime",
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "roof", "surface", "temp", "wind", "stadium",
]


def _kickoff_utc(row: pd.Series) -> pd.Timestamp:
    if pd.isna(row["gameday"]):
        return pd.NaT
    t = row["gametime"] if pd.notna(row["gametime"]) else "17:00"
    try:
        return pd.Timestamp(f"{row['gameday']} {t}", tz="US/Eastern").tz_convert("UTC")
    except Exception:
        return pd.Timestamp(row["gameday"], tz="UTC")


def _status(row: pd.Series, now: pd.Timestamp) -> str:
    if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
        return "final"
    if pd.notna(row["kickoff_utc"]) and row["kickoff_utc"] <= now:
        return "live"
    return "upcoming"


def build() -> pd.DataFrame:
    frames = []
    for p in sorted(SRC.glob("schedules_*.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit(f"No schedules found in {SRC}. Run nfl_data_py_ingest first.")

    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].copy()

    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    df["kickoff_utc"] = df.apply(_kickoff_utc, axis=1)

    now = pd.Timestamp.now(tz="UTC")
    df["status"] = df.apply(_status, axis=1, args=(now,))

    df = df.sort_values(["season", "week", "kickoff_utc"]).reset_index(drop=True)

    out = PROCESSED / "games.parquet"
    df.to_parquet(out, index=False)
    finals = int((df["status"] == "final").sum())
    upcoming = int((df["status"] == "upcoming").sum())
    live = int((df["status"] == "live").sum())
    print(f"games.parquet: {len(df):,} rows  (final={finals}  live={live}  upcoming={upcoming})")
    return df


if __name__ == "__main__":
    build()
