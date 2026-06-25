"""Bullpen fatigue features — outs/IP that relievers threw recently.

For each (game_pk, side):
  - Identify the starter (first pitcher at_bat_number==min for that side)
  - Sum IP of all non-starter pitchers from player_box → bullpen_ip
  - Rolling LAGGED sums per team:
      bullpen_ip_l1d  = bullpen IP in previous game
      bullpen_ip_l3d  = rolling sum of bullpen IP over last 3 games
      bullpen_ip_l5d  = rolling sum over last 5 games
      bullpen_app_l3d = # appearances in last 3 games (proxy for "how depleted")
      bullpen_runs_allowed_l5 = bullpen runs allowed in last 5 games (quality proxy)

Output: data/processed/features_bullpen.parquet
Key: game_pk, side
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def _parse_ip(value) -> float:
    """MLB Stats API stores IP as '5.2' meaning 5 innings + 2 outs. Convert to decimal."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    s = str(value)
    if "." in s:
        try:
            whole, frac = s.split(".", 1)
            return float(whole) + int(frac[0]) / 3.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _starters_per_game_side(pitches_path: Path) -> pd.DataFrame:
    """For each (game_pk, pitching_side), pick the pitcher with the lowest at_bat_number."""
    p = pd.read_parquet(pitches_path, columns=[
        "game_pk", "inning_topbot", "pitcher", "at_bat_number",
    ])
    # inning_topbot=Top → home team in field → home pitcher
    p["side"] = np.where(p["inning_topbot"].astype(str).str.startswith("T"),
                          "home", "away")
    idx = p.groupby(["game_pk", "side"])["at_bat_number"].idxmin()
    return p.loc[idx, ["game_pk", "side", "pitcher"]].rename(columns={"pitcher": "starter_id"})


def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet",
                             columns=["game_pk", "game_date", "home_team_abbrev",
                                      "away_team_abbrev", "home_score", "away_score"])
    games["game_date"] = pd.to_datetime(games["game_date"])

    starters = _starters_per_game_side(PROCESSED / "pitches.parquet")

    pb = pd.read_parquet(PROCESSED / "player_box.parquet",
                          columns=["game_pk", "side", "player_id",
                                   "pit_inningsPitched", "pit_runs",
                                   "pit_earnedRuns", "appeared_pitching"])
    pb = pb[pb["appeared_pitching"] == True].copy()
    pb["ip_dec"] = pb["pit_inningsPitched"].map(_parse_ip)
    pb["runs"] = pd.to_numeric(pb["pit_runs"], errors="coerce").fillna(0.0)

    # Merge starter id so we can filter to bullpen pitchers
    pb["player_id"] = pb["player_id"].astype("Int64")
    starters["starter_id"] = starters["starter_id"].astype("Int64")
    pb = pb.merge(starters, on=["game_pk", "side"], how="left")
    pb["is_starter"] = (pb["player_id"] == pb["starter_id"])

    bullpen = pb[~pb["is_starter"]].groupby(["game_pk", "side"]).agg(
        bullpen_ip=("ip_dec", "sum"),
        bullpen_apps=("player_id", "nunique"),
        bullpen_runs_allowed=("runs", "sum"),
    ).reset_index()

    # Add team identity per (game_pk, side) for rolling per-team
    home_pk = games[["game_pk", "game_date", "home_team_abbrev"]].rename(
        columns={"home_team_abbrev": "team"})
    home_pk["side"] = "home"
    away_pk = games[["game_pk", "game_date", "away_team_abbrev"]].rename(
        columns={"away_team_abbrev": "team"})
    away_pk["side"] = "away"
    team_lookup = pd.concat([home_pk, away_pk], ignore_index=True)

    df = team_lookup.merge(bullpen, on=["game_pk", "side"], how="left")
    df["bullpen_ip"] = df["bullpen_ip"].fillna(0.0)
    df["bullpen_apps"] = df["bullpen_apps"].fillna(0.0)
    df["bullpen_runs_allowed"] = df["bullpen_runs_allowed"].fillna(0.0)

    # Per-team rolling LAGGED windows (sorted chronologically)
    df = df.sort_values(["team", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("team", sort=False)

    def _rolling_shifted_sum(s, w):
        return s.shift(1).rolling(window=w, min_periods=1).sum()

    df["bullpen_ip_l1d"] = g["bullpen_ip"].shift(1).fillna(0.0)
    df["bullpen_ip_l3d"] = g["bullpen_ip"].transform(lambda s: _rolling_shifted_sum(s, 3))
    df["bullpen_ip_l5d"] = g["bullpen_ip"].transform(lambda s: _rolling_shifted_sum(s, 5))
    df["bullpen_apps_l3d"] = g["bullpen_apps"].transform(lambda s: _rolling_shifted_sum(s, 3))
    df["bullpen_runs_l5"] = g["bullpen_runs_allowed"].transform(lambda s: _rolling_shifted_sum(s, 5))

    keep = ["game_pk", "side",
            "bullpen_ip_l1d", "bullpen_ip_l3d", "bullpen_ip_l5d",
            "bullpen_apps_l3d", "bullpen_runs_l5"]
    out = df[keep].copy()
    p = PROCESSED / "features_bullpen.parquet"
    out.to_parquet(p, index=False)
    print(f"wrote {p} ({len(out):,} rows)")
    print(f"  median bullpen_ip_l3d: {out['bullpen_ip_l3d'].median():.2f}")
    return p


if __name__ == "__main__":
    build()
