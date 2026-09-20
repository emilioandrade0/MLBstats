"""Rolling EPA/play features per team — walk-forward safe.

For each game (team perspective), computes:
  epa_off_l4, epa_off_l8    offensive EPA/play over last 4 / 8 games PRIOR to kickoff
  epa_def_l4, epa_def_l8    defensive EPA/play (opponent EPA against this team)
  pass_epa_off_l4, rush_epa_off_l4
  success_rate_off_l4

Uses only plays whose game_date < current kickoff → no leakage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import PROCESSED, RAW

PBP_DIR = RAW / "nfl_data_py"


def _load_pbp(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = PBP_DIR / f"pbp_{s}.parquet"
        if p.exists():
            cols = [
                "game_id", "season", "week", "game_date", "posteam", "defteam",
                "play_type", "epa", "success", "pass", "rush",
            ]
            df = pd.read_parquet(p, columns=[c for c in cols if c in pd.read_parquet(p, columns=None).columns[:0].tolist() or True])
            # Robust read: some seasons may not have every column
            df = pd.read_parquet(p)
            keep = [c for c in cols if c in df.columns]
            frames.append(df[keep])
    if not frames:
        print("  WARN: no pbp files found — EPA features will be all-NaN.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _team_game_agg(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per (game_id, team) with EPA sums / counts."""
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp["game_date"] = pd.to_datetime(pbp["game_date"], errors="coerce")

    off = (pbp.groupby(["game_id", "season", "week", "game_date", "posteam"], as_index=False)
              .agg(off_plays=("epa", "size"),
                   off_epa=("epa", "sum"),
                   off_success=("success", "sum"),
                   pass_epa=("epa", lambda s: s[pbp.loc[s.index, "pass"] == 1].sum()),
                   rush_epa=("epa", lambda s: s[pbp.loc[s.index, "rush"] == 1].sum()))
              .rename(columns={"posteam": "team"}))
    def_ = (pbp.groupby(["game_id", "defteam"], as_index=False)
               .agg(def_plays=("epa", "size"), def_epa=("epa", "sum"))
               .rename(columns={"defteam": "team"}))
    merged = off.merge(def_, on=["game_id", "team"], how="left")
    return merged.sort_values(["team", "game_date"]).reset_index(drop=True)


def _rolling(df: pd.DataFrame, windows: tuple[int, ...] = (4, 8)) -> pd.DataFrame:
    """Rolling sums SHIFTED by 1 game to guarantee no leakage."""
    df = df.sort_values(["team", "game_date"]).copy()
    grouped = df.groupby("team", sort=False)
    for w in windows:
        for src, dst in [
            ("off_epa", f"off_epa_l{w}"),
            ("def_epa", f"def_epa_l{w}"),
            ("off_plays", f"off_plays_l{w}"),
            ("def_plays", f"def_plays_l{w}"),
            ("off_success", f"off_success_l{w}"),
            ("pass_epa", f"pass_epa_l{w}"),
            ("rush_epa", f"rush_epa_l{w}"),
        ]:
            df[dst] = grouped[src].shift(1).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        # Per-play rates
        df[f"epa_off_per_play_l{w}"] = df[f"off_epa_l{w}"] / df[f"off_plays_l{w}"].replace(0, pd.NA)
        df[f"epa_def_per_play_l{w}"] = df[f"def_epa_l{w}"] / df[f"def_plays_l{w}"].replace(0, pd.NA)
        df[f"success_rate_off_l{w}"] = df[f"off_success_l{w}"] / df[f"off_plays_l{w}"].replace(0, pd.NA)
    return df


def build() -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games.parquet", columns=["game_id", "season", "home_team", "away_team"])
    seasons = sorted(games["season"].unique().tolist())

    print(f"Loading PBP for seasons {seasons[0]}..{seasons[-1]}")
    pbp = _load_pbp(seasons)
    if pbp.empty:
        # Emit an empty features file with just the join key so downstream merges succeed.
        empty = games[["game_id"]].copy()
        path = PROCESSED / "features_epa.parquet"
        empty.to_parquet(path, index=False)
        print(f"features_epa.parquet: {len(empty):,} rows (empty — no PBP available)")
        return empty
    tg = _team_game_agg(pbp)
    tg = _rolling(tg)

    keep = [c for c in tg.columns if c.startswith(("epa_", "success_rate_"))] + ["game_id", "team"]
    tg = tg[keep]

    # Join per-team stats onto games as home/away
    home = tg.rename(columns={c: f"home_{c}" for c in tg.columns if c not in ("game_id", "team")})
    home = home.rename(columns={"team": "home_team"})
    away = tg.rename(columns={c: f"away_{c}" for c in tg.columns if c not in ("game_id", "team")})
    away = away.rename(columns={"team": "away_team"})

    out = games.merge(home, on=["game_id", "home_team"], how="left") \
               .merge(away, on=["game_id", "away_team"], how="left")
    out = out.drop(columns=["season", "home_team", "away_team"])

    path = PROCESSED / "features_epa.parquet"
    out.to_parquet(path, index=False)
    print(f"features_epa.parquet: {len(out):,} rows, {len(out.columns)} cols")
    return out


if __name__ == "__main__":
    build()
