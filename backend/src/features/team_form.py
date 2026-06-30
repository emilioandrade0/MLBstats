"""Team rolling features — strictly lagged so game G only uses games < G.

Output: data/processed/features_team.parquet
Key: (game_pk, side)
Columns (per side):
  runs_scored_l10, runs_allowed_l10, run_diff_l10
  win_pct_l30
  off_xwoba_l30  (rolling mean of team xwOBA from Statcast)
  def_xwoba_l30  (rolling mean of opponent xwOBA conceded)
  off_barrel_l30, def_barrel_l30
  off_k_pct_l30, def_k_pct_l30
  off_bb_pct_l30, def_bb_pct_l30
  days_since_last_game
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def _per_game_team_offense_from_pitches(pitches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Statcast to per-(game, batting_team) offense quality stats.

    The Savant pitch row has `inning_topbot` ('Top'/'Bot'). Top half = away
    batting. So the batting team is: away_team if Top else home_team.
    """
    p = pitches.copy()
    is_top = p["inning_topbot"].astype(str).str.startswith("T")
    p["bat_team"] = np.where(is_top, p["away_team"], p["home_team"])
    p["pit_team"] = np.where(is_top, p["home_team"], p["away_team"])

    # PA-end marker: woba_denom > 0 marks the final pitch of a PA where outcome is valid.
    is_pa_end = p["woba_denom"].fillna(0) > 0
    pa = p[is_pa_end].copy()

    # batted ball indicators
    pa["is_batted"] = pa["launch_speed"].notna()
    pa["is_barrel"] = (pa["launch_speed_angle"].fillna(0) == 6).astype(int)
    pa["is_k"] = pa["events"].fillna("").str.contains("strikeout", case=False).astype(int)
    pa["is_bb"] = pa["events"].fillna("").isin(["walk", "intent_walk"]).astype(int)

    grp_off = pa.groupby(["game_pk", "bat_team", "game_date"], dropna=False)
    off = grp_off.agg(
        off_pa=("woba_denom", "sum"),
        off_xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        off_xwoba_n=("estimated_woba_using_speedangle", "count"),
        off_barrels=("is_barrel", "sum"),
        off_batted=("is_batted", "sum"),
        off_k=("is_k", "sum"),
        off_bb=("is_bb", "sum"),
    ).reset_index().rename(columns={"bat_team": "team"})
    off["off_xwoba"] = off["off_xwoba_sum"] / off["off_xwoba_n"].replace(0, np.nan)
    off["off_barrel_rate"] = off["off_barrels"] / off["off_batted"].replace(0, np.nan)
    off["off_k_pct"] = off["off_k"] / off["off_pa"].replace(0, np.nan)
    off["off_bb_pct"] = off["off_bb"] / off["off_pa"].replace(0, np.nan)

    grp_def = pa.groupby(["game_pk", "pit_team", "game_date"], dropna=False)
    deff = grp_def.agg(
        def_pa=("woba_denom", "sum"),
        def_xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        def_xwoba_n=("estimated_woba_using_speedangle", "count"),
        def_barrels=("is_barrel", "sum"),
        def_batted=("is_batted", "sum"),
        def_k=("is_k", "sum"),
        def_bb=("is_bb", "sum"),
    ).reset_index().rename(columns={"pit_team": "team"})
    deff["def_xwoba"] = deff["def_xwoba_sum"] / deff["def_xwoba_n"].replace(0, np.nan)
    deff["def_barrel_rate"] = deff["def_barrels"] / deff["def_batted"].replace(0, np.nan)
    deff["def_k_pct"] = deff["def_k"] / deff["def_pa"].replace(0, np.nan)
    deff["def_bb_pct"] = deff["def_bb"] / deff["def_pa"].replace(0, np.nan)

    return off.merge(deff, on=["game_pk", "team", "game_date"], how="outer")


def _per_game_team_runs(games: pd.DataFrame) -> pd.DataFrame:
    """Stack games into long form: one row per (game_pk, team_abbrev) with runs scored/allowed."""
    home = games[["game_pk", "game_date", "home_team_abbrev", "home_score", "away_score"]].copy()
    home.columns = ["game_pk", "game_date", "team", "runs_scored", "runs_allowed"]
    home["is_home"] = 1
    away = games[["game_pk", "game_date", "away_team_abbrev", "away_score", "home_score"]].copy()
    away.columns = ["game_pk", "game_date", "team", "runs_scored", "runs_allowed"]
    away["is_home"] = 0
    return pd.concat([home, away], ignore_index=True)


def _rolling_lagged(df: pd.DataFrame, team_col: str, date_col: str,
                    cols: list[str], windows: dict[str, int]) -> pd.DataFrame:
    """For each team (sorted by date), compute rolling mean of `cols` over the
    `windows[name]` PRIOR games (lagged by 1 to forbid leakage)."""
    df = df.sort_values([team_col, date_col, "game_pk"], kind="mergesort").reset_index(drop=True)
    g = df.groupby(team_col, sort=False)
    out_cols: dict[str, pd.Series] = {}
    for name, w in windows.items():
        for c in cols:
            if c not in df.columns:
                continue
            shifted = g[c].shift(1)
            out_cols[f"{c}_{name}"] = shifted.groupby(df[team_col]).transform(
                lambda s, w=w: s.rolling(window=w, min_periods=max(3, w // 3)).mean()
            )
    # days since last game (using game_date)
    df["_d"] = pd.to_datetime(df[date_col])
    last_d = g["_d"].shift(1)
    out_cols["days_since_last_game"] = (df["_d"] - last_d).dt.days
    df = df.drop(columns=["_d"])
    return pd.concat([df, pd.DataFrame(out_cols, index=df.index)], axis=1)


def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    games = games.dropna(subset=["game_date", "home_team_abbrev", "away_team_abbrev"])
    # Only keep MLB regular + postseason for team-form (drop spring training).
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])]

    runs = _per_game_team_runs(games)

    # Per-game Statcast quality stats (one row per team per game).
    pitches = pd.read_parquet(
        PROCESSED / "pitches.parquet",
        columns=[
            "game_pk", "game_date", "home_team", "away_team", "inning_topbot",
            "estimated_woba_using_speedangle", "woba_denom", "events",
            "launch_speed", "launch_speed_angle",
        ],
    )
    statcast = _per_game_team_offense_from_pitches(pitches)

    df = runs.merge(statcast, on=["game_pk", "team", "game_date"], how="left")
    df["win"] = (df["runs_scored"] > df["runs_allowed"]).astype(int)

    feat_cols = [
        "runs_scored", "runs_allowed", "win",
        "off_xwoba", "off_barrel_rate", "off_k_pct", "off_bb_pct",
        "def_xwoba", "def_barrel_rate", "def_k_pct", "def_bb_pct",
    ]
    df = _rolling_lagged(df, "team", "game_date", feat_cols, {"l10": 10, "l30": 30})

    # Derived: run_diff_l10
    df["run_diff_l10"] = df["runs_scored_l10"] - df["runs_allowed_l10"]
    df = df.rename(columns={"win_l30": "win_pct_l30"})

    keep = ["game_pk", "team", "game_date", "is_home", "days_since_last_game",
            "runs_scored_l10", "runs_allowed_l10", "run_diff_l10", "win_pct_l30",
            "off_xwoba_l30", "off_barrel_rate_l30", "off_k_pct_l30", "off_bb_pct_l30",
            "def_xwoba_l30", "def_barrel_rate_l30", "def_k_pct_l30", "def_bb_pct_l30"]
    # NOTE: L10 versions of xwoba/barrel/k_pct/bb_pct are also computed by the
    # rolling step above but discarded here — they were tested in walk-forward
    # (src/model/team_l10_walkforward.py) and degraded the model (-0.19pp acc,
    # -1.4 AUC) due to high collinearity with the L30 variants.
    keep = [c for c in keep if c in df.columns]
    out_df = df[keep].copy()
    out = PROCESSED / "features_team.parquet"
    out_df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(out_df):,} rows)")
    return out


if __name__ == "__main__":
    build()
