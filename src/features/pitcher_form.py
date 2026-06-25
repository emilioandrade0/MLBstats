"""Starter pitcher rolling features — strictly lagged.

For each (game_pk, starter_pitcher_id) we look at the pitcher's last 15 starts
BEFORE this game and compute Statcast peripherals.

Output: data/processed/features_pitcher.parquet
Key: (game_pk, pitcher_id, side)
Columns:
  starter_pa_l15, starter_xwoba_l15, starter_k_pct_l15, starter_bb_pct_l15,
  starter_velo_l15, starter_spin_l15, starter_barrel_against_l15,
  starter_days_rest
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def _per_start_stats(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, pitcher) with that start's Statcast aggregates.

    'Start' is loosely any appearance — but rolling will average across starts,
    so non-starts dilute. We filter to the starter later via probable_pitcher id.
    """
    p = pitches.copy()
    is_pa_end = p["woba_denom"].fillna(0) > 0
    pa = p[is_pa_end].copy()
    pa["is_batted"] = pa["launch_speed"].notna()
    pa["is_barrel"] = (pa["launch_speed_angle"].fillna(0) == 6).astype(int)
    pa["is_k"] = pa["events"].fillna("").str.contains("strikeout", case=False).astype(int)
    pa["is_bb"] = pa["events"].fillna("").isin(["walk", "intent_walk"]).astype(int)

    g = pa.groupby(["game_pk", "pitcher", "game_date"], dropna=False)
    out = g.agg(
        pa=("woba_denom", "sum"),
        xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        xwoba_n=("estimated_woba_using_speedangle", "count"),
        barrels=("is_barrel", "sum"),
        batted=("is_batted", "sum"),
        k=("is_k", "sum"),
        bb=("is_bb", "sum"),
        velo_mean=("release_speed", "mean"),
        spin_mean=("release_spin_rate", "mean"),
    ).reset_index()
    out["xwoba"] = out["xwoba_sum"] / out["xwoba_n"].replace(0, np.nan)
    out["barrel_against"] = out["barrels"] / out["batted"].replace(0, np.nan)
    out["k_pct"] = out["k"] / out["pa"].replace(0, np.nan)
    out["bb_pct"] = out["bb"] / out["pa"].replace(0, np.nan)
    return out[["game_pk", "pitcher", "game_date", "pa", "xwoba", "barrel_against",
                "k_pct", "bb_pct", "velo_mean", "spin_mean"]]


def _rolling_lagged_pitcher(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    df = df.sort_values(["pitcher", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("pitcher", sort=False)
    cols = ["xwoba", "barrel_against", "k_pct", "bb_pct", "velo_mean", "spin_mean", "pa"]
    out = {}
    for c in cols:
        shifted = g[c].shift(1)
        out[f"starter_{c}_l{window}"] = shifted.groupby(df["pitcher"]).transform(
            lambda s: s.rolling(window=window, min_periods=3).mean()
        )
    df["_d"] = pd.to_datetime(df["game_date"])
    last = g["_d"].shift(1)
    out["starter_days_rest"] = (df["_d"] - last).dt.days
    df = df.drop(columns=["_d"])
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def build() -> Path:
    pitches = pd.read_parquet(
        PROCESSED / "pitches.parquet",
        columns=["game_pk", "pitcher", "game_date", "woba_denom",
                 "estimated_woba_using_speedangle", "events", "launch_speed",
                 "launch_speed_angle", "release_speed", "release_spin_rate"],
    )
    starts = _per_start_stats(pitches)
    rolled = _rolling_lagged_pitcher(starts, window=15)

    # We'll keep both raw per-start and the rolling-lagged features so the
    # downstream joiner can pick the pitcher's row for a given (game_pk, pitcher).
    keep = ["game_pk", "pitcher", "game_date",
            "starter_xwoba_l15", "starter_barrel_against_l15",
            "starter_k_pct_l15", "starter_bb_pct_l15",
            "starter_velo_mean_l15", "starter_spin_mean_l15",
            "starter_pa_l15", "starter_days_rest"]
    keep = [c for c in keep if c in rolled.columns]
    out_df = rolled[keep].copy()
    out = PROCESSED / "features_pitcher.parquet"
    out_df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(out_df):,} rows)")
    return out


if __name__ == "__main__":
    build()
