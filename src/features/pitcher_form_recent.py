"""Starter pitcher RECENT form — last 5 starts (vs the L15 already in model).

Hypothesis: in-season form swings happen over 3-5 starts, not 15. A pitcher
who just got rocked in his last 3 outings shows up in L5 (xwoba_l5 spikes)
before L15 has time to register the change. L5 catches mechanical issues,
injuries hidden in plain sight, batter book updates faster.

Same Statcast peripherals as the L15 features:
  starter_xwoba_l5, starter_barrel_against_l5,
  starter_k_pct_l5, starter_bb_pct_l5,
  starter_velo_mean_l5, starter_spin_mean_l5, starter_pa_l5

Output: data/processed/features_pitcher_l5.parquet

Run standalone:
  python -m src.features.pitcher_form_recent
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

WINDOW = 5


def _per_start_stats(pitches: pd.DataFrame) -> pd.DataFrame:
    p = pitches.copy()
    is_pa_end = p["woba_denom"].fillna(0) > 0
    pa = p[is_pa_end].copy()
    pa["is_batted"] = pa["launch_speed"].notna()
    pa["is_barrel"] = (pa["launch_speed_angle"].fillna(0) == 6).astype(int)
    pa["is_k"]  = pa["events"].fillna("").str.contains("strikeout", case=False).astype(int)
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
    return out[["game_pk", "pitcher", "game_date", "pa", "xwoba",
                "barrel_against", "k_pct", "bb_pct", "velo_mean", "spin_mean"]]


def build() -> Path:
    pitches = pd.read_parquet(
        PROCESSED / "pitches.parquet",
        columns=["game_pk", "pitcher", "game_date", "woba_denom",
                 "estimated_woba_using_speedangle", "events", "launch_speed",
                 "launch_speed_angle", "release_speed", "release_spin_rate"],
    )
    starts = _per_start_stats(pitches)
    df = starts.sort_values(["pitcher", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("pitcher", sort=False)
    cols = ["xwoba", "barrel_against", "k_pct", "bb_pct", "velo_mean", "spin_mean", "pa"]
    for c in cols:
        df[f"starter_{c}_l{WINDOW}"] = g[c].shift(1).rolling(WINDOW, min_periods=2).mean()

    keep = ["game_pk", "pitcher", "game_date"] + [f"starter_{c}_l{WINDOW}" for c in cols]
    out_df = df[keep].copy()
    out = PROCESSED / "features_pitcher_l5.parquet"
    out_df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(out_df):,} rows)")
    nan_frac = out_df[[c for c in keep if c not in ("game_pk","pitcher","game_date")]].isna().mean()
    print("NaN fractions:")
    print(nan_frac.round(3).to_string())
    return out


if __name__ == "__main__":
    build()
