"""Pitcher trend + blowup features — captures direction-of-form, not just mean.

Two distinct signals not currently in the model:
  1. TREND (slope of OLS regression on last N starts):
     - starter_xwoba_trend_l5  — positive = getting WORSE recently
     - starter_k_pct_trend_l5  — positive = getting BETTER
     - starter_velo_trend_l5   — velocity going down = injury risk
  2. BLOWUP RATE (volatility-of-disaster):
     - starter_blowup_rate_l10 — fraction of L10 starts with 5+ runs allowed
     - starter_disaster_count_l10 — count of starts in L10 with 5+ ER

The mean (xwoba_l15) doesn't distinguish:
  - A pitcher who allowed 0,0,0,5,5 ER in 5 starts (volatile, 2.0 ERA mean)
  - A pitcher who allowed 2,2,2,2,2 ER (consistent, 2.0 ERA mean)
The trend + blowup features do.

Output: data/processed/features_pitcher_trend.parquet
Key: (game_pk, pitcher)

Run standalone:
  python -m src.features.pitcher_trend
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_pitcher_trend.parquet"


def _per_start_stats(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, pitcher) with that start's aggregates."""
    p = pitches.copy()
    is_pa_end = p["woba_denom"].fillna(0) > 0
    pa = p[is_pa_end].copy()
    pa["is_batted"] = pa["launch_speed"].notna()
    pa["is_k"]  = pa["events"].fillna("").str.contains("strikeout", case=False).astype(int)
    pa["is_bb"] = pa["events"].fillna("").isin(["walk", "intent_walk"]).astype(int)

    g = pa.groupby(["game_pk", "pitcher", "game_date"], dropna=False)
    out = g.agg(
        pa=("woba_denom", "sum"),
        xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        xwoba_n=("estimated_woba_using_speedangle", "count"),
        k=("is_k", "sum"),
        bb=("is_bb", "sum"),
        velo_mean=("release_speed", "mean"),
    ).reset_index()
    out["xwoba"] = out["xwoba_sum"] / out["xwoba_n"].replace(0, np.nan)
    out["k_pct"] = out["k"]  / out["pa"].replace(0, np.nan)
    out["bb_pct"] = out["bb"] / out["pa"].replace(0, np.nan)
    return out[["game_pk", "pitcher", "game_date", "pa",
                 "xwoba", "k_pct", "bb_pct", "velo_mean"]]


def _per_start_runs_allowed(pb: pd.DataFrame) -> pd.DataFrame:
    """Per-start earned runs allowed (for blowup detection).

    Uses player_box pitching stats. A 'start' is when the pitcher appeared as
    starting pitcher.
    """
    pit = pb[(pb["appeared_pitching"] == True)].copy()
    pit["er"] = pd.to_numeric(pit["pit_earnedRuns"], errors="coerce").fillna(0).astype(int)
    # The starter has the most batters faced typically — pick the pitcher with
    # max bf per (game_pk, side).
    pit["bf"] = pd.to_numeric(pit["pit_battersFaced"], errors="coerce").fillna(0)
    idx = pit.groupby(["game_pk", "side"])["bf"].idxmax()
    starters = pit.loc[idx, ["game_pk", "player_id", "er"]].rename(
        columns={"player_id": "pitcher"})
    return starters


def _slope(values: pd.Series) -> float:
    """OLS slope of values against position index. NaN if < 3 points or all-NaN."""
    v = values.dropna().to_numpy()
    if len(v) < 3:
        return np.nan
    x = np.arange(len(v), dtype=float)
    # Slope = cov(x,y) / var(x)
    x_mean = x.mean()
    y_mean = v.mean()
    num = ((x - x_mean) * (v - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    if den == 0:
        return np.nan
    return float(num / den)


def build() -> Path:
    pitches = pd.read_parquet(
        PROCESSED / "pitches.parquet",
        columns=["game_pk", "pitcher", "game_date", "woba_denom",
                 "estimated_woba_using_speedangle", "events",
                 "launch_speed", "release_speed"],
    )
    starts = _per_start_stats(pitches)

    # Add per-start ER from player_box
    pb = pd.read_parquet(
        PROCESSED / "player_box.parquet",
        columns=["game_pk", "side", "player_id", "appeared_pitching",
                 "pit_earnedRuns", "pit_battersFaced"],
    )
    starters_er = _per_start_runs_allowed(pb)
    starters_er["pitcher"] = starters_er["pitcher"].astype("Int64")
    starts["pitcher"] = starts["pitcher"].astype("Int64")
    starts = starts.merge(starters_er, on=["game_pk", "pitcher"], how="left")
    starts["er"] = starts["er"].fillna(0).astype(int)
    starts["is_blowup"] = (starts["er"] >= 5).astype(int)

    # Sort chronologically per pitcher
    starts = starts.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    g = starts.groupby("pitcher", sort=False)

    # ── Trend features (slope of last 5 starts, shifted by 1 to avoid lookahead) ──
    print("computing slopes (this takes ~30s on full dataset)...")
    for col in ["xwoba", "k_pct", "velo_mean"]:
        s = g[col].shift(1)
        # Rolling apply with our _slope function
        starts[f"starter_{col}_trend_l5"] = s.groupby(starts["pitcher"]).transform(
            lambda x: x.rolling(window=5, min_periods=3).apply(_slope, raw=False)
        )

    # ── Blowup rate (% of last 10 starts with 5+ ER) ──
    starts["starter_blowup_rate_l10"] = g["is_blowup"].shift(1).groupby(starts["pitcher"]).transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    )
    starts["starter_disaster_count_l10"] = g["is_blowup"].shift(1).groupby(starts["pitcher"]).transform(
        lambda x: x.rolling(window=10, min_periods=3).sum()
    )

    keep = ["game_pk", "pitcher", "game_date",
             "starter_xwoba_trend_l5", "starter_k_pct_trend_l5",
             "starter_velo_mean_trend_l5",
             "starter_blowup_rate_l10", "starter_disaster_count_l10"]
    keep = [c for c in keep if c in starts.columns]
    out_df = starts[keep].copy()
    out_df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out_df):,} rows)")
    nan_frac = out_df[[c for c in keep if c not in ("game_pk","pitcher","game_date")]].isna().mean()
    print("NaN fractions:")
    print(nan_frac.round(3).to_string())
    print()
    print("Sample stats:")
    print(out_df[[c for c in keep if c not in ("game_pk","pitcher","game_date")]].describe().round(4).to_string())
    return OUT


if __name__ == "__main__":
    build()
