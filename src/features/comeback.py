"""Rolling comeback features per team — pre-game lookback, no lookahead.

Features generated (rolling last 30 games, shift(1) to avoid lookahead):
  cb_rate_l30         rolling comeback-win rate  (comebacks / games played)
  cb_win_rate_l30     comeback wins / total wins  (clutch rate among winners)
  cb_avg_deficit_l30  avg run deficit when team came back (0 if no comeback)
  cb_max_deficit_l30  max deficit overcome in window

Saved to data/processed/features_comeback.parquet with columns:
  game_pk, side (home/away), cb_rate_l30, cb_win_rate_l30,
  cb_avg_deficit_l30, cb_max_deficit_l30

Run standalone:
  python -m src.features.comeback
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_comeback.parquet"

WINDOW = 30
MIN_PERIODS = 10


def build() -> Path:
    cb_path = PROCESSED / "comeback_analysis.parquet"
    if not cb_path.exists():
        raise FileNotFoundError(
            "Run python -m src.analysis.comeback_analysis first to generate comeback_analysis.parquet"
        )

    cb = pd.read_parquet(cb_path)
    cb["game_date"] = pd.to_datetime(cb["game_date"])
    cb = cb.dropna(subset=["home_team_abbrev", "away_team_abbrev", "home_win"])
    cb = cb.sort_values("game_date").reset_index(drop=True)

    # ── Build team-game timeline ──────────────────────────────────────────────
    rows = []
    for _, r in cb.iterrows():
        hw = int(r["home_win"])
        was_cb = bool(r.get("was_comeback", False))
        deficit = float(r.get("max_deficit_winner", 0) or 0)

        rows.append({
            "team": r["home_team_abbrev"],
            "game_pk": r["game_pk"],
            "game_date": r["game_date"],
            "won": hw == 1,
            "comeback_win": hw == 1 and was_cb,
            "deficit_when_cb": deficit if (hw == 1 and was_cb) else 0.0,
        })
        rows.append({
            "team": r["away_team_abbrev"],
            "game_pk": r["game_pk"],
            "game_date": r["game_date"],
            "won": hw == 0,
            "comeback_win": hw == 0 and was_cb,
            "deficit_when_cb": deficit if (hw == 0 and was_cb) else 0.0,
        })

    tdf = pd.DataFrame(rows)
    tdf = tdf.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

    # ── Rolling features (shift(1) = strictly prior games) ───────────────────
    def _roll(series, agg="mean"):
        shifted = series.shift(1)
        if agg == "mean":
            return shifted.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
        if agg == "max":
            return shifted.rolling(WINDOW, min_periods=MIN_PERIODS).max()
        if agg == "sum":
            return shifted.rolling(WINDOW, min_periods=MIN_PERIODS).sum()
        raise ValueError(agg)

    grp = tdf.groupby("team", group_keys=False)

    tdf["cb_rate_l30"]     = grp["comeback_win"].transform(lambda x: _roll(x.astype(float)))
    tdf["wins_l30"]        = grp["won"].transform(lambda x: _roll(x.astype(float), "sum"))
    tdf["cb_wins_l30"]     = grp["comeback_win"].transform(lambda x: _roll(x.astype(float), "sum"))
    tdf["cb_avg_def_l30"]  = grp["deficit_when_cb"].transform(lambda x: _roll(x))
    tdf["cb_max_def_l30"]  = grp["deficit_when_cb"].transform(lambda x: _roll(x, "max"))

    # cb_win_rate = cb wins / total wins (how clutch are their wins?)
    tdf["cb_win_rate_l30"] = (tdf["cb_wins_l30"] / tdf["wins_l30"].clip(lower=1)).round(4)

    feat_cols = ["cb_rate_l30", "cb_win_rate_l30", "cb_avg_def_l30", "cb_max_def_l30"]

    # ── Join back to games for home + away sides ─────────────────────────────
    home_side = (
        tdf[tdf["team"].isin(cb["home_team_abbrev"])]
        .merge(cb[["game_pk", "home_team_abbrev"]], left_on=["game_pk", "team"],
               right_on=["game_pk", "home_team_abbrev"], how="inner")
        [["game_pk"] + feat_cols]
        .assign(side="home")
    )
    away_side = (
        tdf[tdf["team"].isin(cb["away_team_abbrev"])]
        .merge(cb[["game_pk", "away_team_abbrev"]], left_on=["game_pk", "team"],
               right_on=["game_pk", "away_team_abbrev"], how="inner")
        [["game_pk"] + feat_cols]
        .assign(side="away")
    )

    out = pd.concat([home_side, away_side], ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows, {len(feat_cols)} features × 2 sides)")
    nan_frac = out[feat_cols].isna().mean()
    print("NaN fractions:\n", nan_frac.to_string())
    return OUT


if __name__ == "__main__":
    build()
