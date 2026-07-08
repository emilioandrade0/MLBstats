"""NRFI/YRFI market — will a run score in the FIRST inning?

Target:
  yrfi = 1 if any run scored in inning 1 (Yes-Run-First-Inning), else 0 (NRFI).

First-inning-specific features (all strictly lagged — no lookahead):
  TEAM first-inning offense (how often they put a run on the board in the 1st):
    {side}_fi_scored_rate_l20   — team scored in inning 1 over last 20 games
  PITCHER first-inning defense (how often the starter is scored on in the 1st):
    {side}_sp_fi_allowed_rate_l15 — starter allowed a 1st-inning run over last 15 starts
  Plus top-of-order quality already in the model (lineup_top4_*).

Output:
  data/processed/nrfi_targets.parquet        (game_pk, yrfi)
  data/processed/features_nrfi.parquet       (game_pk + first-inning features)

Run:
  python -m src.features.nrfi
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT_TARGET = PROCESSED / "nrfi_targets.parquet"
OUT_FEATS = PROCESSED / "features_nrfi.parquet"


def _first_inning_runs() -> pd.DataFrame:
    """Per game: runs scored in inning 1 by away and home (and YRFI flag)."""
    plays = pd.read_parquet(
        PROCESSED / "plays.parquet",
        columns=["game_pk", "inning", "is_top", "away_score", "home_score"],
    )
    i1 = plays[plays["inning"] == 1].copy()
    # Running score is post-play; the max within inning 1 = runs scored in the 1st.
    agg = i1.groupby("game_pk").agg(
        away_fi=("away_score", "max"),
        home_fi=("home_score", "max"),
    ).reset_index()
    agg["away_fi"] = agg["away_fi"].fillna(0).astype(int)
    agg["home_fi"] = agg["home_fi"].fillna(0).astype(int)
    agg["yrfi"] = ((agg["away_fi"] + agg["home_fi"]) > 0).astype(int)
    return agg


def build() -> Path:
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "game_type",
                 "home_team_abbrev", "away_team_abbrev",
                 "probable_home_pitcher_id", "probable_away_pitcher_id"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()

    fi = _first_inning_runs()
    games = games.merge(fi, on="game_pk", how="inner")

    # ── Target ───────────────────────────────────────────────────────────────
    target = games[["game_pk", "yrfi"]].copy()
    target.to_parquet(OUT_TARGET, index=False)
    print(f"wrote {OUT_TARGET}  ({len(target):,} rows)  YRFI base rate: {target['yrfi'].mean():.4f}")

    # ── TEAM first-inning scored rate (per team, rolling lagged) ─────────────
    rows = []
    for _, r in games.iterrows():
        rows.append({"game_pk": r["game_pk"], "game_date": r["game_date"],
                     "team": r["away_team_abbrev"], "side": "away",
                     "scored_fi": int(r["away_fi"] > 0)})
        rows.append({"game_pk": r["game_pk"], "game_date": r["game_date"],
                     "team": r["home_team_abbrev"], "side": "home",
                     "scored_fi": int(r["home_fi"] > 0)})
    tdf = pd.DataFrame(rows).sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)
    tdf["fi_scored_rate_l20"] = (
        tdf.groupby("team")["scored_fi"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )

    # ── PITCHER first-inning allowed rate (per starter, rolling lagged) ──────
    # A starter "allowed a 1st-inning run" if the opposing team scored in the 1st
    # in that game. Map starter → did opp score in inning 1.
    pit_rows = []
    for _, r in games.iterrows():
        # home starter faces away offense in the 1st (top of 1st)
        pit_rows.append({"game_pk": r["game_pk"], "game_date": r["game_date"],
                         "pitcher": r["probable_home_pitcher_id"],
                         "allowed_fi": int(r["away_fi"] > 0), "side": "home"})
        # away starter faces home offense in bottom of 1st
        pit_rows.append({"game_pk": r["game_pk"], "game_date": r["game_date"],
                         "pitcher": r["probable_away_pitcher_id"],
                         "allowed_fi": int(r["home_fi"] > 0), "side": "away"})
    pdf = pd.DataFrame(pit_rows).dropna(subset=["pitcher"])
    pdf["pitcher"] = pdf["pitcher"].astype("Int64")
    pdf = pdf.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    pdf["sp_fi_allowed_rate_l15"] = (
        pdf.groupby("pitcher")["allowed_fi"]
        .transform(lambda s: s.shift(1).rolling(15, min_periods=3).mean())
    )

    # ── Assemble per game_pk × side, then pivot to home/away columns ─────────
    team_feat = tdf[["game_pk", "side", "fi_scored_rate_l20"]]
    pit_feat = pdf[["game_pk", "side", "sp_fi_allowed_rate_l15"]]
    feat = team_feat.merge(pit_feat, on=["game_pk", "side"], how="outer")

    home = feat[feat["side"] == "home"].drop(columns=["side"]).rename(
        columns={"fi_scored_rate_l20": "home_fi_scored_rate_l20",
                 "sp_fi_allowed_rate_l15": "home_sp_fi_allowed_rate_l15"})
    away = feat[feat["side"] == "away"].drop(columns=["side"]).rename(
        columns={"fi_scored_rate_l20": "away_fi_scored_rate_l20",
                 "sp_fi_allowed_rate_l15": "away_sp_fi_allowed_rate_l15"})
    out = home.merge(away, on="game_pk", how="outer")

    # Composite "YRFI pressure": each side's scoring rate × opposing starter's
    # allowed rate — the cleanest interaction for first-inning runs.
    out["yrfi_pressure_away"] = out["away_fi_scored_rate_l20"] * out["home_sp_fi_allowed_rate_l15"]
    out["yrfi_pressure_home"] = out["home_fi_scored_rate_l20"] * out["away_sp_fi_allowed_rate_l15"]
    out["yrfi_pressure_total"] = out[["yrfi_pressure_away", "yrfi_pressure_home"]].sum(axis=1)

    out.to_parquet(OUT_FEATS, index=False)
    print(f"wrote {OUT_FEATS}  ({len(out):,} rows)")
    nn = out.drop(columns=["game_pk"]).isna().mean()
    print("NaN fractions:")
    print(nn.round(3).to_string())
    return OUT_FEATS


if __name__ == "__main__":
    build()
