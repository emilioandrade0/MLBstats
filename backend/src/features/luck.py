"""Statcast luck / regression features — actual vs expected over/performance.

Regression-to-the-mean is real: BABIP (batting avg on balls in play) is
~70% luck/defense and regresses to ~.290. A team riding a high recent BABIP
has been LUCKY and will likely cool off; the market may still be chasing the
hot results. Fading luck is a classic edge.

Features (team-level, strictly lagged):
  off_babip_l20      — team's offensive BABIP last 20 games
  def_babip_l20      — BABIP allowed by team's pitching last 20 games
  off_babip_luck     — off_babip_l20 - league_mean (positive = lucky, regresses)
  def_babip_luck     — league_mean - def_babip_l20 (positive = lucky pitching)
  babip_luck_net     — off_babip_luck + def_babip_luck (total luck — fade if high)

Output: data/processed/features_luck.parquet  (game_pk, side)

Run:
  python -m src.features.luck
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_luck.parquet"

LEAGUE_BABIP = 0.290
WINDOW = 20
MIN_P = 6


def build() -> Path:
    pb = pd.read_parquet(
        PROCESSED / "player_box.parquet",
        columns=["game_pk", "side", "bat_atBats", "bat_hits",
                 "bat_homeRuns", "bat_strikeOuts", "bat_sacFlies"],
    )
    # Per (game, batting side) BABIP components
    off = pb.groupby(["game_pk", "side"]).agg(
        ab=("bat_atBats", "sum"), h=("bat_hits", "sum"),
        hr=("bat_homeRuns", "sum"), k=("bat_strikeOuts", "sum"),
        sf=("bat_sacFlies", "sum"),
    ).reset_index()
    off["bip"] = off["ab"] - off["k"] - off["hr"] + off["sf"]
    off["babip"] = (off["h"] - off["hr"]) / off["bip"].replace(0, np.nan)

    # Map (game_pk, batting side) → team, and also the OPPOSING (defensive) side
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "game_type",
                 "home_team_abbrev", "away_team_abbrev"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()

    # Offensive BABIP per team
    rows = []
    om = {(r.game_pk, r.side): r.babip for r in off.itertuples(index=False)}
    for _, g in games.iterrows():
        pk = g["game_pk"]
        home_off = om.get((pk, "home"), np.nan)
        away_off = om.get((pk, "away"), np.nan)
        # Defensive BABIP = the BABIP the OPPONENT put up while batting
        rows.append({"game_pk": pk, "game_date": g["game_date"],
                     "team": g["home_team_abbrev"], "side": "home",
                     "off_babip": home_off, "def_babip": away_off})
        rows.append({"game_pk": pk, "game_date": g["game_date"],
                     "team": g["away_team_abbrev"], "side": "away",
                     "off_babip": away_off, "def_babip": home_off})
    tdf = pd.DataFrame(rows).sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

    grp = tdf.groupby("team")
    tdf["off_babip_l20"] = grp["off_babip"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_P).mean())
    tdf["def_babip_l20"] = grp["def_babip"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_P).mean())
    tdf["off_babip_luck"] = tdf["off_babip_l20"] - LEAGUE_BABIP
    tdf["def_babip_luck"] = LEAGUE_BABIP - tdf["def_babip_l20"]
    tdf["babip_luck_net"] = tdf["off_babip_luck"] + tdf["def_babip_luck"]

    out = tdf[["game_pk", "side", "off_babip_l20", "def_babip_l20",
               "off_babip_luck", "def_babip_luck", "babip_luck_net"]].copy()
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows)")
    print(out.drop(columns=["game_pk", "side"]).describe().round(4).to_string())

    # Signal check: does high net luck predict NEXT-game underperformance?
    res = pd.read_parquet(PROCESSED / "games.parquet",
                          columns=["game_pk", "home_team_abbrev", "away_team_abbrev",
                                   "home_score", "away_score"]).dropna()
    res["home_win"] = (res["home_score"] > res["away_score"]).astype(int)
    home_luck = out[out["side"] == "home"][["game_pk", "babip_luck_net"]].rename(
        columns={"babip_luck_net": "luck_h"})
    away_luck = out[out["side"] == "away"][["game_pk", "babip_luck_net"]].rename(
        columns={"babip_luck_net": "luck_a"})
    chk = res.merge(home_luck, on="game_pk").merge(away_luck, on="game_pk")
    chk["luck_diff"] = chk["luck_h"] - chk["luck_a"]
    chk = chk.dropna(subset=["luck_diff"])
    chk["lq"] = pd.qcut(chk["luck_diff"], 5, labels=["home_unlucky", "2", "3", "4", "home_lucky"])
    print("\nHome win rate by luck differential (home luck - away luck):")
    print("(if luck regresses, 'home_lucky' should WIN LESS than raw form suggests)")
    print(chk.groupby("lq", observed=False)["home_win"].agg(["mean", "count"]).round(4).to_string())
    return OUT


if __name__ == "__main__":
    build()
