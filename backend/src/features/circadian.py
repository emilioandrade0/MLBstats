"""Circadian / travel-direction features — body-clock displacement.

Sports-science finding: traveling EASTWARD (losing hours) degrades performance
more than westward travel, because the body clock lags. A Pacific-time team
playing a 1pm Eastern game is operating at a 10am body-clock — a real edge to
the home side.

We compute, per game:
  circ_disp_away   — away team's eastward timezone displacement from its HOME
                     tz (+N = traveled N zones east = disadvantaged; <=0 = none)
  circ_disp_home   — same for home team (usually 0; >0 only if the home park is
                     west of where they've been — rare, kept for symmetry)
  circ_adv_home    — circ_disp_away - circ_disp_home (home circadian advantage)
  is_day_game      — 1 if day game (east-travel penalty is worse in day games)
  circ_x_day_home  — circ_adv_home * is_day_game (the interaction that matters)

Timezone scale: ET=0, CT=1, MT=2, PT=3.  Eastward = toward 0.

Output: data/processed/features_circadian.parquet  (game_pk level)

Run:
  python -m src.features.circadian
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_circadian.parquet"

_STATE_TZ = {
    "MD":0,"DC":0,"NY":0,"PA":0,"OH":0,"MI":0,"IN":0,"GA":0,"FL":0,"NC":0,
    "TN":0,"VA":0,"WV":0,"CT":0,"MA":0,"NJ":0,"DE":0,"ON":0,
    "IL":1,"WI":1,"MN":1,"MO":1,"IA":1,"TX":1,"LA":1,"AR":1,"OK":1,"KS":1,
    "MS":1,"AL":1,"NE":1,"SD":1,"ND":1,
    "CO":2,"AZ":2,"NM":2,"UT":2,"ID":2,"MT":2,"WY":2,"NV":2,
    "CA":3,"WA":3,"OR":3,
}


def _tz(state):
    if not state or pd.isna(state):
        return np.nan
    return _STATE_TZ.get(str(state).strip().upper(), np.nan)


def build() -> Path:
    g = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "game_type", "day_night",
                 "home_team_abbrev", "away_team_abbrev", "venue_state"],
    )
    g = g[g["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    g["venue_tz"] = g["venue_state"].map(_tz)

    # Each team's HOME timezone = the tz of venues where it is the home team
    # (mode across the dataset).
    home_tz = (
        g.dropna(subset=["venue_tz"])
         .groupby("home_team_abbrev")["venue_tz"]
         .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    )
    team_home_tz = home_tz.to_dict()

    g["away_home_tz"] = g["away_team_abbrev"].map(team_home_tz)
    g["home_home_tz"] = g["home_team_abbrev"].map(team_home_tz)

    # Eastward displacement = home_tz_of_team - venue_tz  (positive = east travel)
    g["circ_disp_away"] = (g["away_home_tz"] - g["venue_tz"]).clip(lower=0)
    g["circ_disp_home"] = (g["home_home_tz"] - g["venue_tz"]).clip(lower=0)
    g["circ_adv_home"]  = g["circ_disp_away"] - g["circ_disp_home"]
    g["is_day_game"]    = (g["day_night"].astype(str).str.lower() == "day").astype("int8")
    g["circ_x_day_home"] = g["circ_adv_home"] * g["is_day_game"]

    for c in ["circ_disp_away", "circ_disp_home", "circ_adv_home", "circ_x_day_home"]:
        g[c] = g[c].fillna(0.0)

    out = g[["game_pk", "circ_disp_away", "circ_disp_home", "circ_adv_home",
             "is_day_game", "circ_x_day_home"]].copy()
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows)")
    print(out.drop(columns=["game_pk"]).describe().round(3).to_string())
    print()
    # Quick signal check: home WR by away eastward displacement
    res = pd.read_parquet(PROCESSED / "games.parquet",
                          columns=["game_pk", "home_score", "away_score"]).dropna()
    res["home_win"] = (res["home_score"] > res["away_score"]).astype(int)
    chk = out.merge(res, on="game_pk", how="inner")
    print("Home win rate by away eastward displacement:")
    print(chk.groupby("circ_disp_away")["home_win"].agg(["mean", "count"]).round(4).to_string())
    return OUT


if __name__ == "__main__":
    build()
