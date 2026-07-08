"""Creative / unconventional features — momentum, fatigue, line movement.

Signals NOT captured by the standard sabermetric features:

  TEAM-level (game_pk, side):
    streak_signed    — current consecutive W/L streak going in (+3 = 3 wins,
                       -2 = 2 losses). Momentum distinct from win_pct_l30.
    games_last_7d    — schedule density (fatigue): games played in prior 7 days.
    day_after_night  — 1 if today is a day game and the team's previous game
                       was at night (classic fatigue spot).
    rest_advantage   — days_rest minus opponent's days_rest (relative freshness).

  GAME-level (game_pk):
    line_move_home   — home implied prob: close - open (sharp money direction).
    line_move_total  — total line: close - open (where O/U money went).

All strictly lagged (shift) to forbid lookahead.

Output:
  data/processed/features_creative_team.parquet
  data/processed/features_creative_game.parquet

Run:
  python -m src.features.creative
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT_TEAM = PROCESSED / "features_creative_team.parquet"
OUT_GAME = PROCESSED / "features_creative_game.parquet"


def _build_team():
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "game_type", "day_night",
                 "home_team_abbrev", "away_team_abbrev",
                 "home_score", "away_score"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    games = games.dropna(subset=["home_score", "away_score"])
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # Per-team timeline
    rows = []
    for _, r in games.iterrows():
        for side, team, won in [("home", r["home_team_abbrev"], r["home_win"] == 1),
                                 ("away", r["away_team_abbrev"], r["home_win"] == 0)]:
            rows.append({
                "game_pk": r["game_pk"], "game_date": r["game_date"],
                "team": team, "side": side, "won": int(won),
                "is_night": int(str(r["day_night"]).lower() == "night"),
            })
    tdf = pd.DataFrame(rows).sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

    # ── Streak (signed), strictly prior games ────────────────────────────────
    streaks = []
    for _, grp in tdf.groupby("team", sort=False):
        cur = 0
        vals = []
        for won in grp["won"]:
            vals.append(cur)              # streak BEFORE this game (lagged)
            if won == 1:
                cur = cur + 1 if cur > 0 else 1
            else:
                cur = cur - 1 if cur < 0 else -1
        streaks.append(pd.Series(vals, index=grp.index))
    tdf["streak_signed"] = pd.concat(streaks).sort_index()

    # ── Schedule density: games in prior 7 days ──────────────────────────────
    dens = []
    for _, grp in tdf.groupby("team", sort=False):
        dates = grp["game_date"].to_numpy()
        out = np.zeros(len(dates))
        for i in range(len(dates)):
            lo = dates[i] - np.timedelta64(7, "D")
            out[i] = ((dates[:i] >= lo) & (dates[:i] < dates[i])).sum()
        dens.append(pd.Series(out, index=grp.index))
    tdf["games_last_7d"] = pd.concat(dens).sort_index()

    # ── Day-after-night ──────────────────────────────────────────────────────
    tdf["prev_night"] = tdf.groupby("team")["is_night"].shift(1)
    tdf["day_after_night"] = ((tdf["is_night"] == 0) & (tdf["prev_night"] == 1)).astype(int)

    # ── days_rest per team (for rest advantage) ──────────────────────────────
    tdf["prev_date"] = tdf.groupby("team")["game_date"].shift(1)
    tdf["days_rest"] = (tdf["game_date"] - tdf["prev_date"]).dt.days.clip(0, 10).fillna(3)

    out = tdf[["game_pk", "side", "streak_signed", "games_last_7d",
               "day_after_night", "days_rest"]].copy()
    return out, games


def _add_rest_advantage(team_out, games):
    """rest_advantage = own days_rest - opponent days_rest (per side)."""
    home = team_out[team_out["side"] == "home"][["game_pk", "days_rest"]].rename(
        columns={"days_rest": "rest_h"})
    away = team_out[team_out["side"] == "away"][["game_pk", "days_rest"]].rename(
        columns={"days_rest": "rest_a"})
    rj = home.merge(away, on="game_pk", how="inner")
    rj["rest_adv_home"] = rj["rest_h"] - rj["rest_a"]
    team_out = team_out.merge(rj[["game_pk", "rest_adv_home"]], on="game_pk", how="left")
    team_out["rest_advantage"] = np.where(
        team_out["side"] == "home", team_out["rest_adv_home"], -team_out["rest_adv_home"])
    return team_out.drop(columns=["rest_adv_home", "days_rest"])


def _build_game():
    oc = pd.read_parquet(PROCESSED / "odds_close.parquet")
    sharp = oc[oc["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()

    def _imp(v):
        if pd.isna(v) or v == 0:
            return np.nan
        v = float(v)
        return (abs(v) / (abs(v) + 100)) if v < 0 else (100 / (v + 100))

    sharp["imp_open"]  = sharp["home_ml_open"].apply(_imp)
    sharp["imp_close"] = sharp["home_ml_close"].fillna(sharp["home_ml_current"]).apply(_imp)
    sharp["line_move_home"] = sharp["imp_close"] - sharp["imp_open"]
    sharp["line_move_total"] = sharp["total_close"] - sharp["total_open"]

    agg = sharp.groupby("espn_event_id").agg(
        line_move_home=("line_move_home", "median"),
        line_move_total=("line_move_total", "median"),
    ).reset_index()
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    out = agg.merge(matched, on="espn_event_id", how="inner") \
             .drop(columns=["espn_event_id"]).drop_duplicates("game_pk")
    return out[["game_pk", "line_move_home", "line_move_total"]]


def build():
    team_out, games = _build_team()
    team_out = _add_rest_advantage(team_out, games)
    team_out.to_parquet(OUT_TEAM, index=False)
    print(f"wrote {OUT_TEAM}  ({len(team_out):,} rows)")
    print(team_out.drop(columns=["game_pk", "side"]).describe().round(3).to_string())
    print()

    game_out = _build_game()
    game_out.to_parquet(OUT_GAME, index=False)
    print(f"wrote {OUT_GAME}  ({len(game_out):,} rows)")
    print(game_out.drop(columns=["game_pk"]).describe().round(4).to_string())


if __name__ == "__main__":
    build()
