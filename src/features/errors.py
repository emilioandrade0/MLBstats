"""Error-pattern features — defensive execution quality.

Why distinct from existing features:
  xWOBA/barrel_rate capture pitching + hitting quality, but NOT fielding execution.
  Error rate is a direct proxy for defensive consistency; error resilience captures
  how much a team's win probability drops when mistakes happen — a team-specific
  vulnerability that no other feature measures.

Features (shift(1) = strictly prior games, no lookahead):
  error_rate_l20          rolling avg errors committed per game (last 20)
  error_rate_l10          rolling avg errors committed per game (last 10, more reactive)
  error_resilience_l30    WR delta: wr_with_error − wr_no_error over last 30 wins+losses
                          negative = team hurt more by errors; near-zero = absorbs errors
  wr_error_games_l30      win rate in games where team committed >=1 error (last 30)

Output: data/processed/features_errors.parquet
  game_pk | side (home/away) | feature columns

Run standalone:
  python -m src.features.errors
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_errors.parquet"


def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_errors", "away_errors",
                                  "home_score", "away_score"])
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # ── Per-team timeline ─────────────────────────────────────────────────────
    rows = []
    for _, r in games.iterrows():
        hw = int(r["home_win"])
        for side, team in [("home", r["home_team_abbrev"]),
                            ("away", r["away_team_abbrev"])]:
            errors = float(r["home_errors"] if side == "home" else r["away_errors"])
            won    = float(hw == 1 if side == "home" else hw == 0)
            rows.append({
                "team": team,
                "game_pk": r["game_pk"],
                "game_date": r["game_date"],
                "errors": errors,
                "won": won,
                "had_error": float(errors >= 1),
                "won_with_error": float(errors >= 1 and won == 1),
                "won_no_error":   float(errors == 0 and won == 1),
                "had_no_error":   float(errors == 0),
            })

    tdf = pd.DataFrame(rows)
    tdf = tdf.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

    # ── Rolling features ──────────────────────────────────────────────────────
    def _roll(s: pd.Series, w: int) -> pd.Series:
        return s.shift(1).rolling(w, min_periods=max(5, w // 4)).mean()

    def _cond_rate(num: pd.Series, denom: pd.Series, w: int) -> pd.Series:
        """Rolling win rate conditioned on denom==1 (e.g. only error games)."""
        return (
            num.where(denom == 1)
            .shift(1)
            .rolling(w, min_periods=5)
            .mean()
        )

    grp = tdf.groupby("team", group_keys=False)

    tdf["error_rate_l20"] = grp["errors"].transform(lambda x: _roll(x, 20))
    tdf["error_rate_l10"] = grp["errors"].transform(lambda x: _roll(x, 10))

    # WR in error games (last 30)
    tdf["wr_error_games_l30"] = grp.apply(
        lambda g: _cond_rate(g["won_with_error"], g["had_error"], 30),
        include_groups=False
    ).reset_index(level=0, drop=True)

    # WR in no-error games (last 30)
    tdf["wr_no_error_l30"] = grp.apply(
        lambda g: _cond_rate(g["won_no_error"], g["had_no_error"], 30),
        include_groups=False
    ).reset_index(level=0, drop=True)

    # Resilience = WR_error − WR_no_error  (negative means errors hurt more)
    tdf["error_resilience_l30"] = tdf["wr_error_games_l30"] - tdf["wr_no_error_l30"]

    feat_cols = [
        "error_rate_l20",
        "error_rate_l10",
        "wr_error_games_l30",
        "error_resilience_l30",
    ]

    # ── Join back to game_pk × side ──────────────────────────────────────────
    gm_teams = games[["game_pk", "home_team_abbrev", "away_team_abbrev"]].copy()

    home_side = (
        tdf.merge(gm_teams, left_on=["game_pk", "team"],
                  right_on=["game_pk", "home_team_abbrev"], how="inner")
        [["game_pk"] + feat_cols].assign(side="home")
    )
    away_side = (
        tdf.merge(gm_teams, left_on=["game_pk", "team"],
                  right_on=["game_pk", "away_team_abbrev"], how="inner")
        [["game_pk"] + feat_cols].assign(side="away")
    )

    out = pd.concat([home_side, away_side], ignore_index=True)
    out.to_parquet(OUT, index=False)

    nan_frac = out[feat_cols].isna().mean()
    print(f"wrote {OUT}  ({len(out):,} rows, {len(feat_cols)} features x 2 sides)")
    print("NaN fractions:\n", nan_frac.round(3).to_string())
    print("\nSample means:")
    print(out[feat_cols].mean().round(3).to_string())
    return OUT


if __name__ == "__main__":
    build()
