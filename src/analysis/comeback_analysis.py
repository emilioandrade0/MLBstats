"""Comeback analysis — who remounts, when, and from how far back.

Outputs data/processed/comeback_analysis.parquet with one row per game:
  - max_deficit_winner:   biggest run deficit the winning team faced
  - min_wp_winner:        lowest in-game WP the winning team had (ESPN)
  - was_comeback:         winner trailed by >=1 at some point
  - comeback_from_n:      max deficit bucket (1, 2, 3, 4, 5+)
  - inning_comeback_start: inning where comeback began (last at-bat where still trailing)

Run:
  python -m src.analysis.comeback_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "comeback_analysis.parquet"


def _plays_summary(plays: pd.DataFrame) -> pd.DataFrame:
    """Per game: max deficit winner faced, was it a comeback."""
    # Only use plays that have a score (scoring plays + all plays update score)
    p = plays[["game_pk", "inning", "half_inning", "away_score", "home_score", "is_top"]].copy()
    p = p.dropna(subset=["away_score", "home_score"])

    def _game_stats(g):
        # Sort by inning then half_inning (top before bottom)
        g = g.sort_values(["inning", "is_top"], ascending=[True, False])
        away_s = g["away_score"].values.astype(float)
        home_s = g["home_score"].values.astype(float)

        final_away = away_s[-1]
        final_home = home_s[-1]
        home_win = int(final_home > final_away)

        # Deficit = winner_score - loser_score (negative = trailing)
        if home_win:
            diff = home_s - away_s  # positive = home leading
        else:
            diff = away_s - home_s  # positive = away leading

        # Max deficit the winner was at (most negative value means trailing most)
        # But we only care about states BEFORE the final play
        if len(diff) > 1:
            pre_final = diff[:-1]
        else:
            pre_final = diff

        max_deficit = float(-pre_final.min())  # how many runs behind (0 = never trailed)
        was_comeback = max_deficit >= 1.0

        # Largest deficit bucket
        bucket = min(int(max_deficit), 5)  # cap at 5+

        # Inning where winner last trailed (for comeback games)
        if was_comeback:
            trailing_idxs = np.where(pre_final < 0)[0]
            last_trailing_idx = int(trailing_idxs[-1])
            innings = g["inning"].values
            comeback_inning = int(innings[last_trailing_idx]) if last_trailing_idx < len(innings) else None
        else:
            comeback_inning = None

        return pd.Series({
            "home_win": home_win,
            "final_away": final_away,
            "final_home": final_home,
            "max_deficit_winner": max_deficit,
            "was_comeback": was_comeback,
            "comeback_bucket": bucket,
            "comeback_inning": comeback_inning,
        })

    return plays.groupby("game_pk").apply(_game_stats, include_groups=False).reset_index()


def _wp_summary(wp: pd.DataFrame, xref: pd.DataFrame) -> pd.DataFrame:
    """Per game (via game_pk): min WP the winner had during the game."""
    # xref to map espn_event_id -> game_pk
    xm = xref[["espn_event_id", "game_pk"]].dropna().drop_duplicates()
    xm["espn_event_id"] = xm["espn_event_id"].astype(str).str.strip(".0").str.split(".").str[0]

    wp = wp.copy()
    wp["espn_event_id"] = wp["espn_event_id"].astype(str).str.strip(".0").str.split(".").str[0]
    wp = wp.merge(xm, on="espn_event_id", how="inner")

    # home_win_pct per play → per game: min and mean
    grp = wp.groupby("game_pk")["home_win_pct"].agg(
        wp_min="min",
        wp_max="max",
        wp_mean="mean",
    ).reset_index()
    return grp


def main():
    print("Loading data…")
    games = pd.read_parquet(PROCESSED / "games.parquet")
    plays = pd.read_parquet(PROCESSED / "plays.parquet")
    wp    = pd.read_parquet(PROCESSED / "win_probability.parquet")
    xref  = pd.read_parquet(PROCESSED / "games_xref.parquet")
    games_xref = pd.read_parquet(PROCESSED / "games_xref.parquet")

    # Filter to regular season finished games
    finished = games[
        games["status_code"].isin(["F", "FR", "FT", "O", "OR"]) &
        games["game_type"].isin(["R", "P", "W"])  # regular + postseason
    ].copy()
    finished["game_date"] = pd.to_datetime(finished["game_date"])

    print(f"  games: {len(finished):,}  plays: {len(plays):,}  wp plays: {len(wp):,}")

    # ── Plays summary ──────────────────────────────────────────────────────
    print("Computing comeback stats from plays…")
    play_stats = _plays_summary(plays)
    print(f"  {len(play_stats):,} games with play data")

    # ── Win-probability summary ────────────────────────────────────────────
    print("Computing min WP from ESPN win-probability…")
    wp_stats = _wp_summary(wp, xref)
    print(f"  {len(wp_stats):,} games with WP data")

    # ── Merge everything ──────────────────────────────────────────────────
    df = finished.merge(play_stats, on="game_pk", how="left")
    df = df.merge(wp_stats, on="game_pk", how="left")

    # Derive: min WP for the WINNER (not necessarily home team)
    df["min_wp_winner"] = np.where(
        df["home_win"] == 1,
        df["wp_min"],
        1.0 - df["wp_max"]  # away winner → min WP = 1 - max home WP
    )
    # Implied moneyline at lowest WP (decimal odds)
    df["implied_moneyline_worst"] = (1.0 / df["min_wp_winner"].clip(0.01, 0.99)).round(2)

    # Deficit bucket labels
    bucket_map = {0: "tied/ahead", 1: "1 run", 2: "2 runs", 3: "3 runs", 4: "4 runs", 5: "5+ runs"}
    df["deficit_label"] = df["comeback_bucket"].map(bucket_map).fillna("unknown")

    # Save
    df.to_parquet(OUT, index=False)
    print(f"\nSaved {len(df):,} rows -> {OUT}")

    # ── Quick summary print ────────────────────────────────────────────────
    print("\n=== OVERALL COMEBACK RATE ===")
    total = df["was_comeback"].notna().sum()
    cbs = df["was_comeback"].sum()
    print(f"  Comeback games (winner trailed >=1 run): {int(cbs):,} / {int(total):,} = {cbs/total*100:.1f}%")

    print("\n=== BY DEFICIT ===")
    dist = df[df["was_comeback"]].groupby("deficit_label").size().sort_index()
    for k, v in dist.items():
        print(f"  {k}: {v:,}")

    print("\n=== TEAMS WITH MOST COMEBACKS (2026) ===")
    df26 = df[df["game_date"].dt.year == 2026]
    away_cb = df26[df26["was_comeback"] & (df26["home_win"] == 0)].groupby("away_team_abbrev").size()
    home_cb = df26[df26["was_comeback"] & (df26["home_win"] == 1)].groupby("home_team_abbrev").size()
    combined = away_cb.add(home_cb, fill_value=0).sort_values(ascending=False)
    print(combined.head(10).to_string())

    print("\n=== BIGGEST IMPLIED MOMIOS AMONG COMEBACKS (2026) ===")
    big = df26[df26["was_comeback"]].nlargest(10, "implied_moneyline_worst")[
        ["game_date","away_team_abbrev","home_team_abbrev","away_score","home_score",
         "max_deficit_winner","min_wp_winner","implied_moneyline_worst","home_win"]
    ]
    print(big.to_string())


if __name__ == "__main__":
    main()
