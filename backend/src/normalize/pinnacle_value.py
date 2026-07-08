"""Pinnacle sharp-value detector.

Pinnacle is the sharpest book in the world (~2% vig, the closing reference for
the whole market). Its de-vigged moneyline ≈ true probability. Any SOFT book
offering a better price than Pinnacle's fair line is +EV by definition — this
is how professional bettors actually profit, no predictive model required.

This module:
  1. Loads the latest odds_api snapshot (Pinnacle + soft books).
  2. Maps full team names → our abbreviations → game_pk (today's slate).
  3. Computes Pinnacle de-vig fair prob per game.
  4. Finds the best soft-book price per side and flags sharp value.

Output: data/processed/pinnacle_value.parquet
  game_pk | pin_fair_home | pin_fair_away
          | best_home_ml | best_home_book | best_away_ml | best_away_book
          | sharp_edge_home | sharp_edge_away  (best soft devig - pinnacle fair... )

Run:
  python -m src.normalize.pinnacle_value
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PROCESSED

TEAM_MAP = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "ATH", "Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

SOFT_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "williamhill_us"}


def _imp(ml):
    if pd.isna(ml) or abs(ml) < 100:
        return np.nan
    ml = float(ml)
    return (abs(ml) / (abs(ml) + 100)) if ml < 0 else (100 / (ml + 100))


def _devig_pair(p_home_raw, p_away_raw):
    s = p_home_raw + p_away_raw
    if not np.isfinite(s) or s <= 0:
        return np.nan, np.nan
    return p_home_raw / s, p_away_raw / s


def build() -> Path | None:
    pin_path = PROCESSED / "odds_pinnacle.parquet"
    if not pin_path.exists():
        print("no odds_pinnacle.parquet — run src.ingest_odds_api first")
        return None
    df = pd.read_parquet(pin_path)
    df["away_abbr"] = df["away_team"].map(TEAM_MAP)
    df["home_abbr"] = df["home_team"].map(TEAM_MAP)
    df = df.dropna(subset=["away_abbr", "home_abbr"])
    # PRE-GAME ONLY: live games have Pinnacle showing live odds while soft books
    # may carry stale pre-game lines, manufacturing fake 20-40pp "edges".
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    if "--all" not in __import__("sys").argv:
        now = pd.Timestamp.now(tz="UTC")
        pre = df[df["commence_time"] > now + pd.Timedelta(minutes=2)]
        if not pre.empty:
            df = pre

    # Map each Pinnacle EVENT → game_pk by teams AND start-time proximity.
    # Series (same teams on consecutive days) would collide on teams alone, so
    # we disambiguate with commence_time vs the game's scheduled first pitch.
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "first_pitch_utc",
                 "home_team_abbrev", "away_team_abbrev"],
    )
    games["first_pitch_utc"] = pd.to_datetime(games["first_pitch_utc"], utc=True, errors="coerce")

    def _match_game(away, home, commence) -> int | None:
        cand = games[(games["away_team_abbrev"] == away) &
                     (games["home_team_abbrev"] == home) &
                     games["first_pitch_utc"].notna()]
        if cand.empty:
            return None
        if pd.notna(commence):
            diff = (cand["first_pitch_utc"] - commence).abs()
            best = cand.loc[diff.idxmin()]
            # Only accept if within 6h of the Pinnacle event's start
            if diff.min() <= pd.Timedelta(hours=6):
                return int(best["game_pk"])
            return None
        return int(cand.iloc[0]["game_pk"])

    rows = []
    for (away, home), grp in df.groupby(["away_abbr", "home_abbr"]):
        commence = grp["commence_time"].iloc[0]
        game_pk = _match_game(away, home, commence)
        if game_pk is None:
            continue

        pin = grp[grp["bookmaker"] == "pinnacle"]
        if pin.empty:
            continue
        p = pin.iloc[0]
        ph_raw, pa_raw = _imp(p["home_ml"]), _imp(p["away_ml"])
        fair_h, fair_a = _devig_pair(ph_raw, pa_raw)
        if not np.isfinite(fair_h):
            continue

        # Best soft-book price per side (max American = best for bettor)
        soft = grp[grp["bookmaker"].isin(SOFT_BOOKS)].copy()
        soft = soft[soft["home_ml"].abs() >= 100]
        if soft.empty:
            best_h_ml = best_h_book = best_a_ml = best_a_book = None
        else:
            hi = soft.loc[soft["home_ml"].idxmax()]
            ai = soft.loc[soft["away_ml"].idxmax()]
            best_h_ml, best_h_book = float(hi["home_ml"]), hi["bookmaker"]
            best_a_ml, best_a_book = float(ai["away_ml"]), ai["bookmaker"]

        # Sharp edge = soft book's RAW implied (with its vig) vs Pinnacle fair.
        # If the best soft price implies a LOWER prob than Pinnacle's fair, the
        # book is paying you more than fair → +EV.
        edge_h = fair_h - _imp(best_h_ml) if best_h_ml is not None else np.nan
        edge_a = fair_a - _imp(best_a_ml) if best_a_ml is not None else np.nan
        # Sanity: Pinnacle is sharp — true value rarely exceeds ~7pp. Anything
        # above 10pp is almost always stale/mismatched data, not real value.
        if np.isfinite(edge_h) and abs(edge_h) > 0.10:
            edge_h = np.nan
        if np.isfinite(edge_a) and abs(edge_a) > 0.10:
            edge_a = np.nan

        rows.append({
            "game_pk": game_pk,
            "pin_fair_home": round(fair_h, 4), "pin_fair_away": round(fair_a, 4),
            "pin_home_ml": float(p["home_ml"]), "pin_away_ml": float(p["away_ml"]),
            "best_home_ml": best_h_ml, "best_home_book": best_h_book,
            "best_away_ml": best_a_ml, "best_away_book": best_a_book,
            "sharp_edge_home_pp": round(edge_h * 100, 2) if np.isfinite(edge_h) else None,
            "sharp_edge_away_pp": round(edge_a * 100, 2) if np.isfinite(edge_a) else None,
        })

    out = pd.DataFrame(rows)
    out_path = PROCESSED / "pinnacle_value.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  ({len(out)} games matched to Pinnacle)")
    if not out.empty:
        show = out[["game_pk", "pin_fair_home", "best_home_book", "sharp_edge_home_pp",
                    "best_away_book", "sharp_edge_away_pp"]]
        print(show.to_string(index=False))
        # Flag the sharp value plays
        val = out[(out["sharp_edge_home_pp"].fillna(-99) >= 1) |
                  (out["sharp_edge_away_pp"].fillna(-99) >= 1)]
        print(f"\n{len(val)} games with sharp value (soft book pays >1pp over Pinnacle fair)")
    return out_path


if __name__ == "__main__":
    build()
