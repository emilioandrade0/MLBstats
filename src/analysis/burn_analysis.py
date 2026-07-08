"""Burn analysis — detect when a team "metió toda la carne al asador" yesterday.

Composite signal from yesterday's game per team:
  - bullpen_ip vs team's own avg (relievers IP burned)
  - extra innings flag (>9 innings)
  - long-duration game (>210 min)
  - many bullpen pitchers used (>=4)
  - closer used in a close game (save situation)
  - comeback game (large deficit overcome)

Burn score = weighted sum of z-scored / boolean signals, scaled to 0-100.

Then we measure the "hangover" effect: when burn_score crosses a threshold,
how does the team perform the NEXT game vs their baseline win rate?

Output: data/processed/burn_analysis.parquet
  game_pk        — today's game (the "next day" after a burn)
  team           — team that played yesterday
  prev_game_pk   — yesterday's game
  prev_game_date
  burn_score     — 0-100, higher = more burnt yesterday
  bullpen_ip_yest, extra_innings, long_game, pitchers_used, used_closer, comeback_yest
  won_today      — did they win today (NaN if game hasn't been played)

Run:
  python -m src.analysis.burn_analysis
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "burn_analysis.parquet"


def _parse_ip(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    s = str(value)
    if "." in s:
        try:
            whole, frac = s.split(".", 1)
            return float(whole) + int(frac[0]) / 3.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _build_team_timeline() -> pd.DataFrame:
    """Per (team, game_date) build burn signals from yesterday's game."""
    games = pd.read_parquet(PROCESSED / "games.parquet")
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_team_abbrev", "away_team_abbrev"]).copy()
    games["home_win"] = np.where(
        games["home_score"].notna() & games["away_score"].notna(),
        (games["home_score"] > games["away_score"]).astype(int),
        np.nan,
    )

    # Bullpen IP per game/side
    pb = pd.read_parquet(
        PROCESSED / "player_box.parquet",
        columns=["game_pk", "side", "player_id", "pit_inningsPitched", "appeared_pitching"],
    )
    pb = pb[pb["appeared_pitching"] == True].copy()
    pb["ip_dec"] = pb["pit_inningsPitched"].map(_parse_ip)

    # Starter per (game_pk, side)
    pitches = pd.read_parquet(
        PROCESSED / "pitches.parquet",
        columns=["game_pk", "inning_topbot", "pitcher", "at_bat_number"],
    )
    pitches["side"] = np.where(
        pitches["inning_topbot"].astype(str).str.startswith("T"), "home", "away"
    )
    idx = pitches.groupby(["game_pk", "side"])["at_bat_number"].idxmin()
    starters = pitches.loc[idx, ["game_pk", "side", "pitcher"]].rename(
        columns={"pitcher": "starter_id"}
    )
    pb["player_id"] = pb["player_id"].astype("Int64")
    starters["starter_id"] = starters["starter_id"].astype("Int64")
    pb = pb.merge(starters, on=["game_pk", "side"], how="left")
    pb["is_starter"] = pb["player_id"] == pb["starter_id"]

    bullpen = (
        pb[~pb["is_starter"]]
        .groupby(["game_pk", "side"])
        .agg(bullpen_ip=("ip_dec", "sum"), bullpen_apps=("player_id", "nunique"))
        .reset_index()
    )

    # Comeback flag from comeback_analysis if available
    cb_path = PROCESSED / "comeback_analysis.parquet"
    comeback_map = {}
    if cb_path.exists():
        cb = pd.read_parquet(cb_path)
        if "is_comeback" in cb.columns and "winner_team" in cb.columns:
            sub = cb[cb["is_comeback"] == True]
            comeback_map = dict(zip(sub["game_pk"], sub["winner_team"]))

    # Per-team rows for each game (one row per team that played)
    rows = []
    for _, g in games.iterrows():
        pk = g["game_pk"]
        margin = abs(g["home_score"] - g["away_score"]) if pd.notna(g["home_score"]) else None
        save_used = pd.notna(g.get("save_pitcher_id"))
        extra = bool(g.get("final_innings", 9) and g["final_innings"] > 9)
        long_g = bool(g.get("duration_minutes", 0) and g["duration_minutes"] > 210)
        cb_winner = comeback_map.get(pk)

        for side, team in [("home", g["home_team_abbrev"]), ("away", g["away_team_abbrev"])]:
            bp = bullpen[(bullpen["game_pk"] == pk) & (bullpen["side"] == side)]
            bp_ip   = float(bp["bullpen_ip"].iloc[0])    if len(bp) else 0.0
            bp_apps = float(bp["bullpen_apps"].iloc[0])  if len(bp) else 0.0
            won = None
            if pd.notna(g.get("home_win")):
                won = int(g["home_win"] == (1 if side == "home" else 0))
            rows.append({
                "team": team,
                "game_pk": pk,
                "game_date": g["game_date"],
                "side": side,
                "bullpen_ip": bp_ip,
                "bullpen_apps": bp_apps,
                "extra_innings": extra,
                "long_game": long_g,
                # closer used in close game (won or lost by <=2 with a save situation)
                "used_closer_close": bool(save_used and margin is not None and margin <= 2),
                "comeback_won": bool(cb_winner == team),
                "won": won,
            })

    tdf = pd.DataFrame(rows)
    tdf = tdf.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)
    return tdf


def _compute_burn_score(tdf: pd.DataFrame) -> pd.DataFrame:
    """Burn score from yesterday's signals per team. Score in 0-100."""
    grp = tdf.groupby("team", sort=False)

    # Team-rolling baseline for bullpen IP (last 30 games)
    tdf["bp_ip_team_avg"] = grp["bullpen_ip"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=5).mean()
    )
    tdf["bp_ip_team_std"] = grp["bullpen_ip"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=5).std()
    ).replace(0, np.nan)

    # Yesterday's values (shift by 1 game per team)
    cols_yest = ["bullpen_ip", "bullpen_apps", "extra_innings", "long_game",
                 "used_closer_close", "comeback_won"]
    for c in cols_yest:
        tdf[f"{c}_yest"] = grp[c].shift(1)

    # Standardized bullpen burn: z-score vs team avg
    tdf["bp_z"] = (
        (tdf["bullpen_ip_yest"] - tdf["bp_ip_team_avg"]) / tdf["bp_ip_team_std"]
    ).clip(-2, 4)
    tdf["bp_z"] = tdf["bp_z"].fillna(0)

    # Components in 0-1 range
    c_bullpen = (tdf["bp_z"].clip(0, 3) / 3.0)                       # 0-1
    c_apps    = (tdf["bullpen_apps_yest"].clip(0, 6) / 6.0).fillna(0)  # 0-1
    c_extra   = tdf["extra_innings_yest"].fillna(False).astype(float)
    c_long    = tdf["long_game_yest"].fillna(False).astype(float)
    c_closer  = tdf["used_closer_close_yest"].fillna(False).astype(float)
    c_comeback = tdf["comeback_won_yest"].fillna(False).astype(float)

    # Weighted composite — bullpen burn weighed heaviest
    weights = {
        "bullpen":  0.30,
        "apps":     0.20,
        "extra":    0.20,
        "long":     0.10,
        "closer":   0.10,
        "comeback": 0.10,
    }
    score = (
        weights["bullpen"]  * c_bullpen
        + weights["apps"]   * c_apps
        + weights["extra"]  * c_extra
        + weights["long"]   * c_long
        + weights["closer"] * c_closer
        + weights["comeback"]* c_comeback
    )
    tdf["burn_score"] = (score * 100).clip(0, 100).round(1)

    # Did "today" win? (the current row's `won`)
    tdf["won_today"] = tdf["won"]
    # Days since yesterday's game (for filtering)
    tdf["prev_game_date"] = grp["game_date"].shift(1)
    tdf["prev_game_pk"]   = grp["game_pk"].shift(1)
    tdf["days_rest"] = (tdf["game_date"] - tdf["prev_game_date"]).dt.days
    return tdf


def main():
    print("building team timeline...")
    tdf = _build_team_timeline()
    print(f"  {len(tdf):,} (team, game) rows")
    print()

    print("computing burn scores...")
    tdf = _compute_burn_score(tdf)
    # Keep only rows where there IS a previous game (so burn_score is meaningful)
    out_cols = [
        "game_pk", "team", "game_date",
        "prev_game_pk", "prev_game_date", "days_rest",
        "burn_score",
        "bullpen_ip_yest", "bullpen_apps_yest", "extra_innings_yest",
        "long_game_yest", "used_closer_close_yest", "comeback_won_yest",
        "won_today",
    ]
    out = tdf.dropna(subset=["prev_game_pk"])[out_cols].copy()
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows)")
    print()

    # ── Diagnostics: hangover effect ──────────────────────────────────────────
    played = out.dropna(subset=["won_today"]).copy()
    played["won_today"] = played["won_today"].astype(int)
    overall_wr = played["won_today"].mean()
    print(f"overall next-game WR (baseline):   {overall_wr:.4f}  N={len(played):,}")
    print()

    print("hangover effect by burn_score bucket (next-day WR):")
    print(f"  {'bucket':>12s} {'N':>7s} {'WR':>7s} {'vs base':>9s}")
    for lo, hi, label in [
        (0, 20, "0-20 (low)"),
        (20, 35, "20-35"),
        (35, 50, "35-50"),
        (50, 65, "50-65"),
        (65, 80, "65-80"),
        (80, 101, "80+ (BURN)"),
    ]:
        sub = played[(played["burn_score"] >= lo) & (played["burn_score"] < hi)]
        if len(sub):
            wr = sub["won_today"].mean()
            delta = wr - overall_wr
            print(f"  {label:>12s} {len(sub):>7,} {wr:>7.4f} {delta:>+9.4f}")
    print()

    # Also stratify by days_rest=0 (back-to-back games, biggest hangover risk)
    b2b = played[played["days_rest"] == 1]
    print(f"back-to-back (days_rest=1) overall: {b2b['won_today'].mean():.4f}  N={len(b2b):,}")
    for lo, hi, label in [(0, 35, "low (<35)"), (35, 65, "mid"), (65, 101, "high (>=65)")]:
        sub = b2b[(b2b["burn_score"] >= lo) & (b2b["burn_score"] < hi)]
        if len(sub):
            print(f"  burn {label:>12s}: WR={sub['won_today'].mean():.4f}  N={len(sub):,}")
    print()

    # Top burns by team (most cumulative burn over season)
    cur_season = played[played["game_date"].dt.year == played["game_date"].dt.year.max()]
    if len(cur_season):
        top = (
            cur_season.groupby("team")
            .agg(burn_avg=("burn_score", "mean"), n=("burn_score", "size"))
            .sort_values("burn_avg", ascending=False)
            .head(10)
        )
        print("most-burned teams this season (avg burn score):")
        print(top.round(2).to_string())

    return OUT


if __name__ == "__main__":
    main()
