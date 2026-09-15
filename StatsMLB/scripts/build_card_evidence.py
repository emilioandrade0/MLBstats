"""Publish validated, compact evidence profiles for StatsMLB game cards.

Profiles are discovered on 2024-25 walk-forward rows and accepted only when
they retain an improvement in 2026.  This deliberately avoids narratives or
small, post-hoc slices in the browser.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "StatsMLB" / "public" / "data" / "card-evidence.json"
MIN_DISCOVERY = 120
MIN_VALIDATION = 60
MIN_DELTA = 0.015


def wilson_lower(wins: int, n: int) -> float:
    if not n:
        return 0.0
    z = 1.96
    p = wins / n
    return (p + z*z/(2*n) - z * math.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)


def main() -> None:
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    wf = pd.read_parquet(ROOT / "data" / "processed" / "walkforward_preds.parquet")
    needed = ["game_pk", "game_date", "win_pct_l30_diff", "run_diff_l10_diff", "days_since_last_game_diff", "market_line_shift_home_pp"]
    frame = wf.merge(train[[c for c in needed if c in train]], on="game_pk", how="inner", suffixes=("", "_train"))
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame = frame.dropna(subset=["p_home", "home_win", "market_p_home"]).copy()
    frame["fav_home"] = frame["p_home"] >= .5
    frame["pick_won"] = frame["fav_home"] == frame["home_win"].astype(bool)
    direction = frame["fav_home"].map({True: 1, False: -1})
    frame["model_market_agree"] = frame["fav_home"] == (frame["market_p_home"] >= .5)
    frame["model_edge_3"] = ((frame["p_home"] - frame["market_p_home"]) * direction) >= .03
    # The training table has a complete, pregame L30 win-rate feature.  We call
    # it "form" here rather than pretending it is the UI's separate L10 stat.
    frame["favorite_form"] = (frame.get("win_pct_l30_diff", 0).fillna(0) * direction) >= .08
    frame["favorite_run_diff"] = (frame.get("run_diff_l10_diff", 0).fillna(0) * direction) >= .35
    frame["favorite_rest"] = (frame.get("days_since_last_game_diff", 0).fillna(0) * direction) >= 1
    # A two-point de-vigged shift is above normal intraday noise (the 2026
    # median absolute shift is below one point). It is evaluated separately
    # and still has to pass the same time-split validation as every other tag.
    frame["market_move_favorite_2"] = (frame.get("market_line_shift_home_pp", 0).fillna(0) * direction) >= 2
    conditions = ["fav_home", "model_market_agree", "model_edge_3", "favorite_form", "favorite_run_diff", "favorite_rest", "market_move_favorite_2"]
    discovery = frame[frame["game_date"].dt.year.le(2025)]
    validation = frame[frame["game_date"].dt.year.eq(2026)]
    base = float(validation["pick_won"].mean())
    profiles = []
    for size in (2, 3):
        for combo in itertools.combinations(conditions, size):
            d = discovery.loc[discovery[list(combo)].all(axis=1), "pick_won"]
            v = validation.loc[validation[list(combo)].all(axis=1), "pick_won"]
            if len(d) < MIN_DISCOVERY or len(v) < MIN_VALIDATION:
                continue
            discovery_rate = float(d.mean())
            rate = float(v.mean())
            wins = int(v.sum())
            delta = rate - base
            if discovery_rate <= float(discovery["pick_won"].mean()) or delta < MIN_DELTA or wilson_lower(wins, len(v)) <= base:
                continue
            profiles.append({"conditions": combo, "sampleSize": len(v), "wins": wins, "winRate": rate, "baselineWinRate": base, "delta": delta, "lower95": wilson_lower(wins, len(v))})
    profiles.sort(key=lambda row: (row["delta"], row["sampleSize"], row["lower95"]), reverse=True)
    payload = {"generatedAt": pd.Timestamp.now("UTC").isoformat(), "method": "Descubrimiento 2024-25; validación walk-forward 2026; perfiles de 2-3 condiciones.", "minimumDiscovery": MIN_DISCOVERY, "minimumValidation": MIN_VALIDATION, "minimumDeltaPoints": MIN_DELTA * 100, "profiles": profiles[:40]}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"card evidence: {len(payload['profiles'])} perfiles validados -> {OUT}")


if __name__ == "__main__":
    main()
