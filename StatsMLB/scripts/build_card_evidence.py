"""Publish per-event signal evidence for StatsMLB game cards.

Each signal is a single, human-readable condition (e.g. "favorite home", "market
disagrees with pick", "favorite short rest").  For every signal we measure how
often the model's pick actually won when that signal was true in 2024-25 and
validate the same rate in 2026 walk-forward.  The frontend then evaluates the
same conditions live for each game and surfaces the top signals -- some of
which will *support* the pick and others will *challenge* it, without inventing
extra text when data is missing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "StatsMLB" / "public" / "data" / "card-evidence.json"
MIN_N = 200


def wilson_lower(wins: int, n: int) -> float:
    if not n:
        return 0.0
    z = 1.96
    p = wins / n
    return (p + z*z/(2*n) - z * math.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)


def build_signal_conditions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    direction = frame["fav_home"].map({True: 1, False: -1})
    win_pct = frame.get("win_pct_l30_diff", 0).fillna(0) * direction
    run_diff = frame.get("run_diff_l10_diff", 0).fillna(0) * direction
    rest = frame.get("days_since_last_game_diff", 0).fillna(0) * direction
    market_shift = frame.get("market_line_shift_home_pp", 0).fillna(0) * direction
    starter_edge = frame.get("starter_xwoba_l15_diff", 0).fillna(0) * direction
    bullpen_load = frame.get("bullpen_ip_l5d_diff", 0).fillna(0) * direction
    elo_edge = frame.get("elo_diff_pre", 0).fillna(0) * direction
    park_run = frame.get("park_run_diff_diff", 0).fillna(0) * direction
    lineup_ops = frame.get("lineup_top4_recent_ops_l15_diff", 0).fillna(0) * direction

    return {
        "fav_home": frame["fav_home"],
        "fav_road": ~frame["fav_home"].astype(bool),
        "market_agrees": frame["fav_home"] == (frame["market_p_home"] >= .5),
        "market_disagrees": frame["fav_home"] != (frame["market_p_home"] >= .5),
        "model_edge_3": ((frame["p_home"] - frame["market_p_home"]) * direction) >= .03,
        "model_edge_5": ((frame["p_home"] - frame["market_p_home"]) * direction) >= .05,
        "market_prices_favorite": ((frame["market_p_home"] - .5) * direction) >= .10,
        "favorite_l30_hot": win_pct >= .08,
        "favorite_l30_cold": win_pct <= -.08,
        "favorite_run_diff_strong": run_diff >= .35,
        "favorite_run_diff_weak": run_diff <= -.35,
        "favorite_rest_advantage": rest >= 1,
        "favorite_short_rest": rest <= -1,
        "market_move_toward_favorite": market_shift >= 2,
        "market_move_against_favorite": market_shift <= -2,
        "favorite_starter_edge": starter_edge <= -.010,
        "favorite_starter_weak": starter_edge >= .010,
        "favorite_bullpen_fresher": bullpen_load <= -1.0,
        "favorite_bullpen_tired": bullpen_load >= 1.0,
        "favorite_elo_edge": elo_edge >= 25,
        "favorite_park_boost": park_run >= .10,
        "favorite_lineup_ops_edge": lineup_ops >= .020,
        "favorite_lineup_ops_deficit": lineup_ops <= -.020,
    }


SIGNAL_META = {
    "fav_home": ("supports", "Favorito local"),
    "fav_road": ("context", "Favorito visitante"),
    "market_agrees": ("supports", "Mercado alineado con el pick"),
    "market_disagrees": ("challenges", "Mercado en contra del pick"),
    "model_edge_3": ("supports", "Modelo con ventaja de 3+ pp vs mercado"),
    "model_edge_5": ("supports", "Modelo con ventaja de 5+ pp vs mercado"),
    "market_prices_favorite": ("context", "Mercado ya premia al favorito"),
    "favorite_l30_hot": ("supports", "Favorito con mejor forma L30"),
    "favorite_l30_cold": ("challenges", "Favorito con peor forma L30"),
    "favorite_run_diff_strong": ("supports", "Diferencial de carreras a favor"),
    "favorite_run_diff_weak": ("challenges", "Diferencial de carreras en contra"),
    "favorite_rest_advantage": ("supports", "Ventaja de descanso"),
    "favorite_short_rest": ("challenges", "Desventaja de descanso"),
    "market_move_toward_favorite": ("supports", "Momio se mueve a favor del pick"),
    "market_move_against_favorite": ("challenges", "Momio se mueve en contra del pick"),
    "favorite_starter_edge": ("supports", "Abridor del favorito con mejor xwOBA L15"),
    "favorite_starter_weak": ("challenges", "Abridor del favorito con peor xwOBA L15"),
    "favorite_bullpen_fresher": ("supports", "Bullpen del favorito más descansado"),
    "favorite_bullpen_tired": ("challenges", "Bullpen del favorito cargado"),
    "favorite_elo_edge": ("supports", "Favorito con ventaja Elo"),
    "favorite_park_boost": ("context", "Parque favorece la ofensiva del favorito"),
    "favorite_lineup_ops_edge": ("supports", "Lineup del favorito con mejor OPS L15"),
    "favorite_lineup_ops_deficit": ("challenges", "Lineup del favorito con OPS L15 en desventaja"),
}


def main() -> None:
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    wf = pd.read_parquet(ROOT / "data" / "processed" / "walkforward_preds.parquet")
    training_cols = [
        "game_pk", "game_date",
        "win_pct_l30_diff", "run_diff_l10_diff", "days_since_last_game_diff",
        "market_line_shift_home_pp", "starter_xwoba_l15_diff", "bullpen_ip_l5d_diff",
        "elo_diff_pre", "park_run_diff_diff", "lineup_top4_recent_ops_l15_diff",
    ]
    have = [c for c in training_cols if c in train.columns]
    frame = wf.merge(train[have], on="game_pk", how="inner", suffixes=("", "_train"))
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame = frame.dropna(subset=["p_home", "home_win", "market_p_home"]).copy()
    frame["fav_home"] = frame["p_home"] >= .5
    frame["pick_won"] = frame["fav_home"] == frame["home_win"].astype(bool)

    discovery = frame[frame["game_date"].dt.year.le(2025)]
    validation = frame[frame["game_date"].dt.year.eq(2026)]
    base_discovery = float(discovery["pick_won"].mean())
    base_validation = float(validation["pick_won"].mean())
    base = float(frame["pick_won"].mean())

    conditions_disc = build_signal_conditions(discovery)
    conditions_val = build_signal_conditions(validation)
    conditions_all = build_signal_conditions(frame)

    signals = []
    for code, meta in SIGNAL_META.items():
        direction_kind, label = meta
        mask_disc = conditions_disc[code].astype(bool)
        mask_val = conditions_val[code].astype(bool)
        mask_all = conditions_all[code].astype(bool)
        n_disc = int(mask_disc.sum())
        n_val = int(mask_val.sum())
        n_all = int(mask_all.sum())
        if n_val < MIN_N:
            continue
        disc_rate = float(discovery.loc[mask_disc, "pick_won"].mean()) if n_disc else 0.0
        val_rate = float(validation.loc[mask_val, "pick_won"].mean())
        val_wins = int(validation.loc[mask_val, "pick_won"].sum())
        all_rate = float(frame.loc[mask_all, "pick_won"].mean())
        all_wins = int(frame.loc[mask_all, "pick_won"].sum())
        delta_disc = disc_rate - base_discovery if n_disc else 0.0
        delta_val = val_rate - base_validation
        delta_all = all_rate - base
        if direction_kind == "supports" and (delta_disc <= 0 or delta_val <= 0):
            continue
        if direction_kind == "challenges" and (delta_disc >= 0 or delta_val >= 0):
            continue
        lower = wilson_lower(all_wins, n_all)
        upper = 2 * all_rate - lower
        signals.append({
            "code": code,
            "label": label,
            "direction": direction_kind,
            "sampleSize": n_all,
            "winRate": all_rate,
            "baselineWinRate": base,
            "delta": delta_all,
            "lower95": lower,
            "upper95": upper,
            "validationSampleSize": n_val,
            "validationWinRate": val_rate,
            "validationDelta": delta_val,
        })

    signals.sort(key=lambda row: abs(row["delta"]), reverse=True)

    payload = {
        "generatedAt": pd.Timestamp.now("UTC").isoformat(),
        "method": "Señales atómicas por evento; validadas en descubrimiento 2024-25 y walk-forward 2026.",
        "baselinePickWinRate": base,
        "minimumSampleSize": MIN_N,
        "signals": signals,
        "profiles": [],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"card evidence: {len(signals)} señales válidas -> {OUT}")


if __name__ == "__main__":
    main()
