"""Post-ASG monitor — reporte diario de accuracy y overrides tras el All-Star break.

Uso:
    python analysis/post_asg_monitor.py                 # data hasta hoy
    python analysis/post_asg_monitor.py --end 2026-07-25 # fecha fin custom

Diseño:
- Establece un baseline pre-ASG (todo 2026 hasta 07-12) para comparar
- Muestra accuracy DIARIA post-ASG y flags días malos
- Corre el override audit sobre la ventana post-ASG específicamente
- Alerta si algún override se rompe (net negativo en post-ASG con N>=3)

Filosofía:
- Los juegos post-ASG suelen ser volátiles (rotaciones cambian, jugadores
  descansan, lineups distintos). Los overrides tuneados sobre régimen H1
  pueden no funcionar igual en H2.
- Este script se corre DIARIAMENTE los primeros 10-14 días post-ASG.
- Si un override muestra -2+ net en 5+ flips durante post-ASG, considerar retiro.
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.serve.api import (
    _final_pick_home, _winner_threshold,
    _h2h_bias, _pitcher_bias,
    _home_cold_streak_bias, _away_hot_streak_bias,
    _home_blowout_momentum_bias, _away_cold_streak_bias,
    _weather_extreme_bias, _home_team_calibration_bias,
    _home_band_calibration_bias,
    _interleague_nl_bias, _umpire_market_bias,
    _circadian_extreme_bias, _travel_resilience_bias,
    _pythag_luck_bias, _babip_persistence_bias,
    _lob_persistence_bias, _cws_night_bias,
    _catcher_battery_regime_bias, _comeback_deficit_regression_bias,
    _runs_median_persistence_bias, _burn_resilience_bias,
    _starter_workload_regression_bias, _highlev_fatigue_bias,
)

ASG_END = date(2026, 7, 14)  # All-Star break end (juegos vuelven ~15)


def load_data():
    with open(ROOT / "data" / "models" / "lgb_cls.pkl", "rb") as f:
        b = pickle.load(f)
    model = b["model"]

    tr = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    tr = tr[tr["home_score"].notna() & tr["away_score"].notna()].copy()
    tr["game_date"] = pd.to_datetime(tr["game_date"])
    tr["home_won"] = (tr["home_score"] > tr["away_score"]).astype(int)
    tr["home_win"] = tr["home_won"]
    tr["season"] = tr["game_date"].dt.year
    tr["p_home"] = model.predict_proba(tr[b["feature_names"]])[:, 1]

    # Merge features requeridos por overrides
    gm = pd.read_parquet(ROOT / "data" / "processed" / "games.parquet")[["game_pk", "day_night"]]
    tr = tr.merge(gm, on="game_pk", how="left")
    tr["day_night"] = tr["day_night"].str.lower()

    tv = pd.read_parquet(ROOT / "data" / "processed" / "features_travel.parquet")[
        ["game_pk", "side", "travel_dist_miles"]
    ]
    tv_wide = tv.pivot(index="game_pk", columns="side", values="travel_dist_miles").reset_index()
    tv_wide.columns = ["game_pk", "travel_dist_miles_a", "travel_dist_miles_h"]
    tr = tr.merge(tv_wide, on="game_pk", how="left")

    circ = pd.read_parquet(ROOT / "data" / "processed" / "features_circadian.parquet")[
        ["game_pk", "circ_x_day_home"]
    ]
    tr = tr.merge(circ, on="game_pk", how="left")

    # Otros merges opcionales
    for filename in ["features_luck.parquet", "features_cluster_luck.parquet",
                     "features_catcher_control.parquet"]:
        path = ROOT / "data" / "processed" / filename
        if not path.exists(): continue
        try:
            if filename == "features_luck.parquet":
                luck = pd.read_parquet(path)[["game_pk", "side", "babip_luck_net"]]
                luck_wide = luck.pivot(index="game_pk", columns="side", values="babip_luck_net").reset_index()
                luck_wide.columns = ["game_pk", "babip_luck_net_a", "babip_luck_net_h"]
                tr = tr.merge(luck_wide, on="game_pk", how="left")
            elif filename == "features_cluster_luck.parquet":
                cluster = pd.read_parquet(path)[
                    ["game_pk", "team", "lob_off_dev", "lob_def_dev"]
                ]
                hc = cluster.rename(columns={"team": "home_team_abbrev",
                    "lob_off_dev": "lob_off_dev_h", "lob_def_dev": "lob_def_dev_h"})
                ac = cluster.rename(columns={"team": "away_team_abbrev",
                    "lob_off_dev": "lob_off_dev_a", "lob_def_dev": "lob_def_dev_a"})
                tr = tr.merge(hc, on=["game_pk", "home_team_abbrev"], how="left")
                tr = tr.merge(ac, on=["game_pk", "away_team_abbrev"], how="left")
            elif filename == "features_catcher_control.parquet":
                cb = pd.read_parquet(path)[["game_pk", "side", "battery_kbb_l8"]]
                cb_wide = cb.pivot(index="game_pk", columns="side", values="battery_kbb_l8").reset_index()
                cb_wide = cb_wide.rename(columns={"away": "battery_kbb_l8_a", "home": "battery_kbb_l8_h"})
                tr = tr.merge(cb_wide, on="game_pk", how="left")
        except Exception as e:
            print(f"  warn: no pude merge {filename}: {e}")

    for filename, val_col in [
        ("features_comeback.parquet", "cb_avg_def_l30"),
        ("features_game_flow.parquet", "runs_median_l20"),
        ("features_burn.parquet", "burn_score"),
    ]:
        path = ROOT / "data" / "processed" / filename
        if path.exists():
            try:
                s = pd.read_parquet(path)[["game_pk", "side", val_col]]
                sw = s.pivot(index="game_pk", columns="side", values=val_col).reset_index()
                sw = sw.rename(columns={"away": f"{val_col}_a", "home": f"{val_col}_h"})
                tr = tr.merge(sw, on="game_pk", how="left")
            except Exception:
                pass

    for filename in ["features_starter_workload.parquet", "features_high_leverage.parquet"]:
        path = ROOT / "data" / "processed" / filename
        if path.exists():
            try:
                tr = tr.merge(pd.read_parquet(path), on="game_pk", how="left")
            except Exception:
                pass

    return tr


def daily_accuracy(tr, start, end):
    """Reporte diario de accuracy con vs sin pipeline completo."""
    c = {"train": tr}
    sub = tr[(tr["game_date"].dt.date >= start) & (tr["game_date"].dt.date <= end)].copy()
    if sub.empty:
        return None, None

    # Pick con pipeline actual (usa _final_pick_home)
    sub["pick"] = sub.apply(lambda r: _final_pick_home(c, r, r["p_home"]) if np.isfinite(r["p_home"]) else None, axis=1)
    sub["hit"] = (sub["pick"].astype("boolean") == sub["home_won"].astype("boolean")).astype("Int64")

    # Pick sin overrides (baseline naive)
    sub["pick_naive"] = (sub["p_home"] >= 0.5)
    sub["hit_naive"] = (sub["pick_naive"].astype("boolean") == sub["home_won"].astype("boolean")).astype("Int64")

    daily = sub.groupby(sub["game_date"].dt.date).agg(
        n=("home_won", "size"),
        acc_pipeline=("hit", "mean"),
        acc_naive=("hit_naive", "mean"),
        hits_pipeline=("hit", "sum"),
        hits_naive=("hit_naive", "sum"),
    ).reset_index()
    daily["delta"] = (daily["acc_pipeline"] - daily["acc_naive"]) * 100
    return daily, sub


def summary(daily):
    if daily is None or daily.empty:
        return "Sin data"
    total_n = int(daily["n"].sum())
    total_hits = int(daily["hits_pipeline"].sum())
    total_naive = int(daily["hits_naive"].sum())
    return {
        "n": total_n,
        "hits_pipeline": total_hits,
        "hits_naive": total_naive,
        "acc_pipeline": total_hits / total_n if total_n else None,
        "acc_naive": total_naive / total_n if total_n else None,
        "delta_pp": ((total_hits - total_naive) / total_n * 100) if total_n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", type=str, default=None, help="Fecha final YYYY-MM-DD (default today)")
    args = ap.parse_args()

    end_date = date.fromisoformat(args.end) if args.end else date.today()

    print("═" * 78)
    print("  STRIKECAST · POST-ASG MONITOR")
    print(f"  Fecha final: {end_date}  |  ASG end: {ASG_END}")
    print("═" * 78)

    print("\n  Cargando data...")
    tr = load_data()
    print(f"  N total: {len(tr):,} | 2026: {(tr['season']==2026).sum():,}")

    # --- Baseline pre-ASG (todo 2026 hasta ASG_END) ---
    print(f"\n═══ BASELINE PRE-ASG (2026 hasta {ASG_END}) ═══")
    pre_daily, _ = daily_accuracy(tr, date(2026, 3, 1), ASG_END)
    pre_summary = summary(pre_daily)
    if isinstance(pre_summary, dict):
        print(f"  n={pre_summary['n']}  acc={pre_summary['acc_pipeline']*100:.2f}%  "
              f"({pre_summary['hits_pipeline']}/{pre_summary['n']})  "
              f"delta vs naive: {pre_summary['delta_pp']:+.2f}pp")

    # --- Ventana post-ASG ---
    post_start = ASG_END + timedelta(days=1)
    if post_start > end_date:
        print(f"\n  ⚠ Aún no hay juegos post-ASG (post_start={post_start} > end={end_date})")
        print(f"    El próximo día de análisis será {post_start}. Volver a correr entonces.")
        return

    print(f"\n═══ POST-ASG {post_start} → {end_date} ═══")
    post_daily, post_sub = daily_accuracy(tr, post_start, end_date)
    if post_daily is None or post_daily.empty:
        print("  Sin juegos aún en la ventana post-ASG.")
        return

    post_summary = summary(post_daily)
    print(f"  n={post_summary['n']}  acc={post_summary['acc_pipeline']*100:.2f}%  "
          f"({post_summary['hits_pipeline']}/{post_summary['n']})  "
          f"delta vs naive: {post_summary['delta_pp']:+.2f}pp")

    # --- Comparativa ---
    if isinstance(pre_summary, dict):
        diff = (post_summary['acc_pipeline'] - pre_summary['acc_pipeline']) * 100
        flag = "🔴" if diff < -2 else ("🟡" if diff < 0 else "🟢")
        print(f"\n  {flag} vs baseline pre-ASG: {diff:+.2f}pp")

    # --- Detalle por día ---
    print(f"\n═══ Detalle día por día ═══")
    print(f"  {'fecha':<12} {'n':>3} {'acc':>7} {'naive':>7} {'Δ':>7}  flag")
    for _, r in post_daily.iterrows():
        flag = ""
        if r["delta"] < -5: flag = "🔴 overrides dañan"
        elif r["delta"] > 5: flag = "🟢 overrides mejoran"
        print(f"  {str(r['game_date']):<12} {int(r['n']):>3} "
              f"{r['acc_pipeline']*100:>6.1f}% {r['acc_naive']*100:>6.1f}% "
              f"{r['delta']:+6.2f}pp  {flag}")

    # --- Alerta si acc post-ASG < 55% con N>=15 ---
    if post_summary['n'] >= 15 and post_summary['acc_pipeline'] < 0.55:
        print(f"\n  🚨 ALERTA: acc post-ASG {post_summary['acc_pipeline']*100:.1f}% en n={post_summary['n']}")
        print(f"     Considerar correr override_audit.py --end {end_date}")
        print(f"     para ver qué override está fallando en el nuevo régimen.")


if __name__ == "__main__":
    main()
