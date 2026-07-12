"""Audit por override — cual esta perdiendo señal en la ventana reciente.

Uso:
    python analysis/override_audit.py                  # 30d, 60d, 2026 completo
    python analysis/override_audit.py --window 14      # ventana custom
    python analysis/override_audit.py --end 2026-07-01 # fecha final custom

Filosofia (disciplina del proyecto):
- Si un override tiene net negativo en 2+ ventanas consecutivas (30d Y 60d),
  se considera para retiro.
- Muestra chica (<3 flips en 30d) = esperar mas data.
- Positivo consistente = mantener.

Ejecutar mensual (o tras cada retrain) para monitorear regime drift dentro
del año. Los overrides que dejan de funcionar aquí se retiran del pipeline
en src/serve/api.py antes de que dañen mas picks.
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

# Ejecutar desde raiz del proyecto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.serve.api import (
    _winner_threshold,
    _h2h_bias, _pitcher_bias,
    _home_cold_streak_bias, _away_hot_streak_bias,
    _home_blowout_momentum_bias, _away_cold_streak_bias,
    _weather_extreme_bias, _home_team_calibration_bias,
    _home_band_calibration_bias,
    _interleague_nl_bias, _umpire_market_bias,
    _circadian_extreme_bias, _travel_resilience_bias,
)


def load_data():
    """Carga train + merges de features necesarios para los overrides."""
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

    gm = pd.read_parquet(ROOT / "data" / "processed" / "games.parquet")[
        ["game_pk", "day_night"]
    ]
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
    return tr


# Lista de overrides que aportan bias sumable a p_home.
# Excluye traps (team/day/night/pickday/pickmonth) porque son flip categorico.
OVERRIDES = [
    ("h2h",              lambda r, p, c: _h2h_bias(r["home_team_abbrev"], r["away_team_abbrev"], int(r["season"]))),
    ("pitcher",          lambda r, p, c: _pitcher_bias(r.get("probable_home_pitcher_id"), r.get("probable_away_pitcher_id"), int(r["season"]))),
    ("home_cold_streak", lambda r, p, c: _home_cold_streak_bias(c, r)),
    ("away_hot_streak",  lambda r, p, c: _away_hot_streak_bias(c, r)),
    ("blowout_momentum", lambda r, p, c: _home_blowout_momentum_bias(c, r)),
    ("away_cold_streak", lambda r, p, c: _away_cold_streak_bias(c, r)),
    ("weather_extreme",  lambda r, p, c: _weather_extreme_bias(r)),
    ("stl_calibration",  lambda r, p, c: _home_team_calibration_bias(r, p)),
    ("home_band_calib",  lambda r, p, c: _home_band_calibration_bias(r, p)),
    ("interleague",      lambda r, p, c: _interleague_nl_bias(r, p)),
    ("umpire_market",    lambda r, p, c: _umpire_market_bias(r)),
    ("circadian",        lambda r, p, c: _circadian_extreme_bias(r, p)),
    ("travel_resil",     lambda r, p, c: _travel_resilience_bias(r)),
]


def audit_override(tr, c, name, bias_fn, windows):
    """Para cada juego en tr, aplicar bias_fn. Si el bias flippea el pick raw,
    contar como win o loss segun el resultado real.
    Retorna dict con {ventana_nombre: {flips, wins, losses}}."""
    results = {w[0]: {"flips": 0, "wins": 0, "losses": 0} for w in windows}
    for idx in tr[tr["season"] == 2026].index:
        r = tr.loc[idx]
        p = r["p_home"]
        if not np.isfinite(p):
            continue
        try:
            b_val = bias_fn(r, p, c)
        except Exception:
            continue
        if b_val == 0:
            continue
        # Pick sin/con bias
        p_wo = p
        p_w = float(np.clip(p + b_val, 0.001, 0.999))
        ht = r["home_team_abbrev"]
        at = r["away_team_abbrev"]
        mp = r.get("market_p_home", float("nan"))
        season = int(r["season"])
        thr_wo = _winner_threshold(mp, p_wo, ht, at, season)
        thr_w = _winner_threshold(mp, p_w, ht, at, season)
        pick_wo = bool(p_wo >= thr_wo)
        pick_w = bool(p_w >= thr_w)
        if pick_wo == pick_w:
            continue
        # Flip real
        hw = bool(r["home_won"])
        won_with = pick_w == hw
        won_without = pick_wo == hw
        d = r["game_date"].date()
        for w_name, w_start in windows:
            if d >= w_start:
                results[w_name]["flips"] += 1
                if won_with and not won_without:
                    results[w_name]["wins"] += 1
                elif won_without and not won_with:
                    results[w_name]["losses"] += 1
    return results


def fmt_cell(d):
    if d["flips"] == 0:
        return "0/0 (—)"
    net = d["wins"] - d["losses"]
    return f"{d['wins']}/{d['flips']} ({net:+d})"


def verdict(d30, d60):
    """Verdicto basado en ventanas 30d y 60d."""
    if d30["flips"] < 3:
        return "muestra chica"
    net_30 = d30["wins"] - d30["losses"]
    net_60 = d60["wins"] - d60["losses"] if d60["flips"] >= 3 else None
    if net_30 < 0 and (net_60 is None or net_60 <= 0):
        return "🔴 PIERDE — CONSIDERA RETIRAR"
    if net_30 > 3:
        return "🟢 gana"
    return "🟡 neutral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30, help="Ventana corta en dias (default 30)")
    ap.add_argument("--end", type=str, default=None, help="Fecha final YYYY-MM-DD (default today)")
    args = ap.parse_args()

    end_date = date.fromisoformat(args.end) if args.end else date.today()
    short_start = end_date - timedelta(days=args.window)
    medium_start = end_date - timedelta(days=args.window * 2)
    year_start = date(end_date.year, 3, 1)

    windows = [
        (f"{args.window}d", short_start),
        (f"{args.window*2}d", medium_start),
        (f"{end_date.year}", year_start),
    ]

    print(f"═══ Audit por override — flippeos y net acc ═══")
    print(f"  Fecha final: {end_date}")
    print(f"  Ventanas: {windows[0][0]} ({short_start}), {windows[1][0]} ({medium_start}), {windows[2][0]} ({year_start})")
    print()

    print("  Cargando data...")
    tr = load_data()
    c = {"train": tr}
    print(f"  N total 2026: {(tr['season']==2026).sum()}")
    print()

    header = f"  {'override':<20}  {windows[0][0]:>13}  {windows[1][0]:>13}  {windows[2][0]:>15}  verdicto"
    print(header)
    print("  " + "-" * (len(header) - 2))

    retire_candidates = []
    for name, fn in OVERRIDES:
        r = audit_override(tr, c, name, fn, windows)
        v = verdict(r[windows[0][0]], r[windows[1][0]])
        cells = [fmt_cell(r[w[0]]) for w in windows]
        print(f"  {name:<20}  {cells[0]:>13}  {cells[1]:>13}  {cells[2]:>15}  {v}")
        if "PIERDE" in v:
            retire_candidates.append(name)

    print()
    print("═══ Recomendacion ═══")
    if retire_candidates:
        print(f"  🔴 Candidatos a retirar (audit falla en 30d y 60d):")
        for name in retire_candidates:
            print(f"     - {name}")
        print()
        print("  Para retirar: setear el bias a 0.0 en src/serve/api.py (linea del")
        print("  card builder y _final_pick_home). Correr backtest para verificar")
        print("  que no cae acc en las temporadas anteriores.")
    else:
        print("  🟢 Todos los overrides mantienen valor. Ningun retiro recomendado.")
    print()
    print("═══ Interpretacion ═══")
    print("  N/M = wins/flips totales en esa ventana")
    print("  (+X) = net (wins - losses)")
    print("  wins = flippeos donde el override CAMBIO al lado correcto")
    print("  losses = flippeos donde el override CAMBIO al lado equivocado")


if __name__ == "__main__":
    main()
