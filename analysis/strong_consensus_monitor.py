"""Monitor prospectivo — modo Consenso Fuerte (fase 5B STARTFROMTHEEND).

Regla (validada retrospectivamente 2025-2026H1-H2):
  pick del selector normal + acuerdo con mercado
  + p_pick del modelo >= 0.60
  + mkt_pick_p (fuerza del favorito de mercado del lado elegido) >= 0.60

Histórico walk-forward (validación in-sample):
  Global 2025-2026: 446/649 = 68.72%
  2025:    329/482 = 68.26%
  2026-H1:  59/89  = 66.29%
  2026-H2:  58/78  = 74.36%
  Cobertura: 47.06% de los picks del selector normal

Este script mide la regla SOLO en juegos posteriores al corte de análisis
(2026-07-23) — validación ciega. Es la única evidencia que puede confirmar
o rechazar el 68.72% cuando acumule N suficiente.

Uso:
    python analysis/strong_consensus_monitor.py
    python analysis/strong_consensus_monitor.py --end 2026-08-10
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import pickle
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.serve.api import _final_pick_home, _winner_threshold

# Corte de análisis retrospectivo. Cualquier juego con fecha <= a este día
# ya fue inspeccionado y NO es evidencia ciega.
BLIND_START = date(2026, 7, 24)


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
    return tr


def evaluate(tr, start, end):
    c = {"train": tr}
    sub = tr[(tr["game_date"].dt.date >= start) & (tr["game_date"].dt.date <= end)].copy()
    if sub.empty:
        return None

    rows = []
    for _, r in sub.iterrows():
        p = r["p_home"]
        if not np.isfinite(p):
            continue
        pick_home = _final_pick_home(c, r, p)
        if pick_home is None:
            continue
        pick_p = float(p) if pick_home else 1.0 - float(p)
        mp = r.get("market_p_home")
        if not (mp is not None and np.isfinite(mp)):
            continue
        mkt_pick_p = float(mp) if pick_home else 1.0 - float(mp)
        agree = (float(mp) >= 0.5) == pick_home
        strong = bool(agree and pick_p >= 0.60 and mkt_pick_p >= 0.60)
        won = bool(pick_home) == (r["home_won"] == 1)
        rows.append({
            "date": r["game_date"].date(),
            "away": r["away_team_abbrev"],
            "home": r["home_team_abbrev"],
            "pick": r["home_team_abbrev"] if pick_home else r["away_team_abbrev"],
            "pick_p": pick_p,
            "mkt_pick_p": mkt_pick_p,
            "strong": strong,
            "won": int(won),
        })
    return pd.DataFrame(rows)


def summarize(df, label):
    if df is None or df.empty:
        print(f"  {label}: sin data")
        return
    total = len(df)
    total_hits = int(df["won"].sum())
    sc = df[df["strong"]]
    sc_n = len(sc)
    sc_hits = int(sc["won"].sum())
    print(f"  {label}:")
    print(f"    picks totales:  {total_hits}/{total} = {total_hits/total*100:.2f}%")
    if sc_n:
        pct = sc_hits / sc_n * 100
        flag = "🟢" if pct >= 65 else "🟡" if pct >= 55 else "🔴"
        cov = sc_n / total * 100
        print(f"    consenso fuerte: {sc_hits}/{sc_n} = {pct:.2f}%  {flag}  (cobertura {cov:.1f}%)")
    else:
        print(f"    consenso fuerte: 0 picks aún")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", type=str, default=None, help="Fecha final YYYY-MM-DD")
    args = ap.parse_args()
    end_date = date.fromisoformat(args.end) if args.end else date.today()

    print("═" * 74)
    print("  STRIKECAST · MONITOR CONSENSO FUERTE (validación ciega)")
    print(f"  Corte ciego desde: {BLIND_START}   |   Hasta: {end_date}")
    print("═" * 74)
    print("\n  Cargando data…")
    tr = load_data()

    # Baseline retrospectivo — solo referencia
    print("\n═══ Referencia retrospectiva (2025-2026 hasta el corte) ═══")
    retro = evaluate(tr, date(2025, 3, 1), BLIND_START)
    summarize(retro, "Retrospectivo (ya usado en calibración — NO es evidencia nueva)")

    # Ventana ciega
    if BLIND_START > end_date:
        print(f"\n  ⚠ Aún no hay juegos ciegos (BLIND_START={BLIND_START} > end={end_date})")
        return

    print(f"\n═══ Ventana ciega {BLIND_START} → {end_date} ═══")
    blind = evaluate(tr, BLIND_START, end_date)
    summarize(blind, "CIEGO (evidencia real)")

    # Detalle diario
    if blind is not None and not blind.empty:
        sc = blind[blind["strong"]].copy()
        if not sc.empty:
            print(f"\n═══ Detalle día a día — consenso fuerte ═══")
            daily = sc.groupby("date").agg(n=("won", "size"), hits=("won", "sum")).reset_index()
            for _, r in daily.iterrows():
                pct = r["hits"] / r["n"] * 100
                print(f"  {r['date']}  {int(r['hits'])}/{int(r['n'])} = {pct:5.1f}%")

    # Veredicto tentativo
    if blind is not None and not blind.empty:
        sc = blind[blind["strong"]]
        if len(sc) >= 30:
            pct = sc["won"].mean() * 100
            print(f"\n═══ Veredicto tentativo (N={len(sc)}) ═══")
            if pct >= 65:
                print(f"  🟢 {pct:.2f}% — CONFIRMA el histórico 68.72%")
            elif pct >= 58:
                print(f"  🟡 {pct:.2f}% — bajo el histórico pero aún útil")
            else:
                print(f"  🔴 {pct:.2f}% — RECHAZA el modo; retirar del UI")
        elif len(sc) > 0:
            print(f"\n  Necesitamos N≥30 para veredicto (actual: {len(sc)})")


if __name__ == "__main__":
    main()
