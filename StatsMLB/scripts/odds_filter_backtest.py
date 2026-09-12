"""Backtest honesto del filtro de mercado.

Pregunta: ¿si sumamos la voz del mercado como filtro/veto al pick del modelo,
sube o baja el win-rate por tier de confianza sobre los juegos históricos?

Aproximación (pragmática, no replica exacto la UI de comparación):
  * Voz "modelo": p_home de walkforward_preds.parquet (walk-forward oficial).
  * Voz "mercado": closing moneyline mediana entre DraftKings/ESPN BET,
    desvigada a p_home_market.
  * Tiers por confianza del modelo:
      LOCK     confidence >= 0.65
      FUERTE   0.58 <= confidence < 0.65
      JUEGA    0.53 <= confidence < 0.58
  * Regla de mercado (parametrizable):
      - BOOST : mercado coincide con pick del modelo con market_prob >= boost_th
      - VETO  : mercado contradice pick con market_prob >= veto_th → tier baja un escalón
  * Baseline: mismos tiers sin voz de mercado.

Salida: public/data/market-filter-audit.json con win-rate por tier en ambos
escenarios + detalle de picks vetados y su outcome real.

Uso:
    python scripts/odds_filter_backtest.py
    python scripts/odds_filter_backtest.py --veto 0.65 --boost 0.60
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PUBLIC_DATA = APP / "public" / "data"

SHARP_BOOKS = {"DraftKings", "ESPN BET"}


def load_market_probs() -> pd.DataFrame:
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    sharp = odds[odds["provider_name"].isin(SHARP_BOOKS)].copy()
    for side in ("home", "away"):
        sharp[f"{side}_ml"] = (
            sharp[f"{side}_ml_close"]
            .fillna(sharp[f"{side}_ml_current"])
            .fillna(sharp[f"{side}_ml_top"])
        )
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id", as_index=False).agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
    )

    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    agg = agg.merge(matched, on="espn_event_id", how="inner").drop_duplicates("game_pk")

    # American → decimal → devig
    def to_dec(v):
        if pd.isna(v) or v == 0:
            return np.nan
        return 1 + (v / 100 if v > 0 else 100 / abs(v))

    agg["dec_home"] = agg["home_ml"].apply(to_dec)
    agg["dec_away"] = agg["away_ml"].apply(to_dec)
    agg = agg.dropna(subset=["dec_home", "dec_away"])
    imp_home = 1 / agg["dec_home"]
    imp_away = 1 / agg["dec_away"]
    total = imp_home + imp_away
    agg["market_p_home"] = imp_home / total
    return agg[["game_pk", "market_p_home"]]


def load_model_preds() -> pd.DataFrame:
    preds = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    preds = preds.dropna(subset=["p_home", "home_win"]).copy()
    preds["game_date"] = pd.to_datetime(preds["game_date"])
    preds["season"] = preds["game_date"].dt.year
    return preds[["game_pk", "game_date", "season", "p_home", "home_win"]]


def tier_of(confidence: float) -> str:
    if confidence >= 0.65:
        return "LOCK"
    if confidence >= 0.58:
        return "FUERTE"
    if confidence >= 0.53:
        return "JUEGA"
    return "PASAR"


def downgrade(tier: str) -> str:
    return {"LOCK": "FUERTE", "FUERTE": "JUEGA", "JUEGA": "PASAR", "PASAR": "PASAR"}[tier]


def upgrade(tier: str) -> str:
    return {"PASAR": "JUEGA", "JUEGA": "FUERTE", "FUERTE": "LOCK", "LOCK": "LOCK"}[tier]


def evaluate(df: pd.DataFrame, veto_th: float, boost_th: float) -> dict:
    """Devuelve conteo/wins por tier en escenario baseline y con-mercado."""
    tiers = ["LOCK", "FUERTE", "JUEGA", "PASAR"]
    base = {t: {"n": 0, "wins": 0} for t in tiers}
    with_mkt = {t: {"n": 0, "wins": 0} for t in tiers}
    vetoed = {"picks_vetoed": 0, "would_have_won": 0, "would_have_lost": 0}
    boosted = {"picks_boosted": 0, "won": 0, "lost": 0}

    for _, row in df.iterrows():
        p_home = row["p_home"]
        pick_home = p_home >= 0.5
        model_conf = max(p_home, 1 - p_home)
        base_tier = tier_of(model_conf)
        won = int(pick_home == bool(row["home_win"]))
        base[base_tier]["n"] += 1
        base[base_tier]["wins"] += won

        mkt = row["market_p_home"]
        mkt_pick_home = mkt >= 0.5
        market_agrees = mkt_pick_home == pick_home
        market_prob_side = mkt if mkt_pick_home else 1 - mkt

        new_tier = base_tier
        if not market_agrees and market_prob_side >= veto_th and base_tier in ("LOCK", "FUERTE", "JUEGA"):
            new_tier = downgrade(base_tier)
            vetoed["picks_vetoed"] += 1
            if won:
                vetoed["would_have_won"] += 1
            else:
                vetoed["would_have_lost"] += 1
        elif market_agrees and market_prob_side >= boost_th and base_tier in ("FUERTE", "JUEGA"):
            new_tier = upgrade(base_tier)
            boosted["picks_boosted"] += 1
            if won:
                boosted["won"] += 1
            else:
                boosted["lost"] += 1

        with_mkt[new_tier]["n"] += 1
        with_mkt[new_tier]["wins"] += won

    def with_pct(d):
        return {t: {"n": v["n"], "wins": v["wins"], "pct": (v["wins"] / v["n"]) if v["n"] else None} for t, v in d.items()}

    return {
        "baseline": with_pct(base),
        "con_mercado": with_pct(with_mkt),
        "vetoed": vetoed,
        "boosted": boosted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--veto", type=float, default=0.62)
    parser.add_argument("--boost", type=float, default=0.58)
    args = parser.parse_args()

    market = load_market_probs()
    model = load_model_preds()
    df = model.merge(market, on="game_pk", how="inner")
    print(f"[backtest] {len(df):,} juegos históricos con modelo + mercado")

    overall = evaluate(df, args.veto, args.boost)

    by_season = {}
    for season, sub in df.groupby("season"):
        by_season[int(season)] = evaluate(sub, args.veto, args.boost) | {"games": int(len(sub))}

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "vetoThreshold": args.veto,
        "boostThreshold": args.boost,
        "totalGames": int(len(df)),
        "overall": overall,
        "bySeason": by_season,
        "notas": [
            "Modelo = walkforward_preds.parquet (walk-forward oficial de STRIKECAST).",
            "Mercado = mediana(DraftKings, ESPN BET) closing moneyline, desvigada.",
            "Tiers = LOCK≥65%, FUERTE≥58%, JUEGA≥53%, PASAR<53% (por confianza del modelo).",
            "Veto = mercado contradice al pick con prob ≥ vetoThreshold → tier baja un escalón.",
            "Boost = mercado confirma pick con prob ≥ boostThreshold → tier sube un escalón (excepto LOCK).",
        ],
    }
    out_path = PUBLIC_DATA / "market-filter-audit.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[backtest] escrito {out_path}")

    # Imprime resumen humano
    print("\n=== RESUMEN OVERALL ===")
    for tier in ("LOCK", "FUERTE", "JUEGA"):
        b = overall["baseline"][tier]
        m = overall["con_mercado"][tier]
        b_pct = f"{b['pct']:.1%}" if b["pct"] is not None else "—"
        m_pct = f"{m['pct']:.1%}" if m["pct"] is not None else "—"
        delta = (m["pct"] - b["pct"]) * 100 if b["pct"] is not None and m["pct"] is not None else None
        delta_str = f"{delta:+.1f}pp" if delta is not None else ""
        print(f"  {tier:8s}  baseline {b['wins']}/{b['n']} ({b_pct})  →  con-mkt {m['wins']}/{m['n']} ({m_pct})  {delta_str}")
    v = overall["vetoed"]
    b = overall["boosted"]
    veto_win_rate = v["would_have_won"] / (v["would_have_won"] + v["would_have_lost"]) if (v["would_have_won"] + v["would_have_lost"]) else None
    print(f"\n  VETOS: {v['picks_vetoed']} picks degradados. De esos, hubieran ganado {v['would_have_won']} ({veto_win_rate:.1%})" if veto_win_rate is not None else f"\n  VETOS: 0")
    boost_win_rate = b["won"] / (b["won"] + b["lost"]) if (b["won"] + b["lost"]) else None
    print(f"  BOOSTS: {b['picks_boosted']} picks elevados. Ganaron {b['won']} ({boost_win_rate:.1%})" if boost_win_rate is not None else "  BOOSTS: 0")


if __name__ == "__main__":
    main()
