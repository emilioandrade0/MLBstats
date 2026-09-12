"""Backtest v2 — filtro basado en MAGNITUD de acuerdo modelo↔mercado.

Hallazgo de v1: modelo y mercado casi siempre coinciden en dirección, así que
el "veto por dirección contraria" nunca dispara y el "boost por acuerdo simple"
diluye picks buenos con picks mediocres.

Nueva hipótesis: la señal útil está en la MAGNITUD.
  * Cuando mercado > modelo (mercado más confiado del mismo lado) →
    los sharps saben algo → boost real.
  * Cuando mercado < modelo (mercado menos confiado o va al otro lado) →
    el modelo puede estar sobrevalorando → veto.

Salida: público/data/market-filter-audit.json
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

    def to_dec(v):
        if pd.isna(v) or v == 0:
            return np.nan
        return 1 + (v / 100 if v > 0 else 100 / abs(v))

    agg["dec_home"] = agg["home_ml"].apply(to_dec)
    agg["dec_away"] = agg["away_ml"].apply(to_dec)
    agg = agg.dropna(subset=["dec_home", "dec_away"])
    ih, ia = 1 / agg["dec_home"], 1 / agg["dec_away"]
    total = ih + ia
    agg["market_p_home"] = ih / total
    return agg[["game_pk", "market_p_home"]]


def load_model_preds() -> pd.DataFrame:
    preds = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    preds = preds.dropna(subset=["p_home", "home_win"]).copy()
    preds["game_date"] = pd.to_datetime(preds["game_date"])
    preds["season"] = preds["game_date"].dt.year
    return preds[["game_pk", "game_date", "season", "p_home", "home_win"]]


def evaluate(df: pd.DataFrame, market_lead_min: float, market_lag_min: float) -> dict:
    """Categoriza cada pick del modelo según señal del mercado y evalúa win-rate."""
    df = df.copy()
    df["pick_home"] = (df["p_home"] >= 0.5).astype(int)
    df["model_conf"] = df[["p_home"]].apply(lambda r: max(r["p_home"], 1 - r["p_home"]), axis=1)
    df["market_conf_pick_side"] = df.apply(
        lambda r: r["market_p_home"] if r["pick_home"] else 1 - r["market_p_home"], axis=1
    )
    df["market_delta"] = df["market_conf_pick_side"] - df["model_conf"]
    # delta > 0 → mercado más confiado que modelo (buena señal)
    # delta < 0 → mercado menos confiado o va al otro lado (mala señal)
    df["won"] = (df["pick_home"] == df["home_win"].astype(int)).astype(int)

    # Segmentación por magnitud de delta
    def cat(d):
        if d >= market_lead_min:
            return "MKT_MAS_ALTO"     # mercado > modelo (por al menos market_lead_min)
        if d <= market_lag_min:
            return "MKT_MAS_BAJO"     # mercado < modelo o al otro lado
        return "COINCIDEN"

    df["cat"] = df["market_delta"].apply(cat)

    # También cortes por confianza base del modelo
    def tier(c):
        if c >= 0.65: return "LOCK"
        if c >= 0.58: return "FUERTE"
        if c >= 0.53: return "JUEGA"
        return "PASAR"

    df["base_tier"] = df["model_conf"].apply(tier)

    # Cruce: tier x categoría de mercado
    matrix = {}
    for base in ("LOCK", "FUERTE", "JUEGA"):
        matrix[base] = {}
        sub_tier = df[df["base_tier"] == base]
        for cat_name in ("MKT_MAS_ALTO", "COINCIDEN", "MKT_MAS_BAJO"):
            sub = sub_tier[sub_tier["cat"] == cat_name]
            n = int(len(sub))
            w = int(sub["won"].sum())
            matrix[base][cat_name] = {
                "n": n,
                "wins": w,
                "pct": (w / n) if n else None,
            }

    # También overall por categoría de mercado (todos los tiers arriba de PASAR)
    overall_cat = {}
    playable = df[df["base_tier"] != "PASAR"]
    for cat_name in ("MKT_MAS_ALTO", "COINCIDEN", "MKT_MAS_BAJO"):
        sub = playable[playable["cat"] == cat_name]
        overall_cat[cat_name] = {
            "n": int(len(sub)),
            "wins": int(sub["won"].sum()),
            "pct": (int(sub["won"].sum()) / len(sub)) if len(sub) else None,
        }

    # Distribución del delta (percentiles) para calibrar umbrales
    deltas = df["market_delta"].dropna()
    dist = {
        "p10": float(np.percentile(deltas, 10)),
        "p25": float(np.percentile(deltas, 25)),
        "p50": float(np.percentile(deltas, 50)),
        "p75": float(np.percentile(deltas, 75)),
        "p90": float(np.percentile(deltas, 90)),
        "mean": float(deltas.mean()),
        "std": float(deltas.std()),
    }

    return {
        "matrix": matrix,
        "overallByMarketCat": overall_cat,
        "deltaDistribution": dist,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead", type=float, default=0.03,
                        help="Delta mínimo (mercado>modelo) para considerar MKT_MAS_ALTO")
    parser.add_argument("--lag", type=float, default=-0.03,
                        help="Delta máximo (mercado<modelo) para considerar MKT_MAS_BAJO")
    args = parser.parse_args()

    market = load_market_probs()
    model = load_model_preds()
    df = model.merge(market, on="game_pk", how="inner")
    print(f"[backtest v2] {len(df):,} juegos con modelo + mercado")

    overall = evaluate(df, args.lead, args.lag)
    by_season = {}
    for season, sub in df.groupby("season"):
        by_season[int(season)] = evaluate(sub, args.lead, args.lag) | {"games": int(len(sub))}

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "leadThreshold": args.lead,
        "lagThreshold": args.lag,
        "totalGames": int(len(df)),
        "overall": overall,
        "bySeason": by_season,
        "notas": [
            "MKT_MAS_ALTO = mercado da al pick del modelo probabilidad al menos leadThreshold puntos MÁS que el modelo.",
            "MKT_MAS_BAJO = mercado da al pick menos de lagThreshold puntos MENOS.",
            "COINCIDEN = mercado y modelo con confianzas parecidas.",
            "Hipótesis: MKT_MAS_ALTO tiene win-rate superior; MKT_MAS_BAJO inferior.",
        ],
    }
    out = PUBLIC_DATA / "market-filter-audit.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[backtest v2] escrito {out}")


if __name__ == "__main__":
    main()
