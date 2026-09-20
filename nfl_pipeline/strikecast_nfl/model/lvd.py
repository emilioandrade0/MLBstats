"""L / V / D — clasificación a 3 vías del resultado, walk-forward estricto.

Clases (margen = home_score - away_score):
  L  Local      → margen  >  +6   (el local gana por más de 6)
  V  Visita     → margen  <  -6   (el visitante gana por más de 6)
  D  Diferencia → -6 ≤ margen ≤ 6 (partido cerrado, sin importar quién gane)

Modelo
------
Regresión logística multinomial sobre (spread, spread²). Se probaron en
walk-forward LightGBM multiclase, regresión de margen, Elo/EPA como features
extra y reglas de umbral: todo converge en ~49% de accuracy (baseline
"siempre D" = 40%). El spread de cierre ya resume la información disponible
y el margen real tiene σ ≈ 12.7 pts, así que el modelo simple es el más robusto
y además entrega probabilidades calibradas por clase (la confianza declarada
coincide con el acierto real por bucket).

Para juegos futuros sin spread publicado se usa un spread implícito por Elo,
ajustado con una regresión lineal spread ~ elo_diff sobre el pool de
entrenamiento (`spread_source = "elo"`).

Walk-forward: para cada temporada T, se entrena con temporadas < T y se
predice semana a semana; tras cada semana jugada se reentrena incluyendo esa
semana. Ninguna predicción usa información posterior a su kickoff.

Output: data/processed/lvd_preds.parquet
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..paths import PROCESSED

warnings.filterwarnings("ignore")

MARGIN_THRESHOLD = 6
CLASSES = np.array(["V", "D", "L"])      # index 0,1,2 — orden estable
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
MIN_TRAIN_GAMES = 250   # una temporada completa basta para 2 features


def label(margin: pd.Series) -> pd.Series:
    """Margen → L / V / D. NaN cuando el juego no ha terminado."""
    out = pd.Series(np.where(margin > MARGIN_THRESHOLD, "L",
                    np.where(margin < -MARGIN_THRESHOLD, "V", "D")), index=margin.index, dtype=object)
    out[margin.isna()] = None
    return out


def _fit(train: pd.DataFrame):
    X = np.column_stack([train["spread_used"], train["spread_used"] ** 2])
    y = train["actual"].map(CLASS_IDX).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    return model.fit(X, y)


def _fit_elo_to_spread(train: pd.DataFrame) -> tuple[float, float]:
    """spread ≈ a * elo_diff + b — para inferir spread cuando el mercado aún no publicó."""
    sub = train.dropna(subset=["spread_line", "elo_diff"])
    if len(sub) < 50:
        return 1 / 25.0, 0.0
    a, b = np.polyfit(sub["elo_diff"].astype(float), sub["spread_line"].astype(float), 1)
    return float(a), float(b)


def _predict(model, games: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([games["spread_used"], games["spread_used"] ** 2])
    return model.predict_proba(X)


def run(start_season: int, end_season: int) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.sort_values("kickoff_utc").reset_index(drop=True)
    df["spread_line"] = pd.to_numeric(df["spread_line"], errors="coerce")
    df["elo_diff"] = pd.to_numeric(df["elo_diff"], errors="coerce")
    is_final = df["status"] == "final"
    df["margin"] = np.where(is_final, df["home_score"] - df["away_score"], np.nan)
    df["actual"] = label(df["margin"])

    seasons = [s for s in sorted(df["season"].unique()) if start_season <= s <= end_season]
    out_rows: list[pd.DataFrame] = []

    for season in seasons:
        past = df[(df["season"] < season) & is_final].copy()
        current = df[df["season"] == season].sort_values(["week", "kickoff_utc"])
        weeks = sorted(current["week"].dropna().unique().tolist())
        pool = past
        print(f"\n-- Temporada {season}  (pool: {len(pool):,} juegos)")

        for wk in weeks:
            wk_games = current[current["week"] == wk].copy()

            # Pool de entrenamiento siempre con spread real (finals con línea de cierre)
            train = pool.dropna(subset=["spread_line"]).copy()
            train["spread_used"] = train["spread_line"]
            if len(train) < MIN_TRAIN_GAMES:
                print(f"  wk {int(wk):>2}: pool insuficiente ({len(train)}), sin predicción")
                pool = pd.concat([pool, wk_games[wk_games["status"] == "final"]], ignore_index=True)
                continue

            a, b = _fit_elo_to_spread(train)
            elo_spread = a * wk_games["elo_diff"] + b
            wk_games["spread_used"] = wk_games["spread_line"].fillna(elo_spread)
            wk_games["spread_source"] = np.where(wk_games["spread_line"].notna(), "market",
                                          np.where(wk_games["elo_diff"].notna(), "elo", None))
            predictable = wk_games["spread_used"].notna()

            probs = np.full((len(wk_games), 3), np.nan)
            if predictable.any():
                model = _fit(train)
                probs[predictable.values] = _predict(model, wk_games[predictable])

            res = wk_games[["game_id", "season", "week", "status", "spread_used", "spread_source", "actual"]].copy()
            res["p_visita"], res["p_diferencia"], res["p_local"] = probs[:, 0], probs[:, 1], probs[:, 2]
            has = ~np.isnan(probs[:, 0])
            res["pick"] = np.where(has, CLASSES[np.nan_to_num(probs, nan=-1).argmax(1)], None)
            res["confidence"] = np.where(has, np.nanmax(probs, axis=1), np.nan)
            res["hit"] = np.where(res["actual"].notna() & res["pick"].notna(),
                                  res["pick"] == res["actual"], None)
            out_rows.append(res)

            finals = wk_games[wk_games["status"] == "final"]
            if not finals.empty:
                pool = pd.concat([pool, finals], ignore_index=True)

            n_hit = int(res["hit"].fillna(False).astype(bool).sum())
            n_eval = int(res["hit"].notna().sum())
            print(f"  wk {int(wk):>2}: {len(res):>2} juegos  "
                  f"{'acc ' + f'{n_hit}/{n_eval}' if n_eval else 'sin resultados'}")

    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    out["hit"] = out["hit"].astype("boolean")
    path = PROCESSED / "lvd_preds.parquet"
    out.to_parquet(path, index=False)
    print(f"\nlvd_preds.parquet: {len(out):,} filas -> {path}")
    return out


def evaluate(preds: pd.DataFrame) -> None:
    ev = preds[preds["hit"].notna()].copy()
    if ev.empty:
        print("Sin juegos finalizados para evaluar.")
        return
    ev["hit"] = ev["hit"].astype(bool)
    print("\n-- Evaluación L/V/D (solo juegos finalizados) --")
    print(f"  accuracy global: {ev['hit'].mean():.3f}   n={len(ev):,}   "
          f"baseline(siempre D)={ (ev['actual'] == 'D').mean():.3f}")
    print("  por temporada:")
    for s, g in ev.groupby("season"):
        print(f"    {s}: {g['hit'].mean():.3f}  (n={len(g)})")
    print("  por confianza:")
    bins = [0, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
    ev["bucket"] = pd.cut(ev["confidence"], bins)
    for b, g in ev.groupby("bucket", observed=True):
        print(f"    {str(b):<12} acc={g['hit'].mean():.3f}  n={len(g)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2100)
    args = ap.parse_args()
    preds = run(args.start, args.end)
    evaluate(preds)


if __name__ == "__main__":
    main()
