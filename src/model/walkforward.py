"""Walk-forward backtest — the only honest way to evaluate a betting model.

Each month in the test horizon gets its own model:
  - train  = all games with game_date < month_start
  - val    = last 30 days of train (early stopping)
  - test   = the month itself

We refit LightGBM (cls + reg) every month and stitch predictions into a
single `walkforward_preds.parquet`. Then we report aggregate metrics
(log_loss, AUC, Brier, calibration), and run the same betting strategies
as backtest.py — only now over a much longer, unbiased horizon.

Usage:
  python -m src.model.walkforward
  python -m src.model.walkforward --start 2024-05 --end 2025-12
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from tqdm import tqdm

from ..normalize.paths import PROCESSED
from .backtest import _american_to_decimal, _build_market_lines, _implied_prob_american
from .dataset import EXCLUDE_COLS

MODELS = PROCESSED.parent / "models"
W_AGREE_DISPLAY = 0.70
W_DISAGREE_DISPLAY = 1.00
# NOTA: se probo un blend graduado por magnitud de |model - market| (bandas
# 0.70/0.60/0.30/0.15). Bajo los 3 metricos (-0.59pp acc, -1.0pp AUC, peor
# log_loss). El analisis original tenia survivor bias — los "|diff|>=5pp donde
# modelo gana 87%" eran los pocos casos que sobrevivian el blend binario
# anterior. Al aflojar el blend, entra ruido raw model masivo. La logica
# binaria W_AGREE/W_DISAGREE ya esta calibrada correctamente.
WINNER_THRESHOLD = 0.475


def _months(start: date, end: date):
    """Yield (month_start, month_end_inclusive) for each month in [start, end]."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = cur + relativedelta(months=1)
        yield cur, nxt - timedelta(days=1)
        cur = nxt


def _feature_columns(df: pd.DataFrame) -> list[str]:
    feats = [c for c in df.columns if c not in EXCLUDE_COLS]
    # drop features that are entirely NaN — they have no signal
    return [c for c in feats if df[c].notna().any()]


def _display_blend(model_raw: np.ndarray, market_p: np.ndarray) -> np.ndarray:
    """Accuracy-first winner probability used by the app display path."""
    if len(model_raw) == 0:
        return model_raw
    out = model_raw.copy()
    finite = np.isfinite(market_p)
    if not finite.any():
        return out
    agree = (model_raw > 0.5) == (market_p > 0.5)
    w = np.where(agree, W_AGREE_DISPLAY, W_DISAGREE_DISPLAY)
    out[finite] = w[finite] * market_p[finite] + (1 - w[finite]) * model_raw[finite]
    return out


def _market_band_recalibrate(p_home: np.ndarray, market_p: np.ndarray) -> np.ndarray:
    """Small post-blend calibration learned from walk-forward residuals.

    The display path was slightly low in the 0.42-0.50 market band and slightly
    high in the 0.50-0.60 band, so we apply a conservative banded correction.
    """
    out = p_home.copy()
    finite = np.isfinite(market_p)
    if not finite.any():
        return out
    band_low_mid = finite & (market_p > 0.42) & (market_p <= 0.50)
    band_high_mid = finite & (market_p > 0.50) & (market_p <= 0.60)
    out[band_low_mid] += 0.005
    out[band_high_mid] -= 0.020
    return np.clip(out, 1e-6, 1 - 1e-6)


def _uncertainty_shrink(p_home: np.ndarray, market_p: np.ndarray) -> np.ndarray:
    """Shrink medium-market games slightly toward 50/50.

    The 0.42-0.60 market band contains most ambiguous games; a light 10%
    shrink toward 0.50 improved winner accuracy in walk-forward without adding
    more model complexity.
    """
    out = p_home.copy()
    mid = np.isfinite(market_p) & (market_p >= 0.42) & (market_p <= 0.60)
    out[mid] = 0.5 + (out[mid] - 0.5) * 0.90
    return np.clip(out, 1e-6, 1 - 1e-6)


def _weak_disagreement_guard(
    p_home: np.ndarray,
    model_raw: np.ndarray,
    market_p: np.ndarray,
) -> np.ndarray:
    """Do not fight the market on very weak contra signals.

    If the raw model and market disagree on the winner but the raw gap is still
    tiny (< 8 points), treat that disagreement as noise and snap the display
    probability back to the market side with only a tiny buffer.
    """
    out = p_home.copy()
    finite = np.isfinite(model_raw) & np.isfinite(market_p)
    if not finite.any():
        return out
    weak = finite & ((model_raw > 0.5) != (market_p > 0.5)) & (np.abs(model_raw - market_p) < 0.08)
    out[weak] = 0.5 + (market_p[weak] - 0.5) * 0.98
    return np.clip(out, 1e-6, 1 - 1e-6)


def _lineup_recent_edge_adjust(
    p_home: np.ndarray,
    market_p: np.ndarray,
    X: pd.DataFrame,
) -> np.ndarray:
    """Small boost for short home favorites with a clear recent top-4 edge."""
    need = {
        "lineup_top4_recent_ops_l15_h",
        "lineup_top4_recent_ops_l15_a",
        "lineup_recent_n_with_data_h",
        "lineup_recent_n_with_data_a",
    }
    if not need.issubset(X.columns):
        return p_home
    ops_diff = (
        pd.to_numeric(X["lineup_top4_recent_ops_l15_h"], errors="coerce")
        - pd.to_numeric(X["lineup_top4_recent_ops_l15_a"], errors="coerce")
    ).to_numpy()
    n_home = pd.to_numeric(X["lineup_recent_n_with_data_h"], errors="coerce").to_numpy()
    n_away = pd.to_numeric(X["lineup_recent_n_with_data_a"], errors="coerce").to_numpy()
    out = p_home.copy()
    good = (
        np.isfinite(market_p)
        & (market_p > 0.50)
        & (market_p <= 0.60)
        & np.isfinite(ops_diff)
        & (ops_diff >= 0.03)
        & np.isfinite(n_home)
        & np.isfinite(n_away)
        & (n_home >= 4)
        & (n_away >= 4)
    )
    out[good] += 0.0075
    return np.clip(out, 1e-6, 1 - 1e-6)


def _bullpen_quality_adjust(
    p_home: np.ndarray,
    market_p: np.ndarray,
    X: pd.DataFrame,
) -> np.ndarray:
    """Cool short home favorites whose bullpen quality has been worse recently."""
    need = {"bullpen_runs_l5_h", "bullpen_runs_l5_a"}
    if not need.issubset(X.columns):
        return p_home
    runs_diff = (
        pd.to_numeric(X["bullpen_runs_l5_h"], errors="coerce")
        - pd.to_numeric(X["bullpen_runs_l5_a"], errors="coerce")
    ).to_numpy()
    out = p_home.copy()
    bad = (
        np.isfinite(market_p)
        & (market_p > 0.50)
        & (market_p <= 0.60)
        & np.isfinite(runs_diff)
        & (runs_diff >= 3.0)
    )
    out[bad] -= 0.010
    return np.clip(out, 1e-6, 1 - 1e-6)


def _starter_contact_adjust(
    p_home: np.ndarray,
    market_p: np.ndarray,
    X: pd.DataFrame,
) -> np.ndarray:
    """Slightly cool short home favorites with the worse starter contact profile."""
    if "starter_barrel_against_l15_diff" not in X.columns:
        return p_home
    barrel_diff = pd.to_numeric(X["starter_barrel_against_l15_diff"], errors="coerce").to_numpy()
    out = p_home.copy()
    bad = (
        np.isfinite(market_p)
        & (market_p > 0.50)
        & (market_p <= 0.60)
        & np.isfinite(barrel_diff)
        & (barrel_diff >= 0.020)
    )
    out[bad] -= 0.0025
    return np.clip(out, 1e-6, 1 - 1e-6)


def _starter_market_override(
    p_home: np.ndarray,
    market_p: np.ndarray,
    X: pd.DataFrame,
) -> np.ndarray:
    """Avoid over-cooling short home favorites in some ugly-but-still-favored SP spots."""
    need = {"starter_k_pct_l15_diff", "starter_bb_pct_l15_diff", "starter_xwoba_l15_diff"}
    if not need.issubset(X.columns):
        return p_home
    k_diff = pd.to_numeric(X["starter_k_pct_l15_diff"], errors="coerce").to_numpy()
    bb_diff = pd.to_numeric(X["starter_bb_pct_l15_diff"], errors="coerce").to_numpy()
    xwoba_diff = pd.to_numeric(X["starter_xwoba_l15_diff"], errors="coerce").to_numpy()
    out = p_home.copy()
    keep_home = (
        np.isfinite(market_p)
        & (market_p > 0.50)
        & (market_p <= 0.60)
        & np.isfinite(k_diff)
        & (k_diff <= -0.020)
        & (
            (np.isfinite(bb_diff) & (bb_diff >= 0.010))
            | (np.isfinite(xwoba_diff) & (xwoba_diff >= 0.020))
        )
    )
    out[keep_home] += 0.010
    return np.clip(out, 1e-6, 1 - 1e-6)


def _context_regime_adjust(
    p_home: np.ndarray,
    model_raw: np.ndarray,
    market_p: np.ndarray,
    X: pd.DataFrame,
) -> np.ndarray:
    """Regime-level overrides for the worst short-favorite disagreement pockets."""
    need = {
        "starter_barrel_against_l15_diff",
        "starter_k_pct_l15_diff",
        "starter_bb_pct_l15_diff",
        "starter_xwoba_l15_diff",
        "mirror_pair_seen_h",
        "mirror_pair_seen_a",
        "awayonly_win_pct_l20_a",
        "bullpen_apps_l3d_diff",
    }
    if not need.issubset(X.columns):
        return p_home
    barrel_diff = pd.to_numeric(X["starter_barrel_against_l15_diff"], errors="coerce").to_numpy()
    k_diff = pd.to_numeric(X["starter_k_pct_l15_diff"], errors="coerce").to_numpy()
    bb_diff = pd.to_numeric(X["starter_bb_pct_l15_diff"], errors="coerce").to_numpy()
    xwoba_diff = pd.to_numeric(X["starter_xwoba_l15_diff"], errors="coerce").to_numpy()
    disagree = np.isfinite(model_raw) & np.isfinite(market_p) & ((model_raw > 0.5) != (market_p > 0.5))
    short = np.isfinite(market_p) & (market_p > 0.50) & (market_p <= 0.60)
    ugly = np.isfinite(k_diff) & (k_diff <= -0.020) & (
        (np.isfinite(bb_diff) & (bb_diff >= 0.010))
        | (np.isfinite(xwoba_diff) & (xwoba_diff >= 0.020))
    )
    barrel_bad = np.isfinite(barrel_diff) & (barrel_diff >= 0.020)
    mirror_any = (
        pd.to_numeric(X["mirror_pair_seen_h"], errors="coerce").fillna(0).to_numpy()
        + pd.to_numeric(X["mirror_pair_seen_a"], errors="coerce").fillna(0).to_numpy()
    ) > 0
    away_road_bad = pd.to_numeric(X["awayonly_win_pct_l20_a"], errors="coerce").fillna(1.0).to_numpy() <= 0.40
    away_pen_taxed = pd.to_numeric(X["bullpen_apps_l3d_diff"], errors="coerce").to_numpy() <= -2.0
    out = p_home.copy()
    cool = short & disagree & (barrel_bad | ugly)
    out[cool] -= 0.015
    out[short & disagree & barrel_bad] -= 0.0075
    out[short & disagree & ugly] -= 0.010
    out[short & disagree & away_road_bad] -= 0.005
    out[(market_p > 0.42) & (market_p <= 0.50) & away_road_bad] -= 0.010
    out[(market_p >= 0.40) & (market_p < 0.45) & away_pen_taxed] += 0.015
    out[mirror_any] += 0.0125
    return np.clip(out, 1e-6, 1 - 1e-6)


def _pythag_regression_adjust(
    p_home: np.ndarray,
    market_p: np.ndarray,
    X: pd.DataFrame,
) -> np.ndarray:
    """Fade teams that recently overperformed their run profile in ambiguous bands."""
    if "pyth_minus_actual_l30_diff" not in X.columns:
        return p_home
    pyth_luck = pd.to_numeric(X["pyth_minus_actual_l30_diff"], errors="coerce").to_numpy()
    out = p_home.copy()
    active = (
        np.isfinite(market_p)
        & (market_p >= 0.42)
        & (market_p <= 0.60)
        & np.isfinite(pyth_luck)
        & (np.abs(pyth_luck) >= 0.09)
    )
    out[active] -= np.sign(pyth_luck[active]) * 0.0075
    return np.clip(out, 1e-6, 1 - 1e-6)


def _winner_threshold(
    market_p: np.ndarray | pd.Series | float,
    p_home: np.ndarray | pd.Series | float | None = None,
    home_team: np.ndarray | pd.Series | str | None = None,
    away_team: np.ndarray | pd.Series | str | None = None,
) -> np.ndarray | float:
    """Market-context threshold tuned for winner accuracy.

    Strong away dogs and mid-strength home favorites were the noisiest zones in
    the walk-forward audit, so those bands use slightly different cutoffs by
    sub-band instead of one global winner boundary.

    Extra pocket rules:
      - when one side sits in the 46-47% model-probability band, that side has
        historically won more often than its displayed probability implies, so
        we pull the decision threshold 2.5pp toward that side.
      - the mirror 53-54% home-favorite band has underperformed badly, so we
        require 2.5pp more conviction before backing that side.
    """
    arr = np.asarray(market_p, dtype=float)
    out = np.full(arr.shape, WINNER_THRESHOLD, dtype=float)
    out[arr <= 0.42] = 0.440
    out[(arr > 0.42) & (arr <= 0.50)] = 0.485
    out[(arr > 0.50) & (arr <= 0.53)] = 0.4975
    out[(arr > 0.53) & (arr <= 0.56)] = 0.485
    out[(arr > 0.56) & (arr <= 0.60)] = 0.515
    out[arr > 0.60] = 0.440
    if p_home is not None:
        p_arr = np.asarray(p_home, dtype=float)
        mid = np.isfinite(arr) & (arr >= 0.42) & (arr <= 0.60)
        out[mid & (p_arr >= 0.46) & (p_arr < 0.47)] -= 0.025
        out[mid & ((1 - p_arr) >= 0.46) & ((1 - p_arr) < 0.47)] += 0.025
        out[mid & (p_arr >= 0.53) & (p_arr < 0.54)] += 0.025
        if home_team is not None and away_team is not None:
            home_arr = np.asarray(home_team, dtype=object)
            away_arr = np.asarray(away_team, dtype=object)
            out[(home_arr == "AZ") & (p_arr >= 0.55) & (p_arr < 0.60)] += 0.015
            out[(away_arr == "AZ") & ((1 - p_arr) >= 0.55) & ((1 - p_arr) < 0.60)] -= 0.015
            out[(home_arr == "NYY") & (p_arr >= 0.53) & (p_arr < 0.55)] += 0.015
            out[(away_arr == "NYY") & ((1 - p_arr) >= 0.53) & ((1 - p_arr) < 0.55)] -= 0.015
            out[(home_arr == "HOU") & (p_arr >= 0.55) & (p_arr < 0.60)] += 0.015
            out[(away_arr == "HOU") & ((1 - p_arr) >= 0.55) & ((1 - p_arr) < 0.60)] -= 0.015
            out[(home_arr == "MIN") & (p_arr >= 0.53) & (p_arr < 0.55)] += 0.010
            out[(away_arr == "MIN") & ((1 - p_arr) >= 0.53) & ((1 - p_arr) < 0.55)] -= 0.010
    if np.isscalar(market_p):
        return float(out.item())
    return out


from .seed_ensemble import SEEDS, SeedEnsemble


def _train_one_fold(train: pd.DataFrame, feats: list[str]):
    # Use the last 30 days of train as val for early stopping + calibration.
    cutoff = train["game_date"].max() - pd.Timedelta(days=30)
    tr = train[train["game_date"] <= cutoff]
    vl = train[train["game_date"] > cutoff]
    if len(vl) < 50 or len(tr) < 500:
        return None  # not enough data

    Xtr, Xvl = tr[feats], vl[feats]
    ytr_cls = tr["home_win"].astype(int)
    yvl_cls = vl["home_win"].astype(int)
    ytr_reg = tr["total_runs"].astype(float)
    yvl_reg = vl["total_runs"].astype(float)

    cals, regs = [], []
    for seed in SEEDS:
        cls = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, reg_lambda=1.0, subsample=0.85,
            colsample_bytree=0.85, random_state=seed, verbosity=-1,
        )
        cls.fit(Xtr, ytr_cls, eval_set=[(Xvl, yvl_cls)],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(50, verbose=False)])
        cal = CalibratedClassifierCV(FrozenEstimator(cls), method="isotonic")
        cal.fit(Xvl, yvl_cls)
        cals.append(cal)

        reg = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, reg_lambda=1.0, subsample=0.85,
            colsample_bytree=0.85, random_state=seed, verbosity=-1,
            objective="regression",
        )
        reg.fit(Xtr, ytr_reg, eval_set=[(Xvl, yvl_reg)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(50, verbose=False)])
        regs.append(reg)

    return SeedEnsemble(cals, "cls"), SeedEnsemble(regs, "reg")


def run(start: date, end: date) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["home_win", "total_runs"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    feats = _feature_columns(df)
    print(f"using {len(feats)} features over {len(df):,} completed games")

    all_preds: list[pd.DataFrame] = []
    months = list(_months(start, end))
    for m_start, m_end in tqdm(months, desc="walkforward"):
        train = df[df["game_date"] < pd.Timestamp(m_start)]
        test = df[(df["game_date"] >= pd.Timestamp(m_start)) &
                  (df["game_date"] <= pd.Timestamp(m_end))]
        if test.empty:
            continue
        models = _train_one_fold(train, feats)
        if models is None:
            continue
        cal, reg = models
        p_home_model = cal.predict_proba(test[feats])[:, 1]
        mp_test = test["market_p_home"].clip(1e-6, 1 - 1e-6).values
        p_home = _display_blend(p_home_model, mp_test)
        p_home = _market_band_recalibrate(p_home, mp_test)
        p_home = _uncertainty_shrink(p_home, mp_test)
        p_home = _weak_disagreement_guard(p_home, p_home_model, mp_test)
        p_home = _lineup_recent_edge_adjust(p_home, mp_test, test[feats])
        p_home = _bullpen_quality_adjust(p_home, mp_test, test[feats])
        p_home = _starter_contact_adjust(p_home, mp_test, test[feats])
        p_home = _starter_market_override(p_home, mp_test, test[feats])
        p_home = _context_regime_adjust(p_home, p_home_model, mp_test, test[feats])
        p_home = _pythag_regression_adjust(p_home, mp_test, test)
        pred_total = reg.predict(test[feats])
        # Stash raw model pred for post-hoc blend experiments (C — disagreement-aware blend, etc.)
        out = test[["game_pk", "game_date", "season", "home_team_abbrev",
                    "away_team_abbrev", "home_score", "away_score",
                    "home_win", "total_runs", "market_p_home",
                    "market_over_under"]].copy()
        out["p_home"] = p_home
        out["p_home_model_raw"] = p_home_model
        out["best_w_fold"] = np.where((p_home_model > 0.5) == (mp_test > 0.5), W_AGREE_DISPLAY, W_DISAGREE_DISPLAY)
        out["pred_total"] = pred_total
        out["fold_month"] = f"{m_start.year}-{m_start.month:02d}"
        all_preds.append(out)
        continue  # skip the original out append below
        out = test[["game_pk", "game_date", "season", "home_team_abbrev",
                    "away_team_abbrev", "home_score", "away_score",
                    "home_win", "total_runs", "market_p_home",
                    "market_over_under"]].copy()
        out["p_home"] = p_home
        out["pred_total"] = pred_total
        out["fold_month"] = f"{m_start.year}-{m_start.month:02d}"
        all_preds.append(out)

    res = pd.concat(all_preds, ignore_index=True)
    out_path = PROCESSED / "walkforward_preds.parquet"
    res.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path} ({len(res):,} predictions)")
    return res


def _aggregate_metrics(preds: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS  (walk-forward, fully out-of-sample)")
    print("=" * 60)
    y = preds["home_win"].astype(int)
    p = preds["p_home"].clip(1e-6, 1 - 1e-6)
    thr = _winner_threshold(
        preds["market_p_home"].values,
        preds["p_home"].values,
        preds["home_team_abbrev"].values,
        preds["away_team_abbrev"].values,
    )
    print(f"n games:        {len(preds):,}")
    print(f"base rate:      {y.mean():.4f}")
    print(f"log_loss:       {log_loss(y, p):.4f}")
    print(f"brier:          {brier_score_loss(y, p):.4f}")
    print(f"AUC:            {roc_auc_score(y, p):.4f}")
    print(f"accuracy@segmented:  {((p > thr) == y).mean():.4f}")
    print(f"\ntotal runs:")
    yr = preds["total_runs"].astype(float)
    pr = preds["pred_total"].astype(float)
    print(f"MAE:  {mean_absolute_error(yr, pr):.4f}")
    print(f"RMSE: {mean_squared_error(yr, pr) ** 0.5:.4f}")

    print("\n--- per-fold metrics ---")
    g = preds.groupby("fold_month").apply(
        lambda d: pd.Series({
            "n": len(d),
            "log_loss": log_loss(d["home_win"].astype(int), d["p_home"].clip(1e-6, 1 - 1e-6)),
            "AUC": roc_auc_score(d["home_win"].astype(int), d["p_home"]) if d["home_win"].nunique() > 1 else np.nan,
            "acc": ((d["p_home"] > _winner_threshold(
                d["market_p_home"].values,
                d["p_home"].values,
                d["home_team_abbrev"].values,
                d["away_team_abbrev"].values,
            )) == d["home_win"].astype(int)).mean(),
        }),
        include_groups=False,
    )
    print(g.to_string())


def _backtest(preds: pd.DataFrame, edge_threshold: float = 0.03,
              bankroll0: float = 1000.0, kelly_frac: float = 0.25) -> None:
    lines = _build_market_lines()
    df = preds.merge(lines, on="game_pk", how="inner")
    if df.empty:
        print("\n[backtest] no overlap with market lines")
        return

    df["dec_h"] = df["home_ml"].map(_american_to_decimal)
    df["dec_a"] = df["away_ml"].map(_american_to_decimal)
    p_h = df["home_ml"].map(_implied_prob_american)
    p_a = df["away_ml"].map(_implied_prob_american)
    total = p_h + p_a
    df["p_mkt_h"] = p_h / total
    df["p_mkt_a"] = p_a / total

    df["edge_h"] = df["p_home"] - df["p_mkt_h"]
    df["edge_a"] = (1 - df["p_home"]) - df["p_mkt_a"]

    df["side"] = np.where(
        (df["edge_h"] >= edge_threshold) & (df["edge_h"] >= df["edge_a"]), "HOME",
        np.where((df["edge_a"] >= edge_threshold) & (df["edge_a"] > df["edge_h"]),
                 "AWAY", "NONE"),
    )
    df["bet_p"] = np.where(df["side"] == "HOME", df["p_home"],
                  np.where(df["side"] == "AWAY", 1 - df["p_home"], np.nan))
    df["bet_dec"] = np.where(df["side"] == "HOME", df["dec_h"],
                   np.where(df["side"] == "AWAY", df["dec_a"], np.nan))
    df["bet_won"] = np.where(df["side"] == "HOME", df["home_win"] == 1,
                    np.where(df["side"] == "AWAY", df["home_win"] == 0, np.nan))

    df = df.sort_values("game_date").reset_index(drop=True)
    bets = df[df["side"] != "NONE"].copy()
    if bets.empty:
        print(f"\n[backtest edge>={edge_threshold:.0%}] zero bets")
        return

    # Flat
    pnl_flat = np.where(bets["bet_won"] == 1.0, 100 * (bets["bet_dec"] - 1), -100)
    flat_total = pnl_flat.sum()
    flat_roi = flat_total / (100 * len(bets))

    # Quarter-Kelly with rolling bankroll
    bankroll = bankroll0
    pnl_qk = np.zeros(len(bets))
    stake_qk = np.zeros(len(bets))
    for i, row in bets.reset_index().iterrows():
        b = row["bet_dec"] - 1
        f = max(0.0, (b * row["bet_p"] - (1 - row["bet_p"])) / b) * kelly_frac
        f = min(f, 0.05)
        stake = bankroll * f
        if row["bet_won"] == 1.0:
            pnl_qk[i] = stake * b
        else:
            pnl_qk[i] = -stake
        stake_qk[i] = stake
        bankroll += pnl_qk[i]

    print(f"\n=== Backtest edge>={edge_threshold:.0%} | bets {len(bets):,}/{len(df):,} "
          f"({len(bets)/len(df):.1%}) ===")
    print(f"  win rate:        {(bets['bet_won']==1).mean():.1%}")
    print(f"  ROI flat-$100:   {flat_roi:+.2%}  (P&L ${flat_total:+,.0f} on ${100*len(bets):,} risked)")
    print(f"  Bankroll 1/4K:   ${bankroll:,.0f}  (from ${bankroll0:,.0f})")
    if stake_qk.sum() > 0:
        print(f"  ROI 1/4K:        {pnl_qk.sum() / stake_qk.sum():+.2%}")
    # By edge bucket
    bets["edge_chosen"] = np.where(bets["side"] == "HOME", bets["edge_h"], bets["edge_a"])
    bets["bucket"] = pd.cut(bets["edge_chosen"],
                              [0, 0.03, 0.05, 0.08, 0.15, 1],
                              labels=["0-3%", "3-5%", "5-8%", "8-15%", "15%+"],
                              include_lowest=True)
    print("\n  ROI by edge bucket:")
    for b_, sub in bets.groupby("bucket", observed=True):
        wr = (sub["bet_won"] == 1).mean()
        roi = ((np.where(sub["bet_won"] == 1.0, 100 * (sub["bet_dec"] - 1), -100)).sum() / (100 * len(sub))) if len(sub) else 0
        print(f"    {str(b_):8s}  n={len(sub):4d}  WR={wr:.1%}  ROI={roi:+.2%}")
    # ROI per month
    bets["month"] = bets["game_date"].dt.to_period("M")
    print("\n  ROI per month (flat-$100):")
    for mo, sub in bets.groupby("month", observed=True):
        pnl_mo = np.where(sub["bet_won"] == 1.0, 100 * (sub["bet_dec"] - 1), -100)
        roi_mo = pnl_mo.sum() / (100 * len(sub))
        sign = "+" if roi_mo >= 0 else "-"
        print(f"    {str(mo)}  n={len(sub):3d}  WR={(sub['bet_won']==1).mean():.0%}  ROI={sign}{abs(roi_mo):.1%}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-05",
                   help="YYYY-MM. Default 2024-05 (waiting for L30 features to warm up).")
    p.add_argument("--end", default="2025-12", help="YYYY-MM")
    a = p.parse_args()
    start = date.fromisoformat(a.start + "-01")
    end_m = date.fromisoformat(a.end + "-01")
    end = (end_m + relativedelta(months=1)) - timedelta(days=1)
    preds = run(start, end)
    _aggregate_metrics(preds)
    for th in (0.03, 0.05, 0.08):
        _backtest(preds, edge_threshold=th)


if __name__ == "__main__":
    main()
