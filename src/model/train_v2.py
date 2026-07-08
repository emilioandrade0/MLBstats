"""Train v2 model experiment — safe parallel to production.

Differences vs train.py / ensemble.py (v1):
  - Train set extended to 2025-09-30 (adds full 2025 regular season)
  - Val  set: 2025-10-01 to 2026-03-31  (postseason-2025 + spring calibration)
  - Test set: 2026-04-01 to today        (entire 2026 season, never seen)
  - Saves to data/models/v2/             (production models untouched)

Run:
  python -m src.model.train_v2           # train + compare
  python -m src.model.train_v2 --deploy  # also copy v2 -> data/models/ if v2 wins

Walk-forward comparison (recommended after train_v2):
  python -m src.model.walkforward --start 2026-01 --end 2026-06
  (retrains monthly folds using all pre-month data — honest v2 eval on 2026)
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from ..normalize.paths import PROCESSED
from .dataset import EXCLUDE_COLS

MODELS_V1 = PROCESSED.parent / "models"
MODELS_V2 = PROCESSED.parent / "models" / "v2"

# Updated splits — v1 used train_end=2024-12-31; v2 adds first half of 2025.
# Val = second half 2025 regular season (~750 games, same type as training)
# Test = entire 2026 season (completely unseen)
TRAIN_END = "2025-05-31"
VAL_END   = "2025-09-30"
TEST_END  = date.today().isoformat()


def _feature_columns(df: pd.DataFrame) -> list[str]:
    feats = [c for c in df.columns if c not in EXCLUDE_COLS]
    nan_frac = df[feats].isna().mean()
    return [c for c in feats if nan_frac[c] < 0.95]


def _load_splits(df: pd.DataFrame, feats: list[str]):
    tr = df[df["game_date"] <= TRAIN_END]
    vl = df[(df["game_date"] > TRAIN_END) & (df["game_date"] <= VAL_END)]
    ts = df[(df["game_date"] > VAL_END)   & (df["game_date"] <= TEST_END)]
    return (
        tr[feats], tr["home_win"].astype(int), tr["total_runs"].astype(float),
        vl[feats], vl["home_win"].astype(int), vl["total_runs"].astype(float),
        ts[feats], ts["home_win"].astype(int), ts["total_runs"].astype(float),
        len(tr), len(vl), len(ts),
    )


def _train_cls(Xtr, ytr, Xvl, yvl):
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, reg_lambda=1.0, reg_alpha=0.0,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def _train_reg(Xtr, ytr, Xvl, yvl):
    m = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, reg_lambda=1.0,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=-1, objective="regression",
    )
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], eval_metric="l2",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def _eval_cls(p, y, label: str) -> dict:
    acc = float(((p > 0.5) == y).mean())
    ll  = float(log_loss(y, p.clip(1e-6, 1 - 1e-6)))
    auc = float(roc_auc_score(y, p))
    br  = float(brier_score_loss(y, p))
    print(f"  {label:35s}  acc={acc:.4f}  ll={ll:.4f}  AUC={auc:.4f}  Brier={br:.4f}")
    return {"accuracy": acc, "log_loss": ll, "auc": auc, "brier": br}


def _load_v1_report() -> dict:
    p = MODELS_V1 / "report.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true",
                    help="Copy v2 models -> data/models/ if v2 accuracy >= v1.")
    args = ap.parse_args()

    MODELS_V2.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["home_win", "total_runs"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    feats = _feature_columns(df)

    (Xtr, ytr_c, ytr_r,
     Xvl, yvl_c, yvl_r,
     Xts, yts_c, yts_r,
     n_tr, n_vl, n_ts) = _load_splits(df, feats)

    print(f"\n{'='*60}")
    print(f"  train_v2  |  {len(feats)} features  |  {len(df):,} games total")
    print(f"  train: {TRAIN_END}  ({n_tr:,})")
    print(f"  val:   {TRAIN_END} to {VAL_END}  ({n_vl:,})")
    print(f"  test:  {VAL_END} to {TEST_END}  ({n_ts:,})")
    print(f"{'='*60}\n")

    if n_ts < 50:
        print(f"  WARNING: only {n_ts} test games — run update_data.bat first!")

    # ── Train classifier ───────────────────────────────────────────────────────
    print("[1/4] LightGBM classifier (home-win)")
    cls = _train_cls(Xtr, ytr_c, Xvl, yvl_c)
    cal = CalibratedClassifierCV(FrozenEstimator(cls), method="isotonic")
    cal.fit(Xvl, yvl_c)
    p_val = cal.predict_proba(Xvl)[:, 1]
    p_tst = cal.predict_proba(Xts)[:, 1]
    print("  val:")
    _eval_cls(p_val, yvl_c.values, "lgb_cls calibrated")
    print("  test (2026 season):")
    v2_cls_test = _eval_cls(p_tst, yts_c.values, "lgb_cls calibrated")

    # ── Train regressor ────────────────────────────────────────────────────────
    print("\n[2/4] LightGBM regressor (total-runs)")
    reg = _train_reg(Xtr, ytr_r, Xvl, yvl_r)
    pred_tot = reg.predict(Xts)
    mae_v2 = float(mean_absolute_error(yts_r.values, pred_tot))
    rmse_v2 = float(mean_squared_error(yts_r.values, pred_tot) ** 0.5)
    print(f"  test MAE={mae_v2:.4f}  RMSE={rmse_v2:.4f}")

    # ── Ensemble (market blend) ────────────────────────────────────────────────
    print("\n[3/4] Ensemble (market blend)")
    df_mkt = df.dropna(subset=["market_p_home"]).copy()
    df_mkt["game_date"] = pd.to_datetime(df_mkt["game_date"])
    ens_feats = [c for c in feats if df_mkt[c].notna().any()]

    ens_tr = df_mkt[df_mkt["game_date"] <= TRAIN_END]
    ens_vl = df_mkt[(df_mkt["game_date"] > TRAIN_END) & (df_mkt["game_date"] <= VAL_END)]
    ens_ts = df_mkt[(df_mkt["game_date"] > VAL_END) & (df_mkt["game_date"] <= TEST_END)]

    ens_cls = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=15,
        min_child_samples=50, reg_lambda=2.0,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=-1,
    )
    ens_cls.fit(
        ens_tr[ens_feats], ens_tr["home_win"].astype(int),
        eval_set=[(ens_vl[ens_feats], ens_vl["home_win"].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    p_mkt_vl = ens_vl["market_p_home"].clip(1e-6, 1 - 1e-6).values
    p_mdl_vl = ens_cls.predict_proba(ens_vl[ens_feats])[:, 1]
    y_vl_ens  = ens_vl["home_win"].astype(int).values

    best_w_acc, best_acc = 1.0, -1.0
    best_w_ll,  best_ll  = 1.0,  1e9
    for w in np.arange(0.0, 1.01, 0.05):
        pv = w * p_mkt_vl + (1 - w) * p_mdl_vl
        acc = ((pv > 0.5) == y_vl_ens).mean()
        ll  = log_loss(y_vl_ens, pv.clip(1e-6, 1 - 1e-6))
        if acc > best_acc:
            best_acc, best_w_acc = acc, w
        if ll < best_ll:
            best_ll, best_w_ll = ll, w

    blend_val = best_w_ll * p_mkt_vl + (1 - best_w_ll) * p_mdl_vl
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(blend_val, y_vl_ens)

    if len(ens_ts) >= 10:
        p_mkt_ts = ens_ts["market_p_home"].clip(1e-6, 1 - 1e-6).values
        p_mdl_ts = ens_cls.predict_proba(ens_ts[ens_feats])[:, 1]
        y_ts_ens  = ens_ts["home_win"].astype(int).values
        print("  test (ensemble blend, 2026 season):")
        pf = best_w_acc * p_mkt_ts + (1 - best_w_acc) * p_mdl_ts
        v2_ens_test = _eval_cls(pf, y_ts_ens, f"blend w={best_w_acc:.2f}")
    else:
        v2_ens_test = v2_cls_test
        print(f"  [ensemble test skipped — only {len(ens_ts)} market games in test period]")

    print(f"  blend weights  acc={best_w_acc:.2f}  ll={best_w_ll:.2f}")

    # ── Feature importances ────────────────────────────────────────────────────
    print("\n[4/4] Top features (v2 classifier)")
    fi = pd.Series(cls.feature_importances_, index=feats).sort_values(ascending=False)
    print(fi.head(15).to_string())

    # ── Save v2 artifacts ──────────────────────────────────────────────────────
    with open(MODELS_V2 / "lgb_cls.pkl", "wb") as f:
        pickle.dump({"model": cls, "calibrator": cal, "feature_names": feats}, f)
    with open(MODELS_V2 / "lgb_reg.pkl", "wb") as f:
        pickle.dump({"model": reg, "feature_names": feats}, f)
    with open(MODELS_V2 / "ensemble.pkl", "wb") as f:
        pickle.dump({
            "model": ens_cls,
            "feature_names": ens_feats,
            "blend_weight_market": float(best_w_acc),
            "blend_weight_market_ll": float(best_w_ll),
            "isotonic_calibrator": iso,
        }, f)

    report_v2 = {
        "version": "v2",
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "test_end": TEST_END,
        "n_train": n_tr, "n_val": n_vl, "n_test": n_ts,
        "features": feats,
        "lgb_cls_test_calibrated": v2_cls_test,
        "ensemble_test": v2_ens_test,
        "top_features_cls": fi.head(30).to_dict(),
    }
    with open(MODELS_V2 / "report.json", "w") as f:
        json.dump(report_v2, f, indent=2, default=str)

    print(f"\nsaved v2 artifacts to {MODELS_V2}")

    # ── Compare v1 vs v2 ──────────────────────────────────────────────────────
    v1_report = _load_v1_report()
    v1_test = v1_report.get("lgb_cls_test_calibrated", {})
    v1_acc = v1_test.get("accuracy")
    v2_acc = v2_cls_test["accuracy"]

    print(f"\n{'='*60}")
    print("  V1 vs V2 comparison (test-set accuracy)")
    print(f"{'='*60}")
    if v1_acc is not None:
        delta = (v2_acc - v1_acc) * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  v1 (test 2025-08 to 2025-12): {v1_acc:.4f}")
        print(f"  v2 (test {VAL_END} to today): {v2_acc:.4f}   ({sign}{delta:.2f}pp)")
        print()
        if delta >= 0:
            print("  v2 >= v1 on test set.")
        else:
            print("  v2 < v1 on test set — keep production as-is.")
        print(f"  NOTE: test periods differ (2025 vs 2026), so this is indicative only.")
        print(f"  For a head-to-head comparison, run walk-forward on 2026 months:")
        print(f"    python -m src.model.walkforward --start 2026-01 --end 2026-06")
    else:
        print(f"  v2 test accuracy: {v2_acc:.4f}")
        print("  (v1 report.json not found — no comparison available)")
    print(f"{'='*60}\n")

    # ── Optional deploy ────────────────────────────────────────────────────────
    if args.deploy:
        if v1_acc is None or v2_acc >= v1_acc:
            print("  Deploying v2 -> data/models/ ...")
            # Backup v1 first
            backup = MODELS_V1 / "v1_backup"
            backup.mkdir(exist_ok=True)
            for fname in ("ensemble.pkl", "lgb_cls.pkl", "lgb_reg.pkl", "report.json"):
                src = MODELS_V1 / fname
                if src.exists():
                    shutil.copy2(src, backup / fname)
            # Copy v2 -> production
            for fname in ("ensemble.pkl", "lgb_cls.pkl", "lgb_reg.pkl", "report.json"):
                src = MODELS_V2 / fname
                if src.exists():
                    shutil.copy2(src, MODELS_V1 / fname)
            print(f"  Deployed. v1 backup at {backup}")
            print("  Restart the server to use the new model.")
        else:
            print(f"  --deploy skipped: v2 ({v2_acc:.4f}) < v1 ({v1_acc:.4f})")


if __name__ == "__main__":
    main()
