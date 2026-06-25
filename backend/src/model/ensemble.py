"""Final ensemble: weighted blend of raw market and lineup-aware LightGBM.

We tune the blend weight `w` on val:
  p_final = w * p_market_raw + (1-w) * p_model

Then evaluate on test with the tuned weight. Saves the artifacts so predict.py
can use them.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..normalize.paths import PROCESSED
from .dataset import EXCLUDE_COLS

MODELS = PROCESSED.parent / "models"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].notna().any()]


def _train_lgb(Xtr, ytr, Xvl, yvl):
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=15,
        min_child_samples=50, reg_lambda=2.0,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def _eval(name, y, p):
    return {
        "name": name,
        "log_loss": float(log_loss(y, p.clip(1e-6, 1 - 1e-6))),
        "AUC":      float(roc_auc_score(y, p)),
        "accuracy": float(((p > 0.5) == y).mean()),
        "Brier":    float(brier_score_loss(y, p)),
    }


def main() -> None:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["home_win", "total_runs", "market_p_home"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    feats = feature_columns(df)
    print(f"using {len(feats)} features, {len(df):,} games\n")

    tr = df[df["game_date"] <= "2024-12-31"]
    vl = df[(df["game_date"] > "2024-12-31") & (df["game_date"] <= "2025-07-31")]
    ts = df[(df["game_date"] > "2025-07-31") & (df["game_date"] <= "2025-12-31")]
    print(f"train {len(tr):,}  val {len(vl):,}  test {len(ts):,}\n")

    model = _train_lgb(tr[feats], tr["home_win"].astype(int),
                       vl[feats], vl["home_win"].astype(int))

    p_mkt_vl = vl["market_p_home"].clip(1e-6, 1 - 1e-6).values
    p_mkt_ts = ts["market_p_home"].clip(1e-6, 1 - 1e-6).values
    p_mdl_vl = model.predict_proba(vl[feats])[:, 1]
    p_mdl_ts = model.predict_proba(ts[feats])[:, 1]
    y_vl = vl["home_win"].astype(int).values
    y_ts = ts["home_win"].astype(int).values

    print("=== Standalone (val + test) ===")
    for name, pv, pt in [("market raw", p_mkt_vl, p_mkt_ts),
                          ("lightgbm",   p_mdl_vl, p_mdl_ts)]:
        print(f"  VAL  {name:12s}: ll={log_loss(y_vl, pv):.4f}  "
              f"AUC={roc_auc_score(y_vl, pv):.4f}  acc={((pv>0.5)==y_vl).mean():.4f}")
        print(f"  TEST {name:12s}: ll={log_loss(y_ts, pt):.4f}  "
              f"AUC={roc_auc_score(y_ts, pt):.4f}  acc={((pt>0.5)==y_ts).mean():.4f}")

    print("\n=== Blend sweep (val) ===")
    best_w_ll, best_ll = None, 1e9
    best_w_acc, best_acc = None, -1.0
    sweep = []
    for w in np.arange(0.0, 1.01, 0.05):
        pv = w * p_mkt_vl + (1 - w) * p_mdl_vl
        ll = log_loss(y_vl, pv.clip(1e-6, 1 - 1e-6))
        auc = roc_auc_score(y_vl, pv)
        acc = ((pv > 0.5) == y_vl).mean()
        sweep.append((w, ll, auc, acc))
        if ll < best_ll:
            best_ll = ll; best_w_ll = w
        if acc > best_acc:
            best_acc = acc; best_w_acc = w
    for w, ll, auc, acc in sweep:
        flags = []
        if w == best_w_ll: flags.append("best ll")
        if w == best_w_acc: flags.append("best acc")
        f = ("  <-- " + ", ".join(flags)) if flags else ""
        print(f"  w={w:.2f}  ll={ll:.4f}  AUC={auc:.4f}  acc={acc:.4f}{f}")

    # ────────────────────────────────────────────────────────────────────────
    # Isotonic calibration on top of the best-accuracy blend.
    # Error analysis showed the blended output is OVER-confident on heavy
    # favorites (predicted 95% → actual 60%). Isotonic maps each predicted
    # bucket back to the empirical win rate seen on val.
    # ────────────────────────────────────────────────────────────────────────
    print("\n=== Isotonic calibration on best-ll blend ===")
    blend_val_for_cal = best_w_ll * p_mkt_vl + (1 - best_w_ll) * p_mdl_vl
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(blend_val_for_cal, y_vl)
    pv_cal = iso.transform(blend_val_for_cal)
    print(f"  val ll  raw={log_loss(y_vl, blend_val_for_cal.clip(1e-6,1-1e-6)):.4f}  "
          f"cal={log_loss(y_vl, pv_cal.clip(1e-6,1-1e-6)):.4f}")

    print(f"\n=== TEST evaluation ===")
    print(f"(blend formula: p = w*market + (1-w)*model)")
    for label, w in [("market only (w=1.00)", 1.00),
                      (f"blend best log_loss (w={best_w_ll:.2f})", best_w_ll),
                      (f"blend best accuracy (w={best_w_acc:.2f})", best_w_acc),
                      ("model only (w=0.00)", 0.00)]:
        pf_ts = w * p_mkt_ts + (1 - w) * p_mdl_ts
        x = _eval(label, y_ts, pf_ts)
        print(f"  {label:35s}  ll={x['log_loss']:.4f}  AUC={x['AUC']:.4f}  "
              f"acc={x['accuracy']:.4f}  Brier={x['Brier']:.4f}")
    # Best-LL blend with isotonic calibration on top
    pf_ts_raw = best_w_ll * p_mkt_ts + (1 - best_w_ll) * p_mdl_ts
    pf_ts_cal = iso.transform(pf_ts_raw)
    x = _eval(f"blend w={best_w_ll:.2f} + isotonic", y_ts, pf_ts_cal)
    print(f"  {'blend best ll + isotonic cal':35s}  ll={x['log_loss']:.4f}  "
          f"AUC={x['AUC']:.4f}  acc={x['accuracy']:.4f}  Brier={x['Brier']:.4f}")
    best_w = best_w_acc  # save the accuracy-optimal blend
    print(f"\nsaving with accuracy-optimal weight: {best_w:.2f}, ll-optimal weight: {best_w_ll:.2f} + iso cal")

    # Save bundle: include isotonic calibrator and BOTH blend weights so the
    # API can use either acc-optimal blend (for picks) or ll-optimal+iso (for
    # well-calibrated probabilities exposed to users).
    with open(MODELS / "ensemble.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feats,
            "blend_weight_market": float(best_w),       # acc-optimal blend
            "blend_weight_market_ll": float(best_w_ll), # ll-optimal blend
            "isotonic_calibrator": iso,                  # fitted on val blend
        }, f)
    print(f"\nsaved {MODELS}/ensemble.pkl  (acc weight = {best_w:.2f}, ll weight = {best_w_ll:.2f}+iso)")

    # Also train + save a fresh total-runs regressor on the SAME feature set
    # so the API doesn't crash on stale feature counts.
    print("\n[totals] training fresh regressor on the same feature set…")
    reg = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, reg_lambda=1.0, subsample=0.85,
        colsample_bytree=0.85, random_state=42, verbosity=-1,
        objective="regression",
    )
    reg.fit(
        tr[feats], tr["total_runs"].astype(float),
        eval_set=[(vl[feats], vl["total_runs"].astype(float))],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_test = reg.predict(ts[feats])
    from sklearn.metrics import mean_absolute_error
    print(f"  test MAE: {mean_absolute_error(ts['total_runs'], pred_test):.3f}")
    with open(MODELS / "lgb_reg.pkl", "wb") as f:
        pickle.dump({"model": reg, "feature_names": feats}, f)
    print(f"saved {MODELS}/lgb_reg.pkl")


if __name__ == "__main__":
    main()
