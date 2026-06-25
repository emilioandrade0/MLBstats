"""Train F5 (first-5-innings) classifier and total-runs regressor.

Same feature set as full-game models, but targets are:
  - f5_home_won: 1 if home led after 5, else 0 (ties count as 0)
  - f5_total_runs: total runs scored across innings 1-5

We do NOT blend with market here — ESPN's odds payload doesn't include F5
markets so we have no market_p_home_f5 to use as prior. The pure LGB
prediction is what we publish, and we surface it as "fair odds" for F5.

Artifacts:
  data/models/f5_cls.pkl   {model, feature_names}
  data/models/f5_reg.pkl   {model, feature_names}
  data/models/f5_report.json
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from ..normalize.paths import PROCESSED
from .dataset import EXCLUDE_COLS

MODELS = PROCESSED.parent / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].notna().any()]


def _split(df: pd.DataFrame, train_end: str, val_end: str, test_end: str):
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    te = pd.Timestamp(train_end); ve = pd.Timestamp(val_end); se = pd.Timestamp(test_end)
    return (
        df[df["game_date"] <= te],
        df[(df["game_date"] > te) & (df["game_date"] <= ve)],
        df[(df["game_date"] > ve) & (df["game_date"] <= se)],
    )


def _train_lgb_cls(Xtr, ytr, Xvl, yvl) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=15,
        min_child_samples=50, reg_lambda=2.0,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def _train_lgb_reg(Xtr, ytr, Xvl, yvl) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=15,
        min_child_samples=50, reg_lambda=2.0,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=-1, objective="regression",
    )
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], eval_metric="l2",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def main() -> None:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["f5_home_won", "f5_total_runs"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    feats = _feature_columns(df)
    print(f"using {len(feats)} features over {len(df):,} games with F5 targets")

    tr, vl, ts = _split(df, "2024-12-31", "2025-07-31", "2025-12-31")
    print(f"train {len(tr):,}  val {len(vl):,}  test {len(ts):,}")
    print(f"\nF5 base rates:")
    print(f"  home wins F5:  {df['f5_home_won'].mean():.3f}")
    print(f"  tied at 5:     {df['f5_tied'].mean():.3f}")
    print(f"  avg total F5:  {df['f5_total_runs'].mean():.2f}\n")

    # --- F5 classifier (home wins F5) ---
    print("[F5 cls] training…")
    ytr = tr["f5_home_won"].astype(int)
    yvl = vl["f5_home_won"].astype(int)
    yts = ts["f5_home_won"].astype(int)
    cls = _train_lgb_cls(tr[feats], ytr, vl[feats], yvl)
    p_val = cls.predict_proba(vl[feats])[:, 1]
    p_ts = cls.predict_proba(ts[feats])[:, 1]
    cls_val = {
        "log_loss": float(log_loss(yvl, p_val.clip(1e-6, 1-1e-6))),
        "auc": float(roc_auc_score(yvl, p_val)),
        "acc": float(((p_val > 0.5) == yvl).mean()),
        "brier": float(brier_score_loss(yvl, p_val)),
        "best_iter": int(cls.best_iteration_ or 0),
    }
    cls_test = {
        "log_loss": float(log_loss(yts, p_ts.clip(1e-6, 1-1e-6))),
        "auc": float(roc_auc_score(yts, p_ts)),
        "acc": float(((p_ts > 0.5) == yts).mean()),
        "brier": float(brier_score_loss(yts, p_ts)),
    }
    # Reference: predicting base rate
    base = ytr.mean()
    ref_test_ll = float(log_loss(yts, np.full_like(p_ts, base)))
    cls_test["ref_baserate_log_loss"] = ref_test_ll
    print(f"  val:  ll={cls_val['log_loss']:.4f}  AUC={cls_val['auc']:.4f}  acc={cls_val['acc']:.4f}")
    print(f"  test: ll={cls_test['log_loss']:.4f}  AUC={cls_test['auc']:.4f}  acc={cls_test['acc']:.4f}")
    print(f"  test reference (constant base rate): ll={ref_test_ll:.4f}")

    # --- F5 total runs regressor ---
    print("\n[F5 reg] training…")
    rtr = tr["f5_total_runs"].astype(float)
    rvl = vl["f5_total_runs"].astype(float)
    rts = ts["f5_total_runs"].astype(float)
    reg = _train_lgb_reg(tr[feats], rtr, vl[feats], rvl)
    pred_val = reg.predict(vl[feats])
    pred_ts = reg.predict(ts[feats])
    reg_val = {
        "mae": float(mean_absolute_error(rvl, pred_val)),
        "rmse": float(mean_squared_error(rvl, pred_val) ** 0.5),
        "best_iter": int(reg.best_iteration_ or 0),
    }
    reg_test = {
        "mae": float(mean_absolute_error(rts, pred_ts)),
        "rmse": float(mean_squared_error(rts, pred_ts) ** 0.5),
        "ref_mean_mae": float(mean_absolute_error(rts, np.full_like(pred_ts, rtr.mean()))),
    }
    print(f"  val:  MAE={reg_val['mae']:.3f}  RMSE={reg_val['rmse']:.3f}")
    print(f"  test: MAE={reg_test['mae']:.3f}  RMSE={reg_test['rmse']:.3f}")
    print(f"  test reference (constant mean): MAE={reg_test['ref_mean_mae']:.3f}")

    # --- Top features ---
    print("\n[F5 cls] top 15 features:")
    fi = pd.Series(cls.feature_importances_, index=feats).sort_values(ascending=False)
    print(fi.head(15).to_string())

    # --- Save ---
    with open(MODELS / "f5_cls.pkl", "wb") as f:
        pickle.dump({"model": cls, "feature_names": feats}, f)
    with open(MODELS / "f5_reg.pkl", "wb") as f:
        pickle.dump({"model": reg, "feature_names": feats}, f)
    report = {"cls_val": cls_val, "cls_test": cls_test,
              "reg_val": reg_val, "reg_test": reg_test,
              "top_features": fi.head(30).to_dict()}
    with open(MODELS / "f5_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nartifacts saved to {MODELS}")


if __name__ == "__main__":
    main()
