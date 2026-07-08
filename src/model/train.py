"""Train the home-win classifier + total-runs regressor.

Two models:
  1. LogisticRegression baseline (interpretable, sanity check)
  2. LightGBM (main model — handles NaN, non-linear, very strong on tabular)

LightGBM is trained with early stopping on the validation set.
We then calibrate the classifier with isotonic regression on the val set so
output probabilities can be compared to implied market probabilities.

Artifacts saved to data/models/.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from .seed_ensemble import SEEDS, SeedEnsemble
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..normalize.paths import PROCESSED
from .dataset import load_split, CATEGORICAL_FEATURES

MODELS = PROCESSED.parent / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def _baseline_logreg(X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series) -> tuple[Pipeline, dict]:
    """Logistic regression on imputed/standardized features. Interpretable baseline."""
    Xt = X_train.fillna(X_train.median(numeric_only=True))
    Xv = X_val.fillna(X_train.median(numeric_only=True))
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipe.fit(Xt, y_train)
    p_val = pipe.predict_proba(Xv)[:, 1]
    metrics = {
        "log_loss": float(log_loss(y_val, p_val)),
        "brier": float(brier_score_loss(y_val, p_val)),
        "auc": float(roc_auc_score(y_val, p_val)),
        "accuracy": float(((p_val > 0.5) == y_val).mean()),
        "mean_pred": float(p_val.mean()),
        "base_rate": float(y_val.mean()),
    }
    return pipe, metrics


def _train_lgb_classifier(X_train, y_train, X_val, y_val) -> tuple[SeedEnsemble, dict]:
    """Train N seed models and return a SeedEnsemble.

    Walk-forward gain vs single-seed (n=5634): +0.06pp acc, +0.07pp AUC.
    Small but consistent noise reduction — worth the Nx training time since
    train.py runs once per deploy.
    """
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    models = []
    for seed in SEEDS:
        m = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=10,
            min_child_samples=40,
            reg_lambda=1.0,
            reg_alpha=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            verbosity=-1,
        )
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            categorical_feature=cat_feats if cat_feats else "auto",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        models.append(m)
    ens = SeedEnsemble(models, "cls")
    p_val = ens.predict_proba(X_val)[:, 1]
    metrics = {
        "log_loss": float(log_loss(y_val, p_val)),
        "brier": float(brier_score_loss(y_val, p_val)),
        "auc": float(roc_auc_score(y_val, p_val)),
        "accuracy": float(((p_val > 0.5) == y_val).mean()),
        "n_seeds": len(SEEDS),
        "mean_pred": float(p_val.mean()),
        "base_rate": float(y_val.mean()),
    }
    return ens, metrics


def _train_lgb_regressor(X_train, y_train, X_val, y_val) -> tuple[SeedEnsemble, dict]:
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    models = []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=10,
            min_child_samples=40,
            reg_lambda=1.0,
            reg_alpha=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            objective="regression",
            verbosity=-1,
        )
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            categorical_feature=cat_feats if cat_feats else "auto",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        models.append(m)
    ens = SeedEnsemble(models, "reg")
    p_val = ens.predict(X_val)
    metrics = {
        "mae": float(mean_absolute_error(y_val, p_val)),
        "rmse": float(mean_squared_error(y_val, p_val) ** 0.5),
        "mean_pred": float(p_val.mean()),
        "mean_actual": float(y_val.mean()),
        "n_seeds": len(SEEDS),
    }
    return ens, metrics


def _calibrate(model, X_val, y_val) -> SeedEnsemble:
    """Isotonic-calibrate EACH seed model, return an ensemble of calibrators.

    Calibrating each seed then averaging gives better-calibrated probabilities
    than averaging then calibrating once, because each seed's raw output has
    its own miscalibration profile.
    """
    cals = []
    for m in model.models:
        c = CalibratedClassifierCV(FrozenEstimator(m), method="isotonic")
        c.fit(X_val, y_val)
        cals.append(c)
    return SeedEnsemble(cals, "cls")


def _holdout_eval(model, X_test, y_test, kind: str) -> dict:
    if kind == "cls":
        p = model.predict_proba(X_test)[:, 1]
        return {
            "log_loss": float(log_loss(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
            "auc": float(roc_auc_score(y_test, p)),
            "accuracy": float(((p > 0.5) == y_test).mean()),
            "mean_pred": float(p.mean()),
            "base_rate": float(y_test.mean()),
        }
    else:
        p = model.predict(X_test)
        return {
            "mae": float(mean_absolute_error(y_test, p)),
            "rmse": float(mean_squared_error(y_test, p) ** 0.5),
        }


def main() -> None:
    split = load_split()
    print(f"train: {len(split.X_train):,}  val: {len(split.X_val):,}  test: {len(split.X_test):,}")
    print(f"features: {len(split.feature_names)}")

    report: dict = {"features": split.feature_names}

    # ---- Logistic baseline (home-win) ----
    print("\n[1/4] LogReg baseline (home-win)")
    pipe, m = _baseline_logreg(split.X_train, split.y_train_cls,
                                 split.X_val, split.y_val_cls)
    report["logreg_val"] = m
    print({k: round(v, 4) for k, v in m.items()})

    # ---- LightGBM home-win ----
    print("\n[2/4] LightGBM home-win")
    lgb_cls, m = _train_lgb_classifier(split.X_train, split.y_train_cls,
                                        split.X_val, split.y_val_cls)
    report["lgb_cls_val"] = m
    print({k: round(v, 4) for k, v in m.items()})

    cal = _calibrate(lgb_cls, split.X_val, split.y_val_cls)
    test_cls = _holdout_eval(cal, split.X_test, split.y_test_cls, "cls")
    report["lgb_cls_test_calibrated"] = test_cls
    print("test (calibrated):", {k: round(v, 4) for k, v in test_cls.items()})

    # ---- LightGBM total-runs ----
    print("\n[3/4] LightGBM total-runs")
    lgb_reg, m = _train_lgb_regressor(split.X_train, split.y_train_reg,
                                        split.X_val, split.y_val_reg)
    report["lgb_reg_val"] = m
    print({k: round(v, 4) for k, v in m.items()})
    test_reg = _holdout_eval(lgb_reg, split.X_test, split.y_test_reg, "reg")
    report["lgb_reg_test"] = test_reg
    print("test:", {k: round(v, 4) for k, v in test_reg.items()})

    # ---- Top feature importances ----
    print("\n[4/4] Top feature importances")
    fi = pd.Series(lgb_cls.feature_importances_, index=split.feature_names).sort_values(ascending=False)
    print(fi.head(20).to_string())
    report["top_features_cls"] = fi.head(30).to_dict()

    # ---- Save artifacts ----
    with open(MODELS / "lgb_cls.pkl", "wb") as f:
        pickle.dump({"model": lgb_cls, "calibrator": cal,
                     "feature_names": split.feature_names}, f)
    with open(MODELS / "lgb_reg.pkl", "wb") as f:
        pickle.dump({"model": lgb_reg, "feature_names": split.feature_names}, f)
    with open(MODELS / "logreg.pkl", "wb") as f:
        pickle.dump({"model": pipe, "feature_names": split.feature_names,
                     "train_medians": split.X_train.median(numeric_only=True).to_dict()}, f)
    with open(MODELS / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nartifacts saved to {MODELS}")


if __name__ == "__main__":
    main()
