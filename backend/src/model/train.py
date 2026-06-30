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


def _train_lgb_classifier(X_train, y_train, X_val, y_val) -> tuple[lgb.LGBMClassifier, dict]:
    # Hyperparameters tuned via walk-forward (src/model/hyperparam_sweep.py):
    # num_leaves=10 + reg=1.0 + team_id categorical wins the sweet band
    # (10-20pp confidence, where +EV picks live):
    #   - sweet band accuracy:  56.19% (best of 16 configs tested)
    #   - log_loss:             0.6941 (best calibration)
    #   - AUC:                  0.5625
    # Trades 0.4pp overall acc vs leaves=16 for +2.3pp on the band that
    # actually generates +EV bets.
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=10,
        min_child_samples=40,
        reg_lambda=1.0,
        reg_alpha=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1,
    )
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        categorical_feature=cat_feats if cat_feats else "auto",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_val = model.predict_proba(X_val)[:, 1]
    metrics = {
        "log_loss": float(log_loss(y_val, p_val)),
        "brier": float(brier_score_loss(y_val, p_val)),
        "auc": float(roc_auc_score(y_val, p_val)),
        "accuracy": float(((p_val > 0.5) == y_val).mean()),
        "best_iter": int(model.best_iteration_) if model.best_iteration_ else 0,
        "mean_pred": float(p_val.mean()),
        "base_rate": float(y_val.mean()),
    }
    return model, metrics


def _train_lgb_regressor(X_train, y_train, X_val, y_val) -> tuple[lgb.LGBMRegressor, dict]:
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=10,
        min_child_samples=40,
        reg_lambda=1.0,
        reg_alpha=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        objective="regression",
        verbosity=-1,
    )
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        categorical_feature=cat_feats if cat_feats else "auto",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_val = model.predict(X_val)
    metrics = {
        "mae": float(mean_absolute_error(y_val, p_val)),
        "rmse": float(mean_squared_error(y_val, p_val) ** 0.5),
        "mean_pred": float(p_val.mean()),
        "mean_actual": float(y_val.mean()),
        "best_iter": int(model.best_iteration_) if model.best_iteration_ else 0,
    }
    return model, metrics


def _calibrate(model: lgb.LGBMClassifier, X_val, y_val) -> CalibratedClassifierCV:
    """Isotonic calibration on the val set so probabilities match observed frequency."""
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    cal.fit(X_val, y_val)
    return cal


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
