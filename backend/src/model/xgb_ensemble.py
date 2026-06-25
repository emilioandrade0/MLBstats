"""Train an XGBoost classifier and ensemble it with LightGBM.

Different algorithms make different mistakes. Averaging their predictions
reduces variance and typically closes calibration gaps. Standard practice
in tabular ML competitions.

Output: data/models/xgb_cls.pkl  {model, feature_names}
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..normalize.paths import PROCESSED
from .dataset import EXCLUDE_COLS

MODELS = PROCESSED.parent / "models"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].notna().any()]


def main() -> None:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["home_win", "market_p_home"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    feats = feature_columns(df)

    tr = df[df["game_date"] <= "2024-12-31"]
    vl = df[(df["game_date"] > "2024-12-31") & (df["game_date"] <= "2025-07-31")]
    ts = df[(df["game_date"] > "2025-07-31") & (df["game_date"] <= "2025-12-31")]
    print(f"train {len(tr):,}  val {len(vl):,}  test {len(ts):,}  features {len(feats)}\n")

    Xtr = tr[feats].astype(float).values; ytr = tr["home_win"].astype(int).values
    Xvl = vl[feats].astype(float).values; yvl = vl["home_win"].astype(int).values
    Xts = ts[feats].astype(float).values; yts = ts["home_win"].astype(int).values

    # XGBoost with sensible defaults for tabular MLB-scale data
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=4,
        min_child_weight=10, reg_lambda=2.0, reg_alpha=0.0,
        subsample=0.85, colsample_bytree=0.85, gamma=0.1,
        random_state=42, eval_metric="logloss",
        early_stopping_rounds=50, verbosity=0, n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)

    p_val = model.predict_proba(Xvl)[:, 1]
    p_test = model.predict_proba(Xts)[:, 1]
    print(f"XGB validation:  ll={log_loss(yvl, p_val):.4f}  AUC={roc_auc_score(yvl, p_val):.4f}  acc={((p_val>0.5)==yvl).mean():.4f}")
    print(f"XGB test:        ll={log_loss(yts, p_test):.4f}  AUC={roc_auc_score(yts, p_test):.4f}  acc={((p_test>0.5)==yts).mean():.4f}")

    # Compare to LGB
    with open(MODELS / "ensemble.pkl", "rb") as f:
        lgb_bundle = pickle.load(f)
    p_lgb_test = lgb_bundle["model"].predict_proba(ts[feats])[:, 1]
    print(f"LGB test (raw):  ll={log_loss(yts, p_lgb_test):.4f}  AUC={roc_auc_score(yts, p_lgb_test):.4f}  acc={((p_lgb_test>0.5)==yts).mean():.4f}")

    # Simple average ensemble
    p_avg = (p_test + p_lgb_test) / 2
    print(f"LGB+XGB avg:     ll={log_loss(yts, p_avg):.4f}  AUC={roc_auc_score(yts, p_avg):.4f}  acc={((p_avg>0.5)==yts).mean():.4f}")

    with open(MODELS / "xgb_cls.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_names": feats}, f)
    print(f"\nsaved {MODELS}/xgb_cls.pkl")


if __name__ == "__main__":
    main()
