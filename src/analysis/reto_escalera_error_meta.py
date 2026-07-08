"""Nested walk-forward error model for selectively flipping Reto picks.

The base predictions are already out-of-sample.  For every outer month this
script trains a second model only on earlier base predictions and learns a
flip threshold only on the tail of that earlier history.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .reto_escalera_backtest import OUT, PRED_COL, _ladder_counts, _universe
from ..normalize.paths import PROCESSED


NUM = [
    PRED_COL, "market_p_home", "model_conf", "market_conf", "model_market_gap",
    "abs_model_market_gap", "agree", "base_home", "dec_base", "dec_opposite",
    "month_sin", "month_cos",
]
CAT = ["home_team_abbrev", "away_team_abbrev"]
CONTEXT = [
    "elo_diff_pre", "win_pct_l30_diff", "run_diff_l10_diff",
    "off_xwoba_l30_diff", "off_barrel_rate_l30_diff", "off_k_pct_l30_diff",
    "def_xwoba_l30_diff", "def_barrel_rate_l30_diff",
    "starter_xwoba_l15_diff", "starter_barrel_against_l15_diff",
    "starter_k_pct_l15_diff", "starter_bb_pct_l15_diff", "starter_pa_l15_diff",
    "bullpen_ip_l5d_diff", "bullpen_apps_l3d_diff", "bullpen_runs_l5_diff",
    "lineup_top4_recent_ops_l15_diff", "lineup_top4_recent_iso_l15_diff",
    "lineup_top4_recent_k_pct_l15_diff", "lineup_recent_n_with_data_diff",
    "days_since_last_game_diff", "market_n_providers", "market_p_home_std",
    "market_line_shift_home_pp", "market_line_shift_abs_pp",
]


def _prepare() -> pd.DataFrame:
    d = _universe().copy()
    train = pd.read_parquet(PROCESSED / "train.parquet", columns=["game_pk"] + CONTEXT)
    # Some market columns are already present in the prediction universe.
    add = [c for c in CONTEXT if c not in d.columns]
    d = d.merge(train[["game_pk"] + add].drop_duplicates("game_pk"), on="game_pk", how="left")
    d["base_home"] = d[PRED_COL] >= .5
    d["base_won"] = np.where(d.base_home, d.home_win == 1, d.home_win == 0)
    d["wrong"] = (~d.base_won).astype(int)
    d["model_conf"] = np.where(d.base_home, d[PRED_COL], 1-d[PRED_COL])
    d["market_conf"] = np.where(d.base_home, d.mkt_h, d.mkt_a)
    d["model_market_gap"] = d["model_conf"] - d["market_conf"]
    d["abs_model_market_gap"] = (d[PRED_COL] - d.market_p_home).abs()
    d["agree"] = (d.base_home == (d.market_p_home >= .5)).astype(int)
    d["dec_base"] = np.where(d.base_home, d.dec_h, d.dec_a)
    d["dec_opposite"] = np.where(d.base_home, d.dec_a, d.dec_h)
    month = d.game_date.dt.month
    d["month_sin"] = np.sin(2*np.pi*month/12)
    d["month_cos"] = np.cos(2*np.pi*month/12)
    return d.dropna(subset=NUM + CAT + ["wrong"]).sort_values("game_date")


def _model(kind: str = "logistic"):
    if kind == "lgb_context":
        return lgb.LGBMClassifier(
            n_estimators=240, learning_rate=.025, num_leaves=7,
            min_child_samples=80, reg_lambda=3.0, reg_alpha=.5,
            colsample_bytree=.75, random_state=42, verbosity=-1,
        )
    if kind == "hist":
        return HistGradientBoostingClassifier(
            learning_rate=.04, max_iter=160, max_leaf_nodes=7,
            min_samples_leaf=40, l2_regularization=2.0, random_state=42,
        )
    if kind == "forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            max_features=.7, class_weight="balanced", random_state=42, n_jobs=-1,
        )
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(C=.25, max_iter=2000, solver="liblinear")),
    ])


def _x(d: pd.DataFrame, kind: str):
    if kind == "logistic":
        return d[NUM+CAT]
    if kind == "lgb_context":
        return d[NUM+CONTEXT]
    return d[NUM]


def _recipe_from_past(train: pd.DataFrame) -> tuple[str, float]:
    cutoff = train.game_date.max() - pd.Timedelta(days=75)
    fit, val = train[train.game_date <= cutoff], train[train.game_date > cutoff]
    if len(fit) < 500 or len(val) < 100:
        return "logistic", 1.1
    daily_idx = _base_daily_rows(val).index
    base = val.loc[daily_idx, "base_won"].to_numpy(bool)
    # Baseline is a candidate. Improvements must occur on the actual daily Reto
    # picks in the past-validation tail, not merely across every MLB game.
    best = (float(base.mean()), "logistic", 1.1)
    for kind in ("logistic", "hist", "forest", "lgb_context"):
        m = _model(kind).fit(_x(fit, kind), fit.wrong)
        pw = m.predict_proba(_x(val, kind))[:, 1]
        pw_daily = pd.Series(pw, index=val.index).loc[daily_idx].to_numpy()
        for threshold in np.arange(.45, .701, .025):
            flip = pw_daily >= threshold
            if flip.sum() < 5:
                continue
            acc = float(np.where(flip, ~base, base).mean())
            candidate = (acc, kind, float(threshold))
            if candidate > best:
                best = candidate
    return best[1], best[2]


def _outer_predictions(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in sorted(d.loc[d.game_date >= "2025-04-01", "game_date"].dt.to_period("M").unique()):
        start, end = period.start_time, period.end_time
        train = d[d.game_date < start]
        test = d[d.game_date.between(start, end)]
        if len(train) < 800 or test.empty:
            continue
        kind, threshold = _recipe_from_past(train)
        m = _model(kind).fit(_x(train, kind), train.wrong)
        out = test.copy()
        out["p_wrong"] = m.predict_proba(_x(test, kind))[:, 1]
        out["meta_kind"] = kind
        out["flip_threshold"] = threshold
        out["flipped"] = out.p_wrong >= threshold
        out["final_home"] = np.where(out.flipped, ~out.base_home, out.base_home)
        out["final_won"] = np.where(out.flipped, ~out.base_won, out.base_won)
        out["final_decimal"] = np.where(out.flipped, out.dec_opposite, out.dec_base)
        out["final_conf"] = np.where(out.flipped, out.p_wrong, 1-out.p_wrong)
        out["outer_month"] = str(period)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _summary(d: pd.DataFrame, label: str) -> dict:
    flips = d.flipped.astype(bool)
    row = {
        "period": label, "games": int(len(d)), "flips": int(flips.sum()),
        "flip_rate": float(flips.mean()),
        "base_accuracy": float(d.base_won.mean()),
        "meta_accuracy": float(d.final_won.mean()),
        "accuracy_delta_pp": float((d.final_won.mean()-d.base_won.mean())*100),
        "flipped_subset_base_accuracy": float(d.loc[flips, "base_won"].mean()) if flips.any() else None,
    }
    return row


def _daily(d: pd.DataFrame, corrected: bool) -> dict:
    if corrected:
        x = d[d.final_decimal.between(1.8, 2.2)].sort_values(
            ["game_date", "final_conf"], ascending=[True, False]).drop_duplicates("game_date")
        won, dec = x.final_won.to_numpy(bool), x.final_decimal.to_numpy(float)
    else:
        x = d[d.dec_base.between(1.8, 2.2)].sort_values(
            ["game_date", "model_conf"], ascending=[True, False]).drop_duplicates("game_date")
        won, dec = x.base_won.to_numpy(bool), x.dec_base.to_numpy(float)
    out = {"days": len(x), "accuracy": float(won.mean()),
           "flat_roi": float(np.where(won, dec-1, -1).mean())}
    for n in (3, 5, 10):
        starts, completed = _ladder_counts(won, n)
        out[f"ladder_{n}"] = {"started": starts, "completed": completed,
                              "rate": completed/starts if starts else 0.0}
    return out


def _daily_same_games_flipped(d: pd.DataFrame) -> dict:
    """Select the normal Reto game first, then flip that exact pick if flagged."""
    x = d[d.dec_base.between(1.8, 2.2)].sort_values(
        ["game_date", "model_conf"], ascending=[True, False]).drop_duplicates("game_date")
    won = x.final_won.to_numpy(bool)
    dec = x.final_decimal.to_numpy(float)
    out = {
        "days": len(x), "flips": int(x.flipped.sum()),
        "accuracy": float(won.mean()),
        "flat_roi": float(np.where(won, dec-1, -1).mean()),
        "final_odds_in_original_band_rate": float(x.final_decimal.between(1.8, 2.2).mean()),
    }
    for n in (3, 5, 10):
        starts, completed = _ladder_counts(won, n)
        out[f"ladder_{n}"] = {"started": starts, "completed": completed,
                              "rate": completed/starts if starts else 0.0}
    return out


def _base_daily_rows(d: pd.DataFrame) -> pd.DataFrame:
    return d[d.dec_base.between(1.8, 2.2)].sort_values(
        ["game_date", "model_conf"], ascending=[True, False]).drop_duplicates("game_date")


def _best_daily_threshold(history: pd.DataFrame) -> float:
    """Choose only from already-resolved daily picks; never from current month."""
    h = _base_daily_rows(history)
    if len(h) < 120:
        return 1.1
    base = h.base_won.to_numpy(bool)
    best = (float(base.mean()), 1.1)
    for threshold in np.arange(.45, .701, .025):
        flip = h.p_wrong.to_numpy() >= threshold
        if flip.sum() < 15:
            continue
        acc = float(np.where(flip, ~base, base).mean())
        # Accuracy first; on ties prefer fewer interventions.
        candidate = (acc, float(threshold))
        if candidate > best:
            best = candidate
    return best[1]


def _daily_calibrated(pred: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    selected = []
    for period in sorted(evaluation.game_date.dt.to_period("M").unique()):
        start, end = period.start_time, period.end_time
        threshold = _best_daily_threshold(pred[pred.game_date < start])
        month = _base_daily_rows(evaluation[evaluation.game_date.between(start, end)]).copy()
        month["daily_threshold"] = threshold
        month["daily_flip"] = month.p_wrong >= threshold
        month["daily_won"] = np.where(month.daily_flip, ~month.base_won, month.base_won)
        month["daily_decimal"] = np.where(month.daily_flip, month.dec_opposite, month.dec_base)
        selected.append(month)
    x = pd.concat(selected, ignore_index=True)
    won = x.daily_won.to_numpy(bool)
    dec = x.daily_decimal.to_numpy(float)
    out = {
        "days": len(x), "flips": int(x.daily_flip.sum()),
        "accuracy": float(won.mean()),
        "flat_roi": float(np.where(won, dec-1, -1).mean()),
        "thresholds": {str(k): float(v) for k, v in x.groupby(x.game_date.dt.to_period("M"))["daily_threshold"].first().items()},
    }
    for n in (3, 5, 10):
        starts, completed = _ladder_counts(won, n)
        out[f"ladder_{n}"] = {"started": starts, "completed": completed,
                              "rate": completed/starts if starts else 0.0}
    return out


def main() -> None:
    pred = _outer_predictions(_prepare())
    dev = pred[pred.game_date < "2026-01-01"]
    hold = pred[pred.game_date >= "2026-01-01"]
    payload = {
        "protocol": "nested monthly walk-forward; threshold learned only from prior 75-day validation tail",
        "game_level": [_summary(dev, "2025 development"), _summary(hold, "2026 holdout")],
        "reto_holdout_2026": {
            "base": _daily(hold, False),
            "same_games_then_flip": _daily_same_games_flipped(hold),
            "daily_calibrated_flip": _daily_calibrated(pred, hold),
            "meta_reselect": _daily(hold, True),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(OUT / "error_meta_predictions.parquet", index=False)
    (OUT / "error_meta_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
