"""Ablation: what features actually move the needle?

Trains 5 LightGBM classifiers with different feature subsets and reports
val + test log_loss / AUC / accuracy. Tells us which feature group matters.

  A: market only           (baseline — pure market closing line)
  B: market + team form    (Statcast rolling per team)
  C: market + team + pitcher (starter rolling)
  D: market + team + pitcher + lineup (everything)
  E: statcast-only (no market)  — what's our model worth without the market?
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..normalize.paths import PROCESSED
from .dataset import EXCLUDE_COLS


MARKET_FEATS = {"market_logit_p_home", "market_over_under", "market_spread",
                "market_n_providers", "market_p_home_std"}
TEAM_PREFIXES = ("runs_scored_l10", "runs_allowed_l10", "run_diff_l10",
                 "win_pct_l30", "off_xwoba_l30", "off_barrel_rate_l30",
                 "off_k_pct_l30", "off_bb_pct_l30", "def_xwoba_l30",
                 "def_barrel_rate_l30", "def_k_pct_l30", "def_bb_pct_l30",
                 "days_since_last_game")
PITCHER_PREFIXES = ("starter_xwoba_l15", "starter_barrel_against_l15",
                    "starter_k_pct_l15", "starter_bb_pct_l15",
                    "starter_velo_mean_l15", "starter_spin_mean_l15",
                    "starter_pa_l15", "starter_days_rest")
LINEUP_PREFIXES = ("lineup_xwoba_vs_sp_hand", "lineup_k_pct_vs_sp_hand",
                   "lineup_bb_pct_vs_sp_hand", "lineup_barrel_vs_sp_hand",
                   "lineup_top4_xwoba_vs_sp_hand", "lineup_top4_k_pct_vs_sp_hand",
                   "lineup_n_with_data")
ELO_PREFIXES = ("home_elo_pre", "away_elo_pre", "elo_diff_pre", "expected_home_win_elo")
BULLPEN_PREFIXES = ("bullpen_ip_l1d", "bullpen_ip_l3d", "bullpen_ip_l5d",
                    "bullpen_apps_l3d", "bullpen_runs_l5")
SERIES_PREFIXES = ("series_game_number", "series_run_diff_so_far",
                   "did_home_win_previous_game_in_series")
ARSENAL_PREFIXES = ("lineup_xwoba_vs_arsenal", "lineup_top4_xwoba_vs_arsenal",
                    "lineup_arsenal_n_with_data")


def _by_prefix(cols: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in cols if any(c.startswith(p) for p in prefixes)]


def _feature_sets(all_feats: list[str]) -> dict[str, list[str]]:
    market = [c for c in all_feats if c in MARKET_FEATS]
    team   = _by_prefix(all_feats, TEAM_PREFIXES)
    pit    = _by_prefix(all_feats, PITCHER_PREFIXES)
    lin    = _by_prefix(all_feats, LINEUP_PREFIXES)
    elo_f  = _by_prefix(all_feats, ELO_PREFIXES)
    bp_f   = _by_prefix(all_feats, BULLPEN_PREFIXES)
    series = _by_prefix(all_feats, SERIES_PREFIXES)
    arsenal = _by_prefix(all_feats, ARSENAL_PREFIXES)
    grouped = set(market + team + pit + lin + elo_f + bp_f + series + arsenal)
    other  = [c for c in all_feats if c not in grouped]
    return {
        "A market-only":     market,
        "B + team form":     market + team,
        "C + pitcher":       market + team + pit,
        "D + lineup":        market + team + pit + lin,
        "E + ELO":           market + team + pit + lin + elo_f,
        "F + bullpen":       market + team + pit + lin + elo_f + bp_f,
        "G + series":        market + team + pit + lin + elo_f + bp_f + series,
        "H + arsenal (all)": market + team + pit + lin + elo_f + bp_f + series + arsenal + other,
        "I statcast-only":   team + pit + lin + elo_f + bp_f + series + arsenal + other,
    }


def _fit_score(Xtr, ytr, Xvl, yvl, Xts, yts) -> dict:
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=15,
        min_child_samples=50, reg_lambda=2.0,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    pv = m.predict_proba(Xvl)[:, 1]
    pt = m.predict_proba(Xts)[:, 1]
    return {
        "val_logloss": log_loss(yvl, pv),
        "val_auc":     roc_auc_score(yvl, pv),
        "val_acc":     ((pv > 0.5) == yvl).mean(),
        "test_logloss": log_loss(yts, pt),
        "test_auc":     roc_auc_score(yts, pt),
        "test_acc":     ((pt > 0.5) == yts).mean(),
        "test_brier":   brier_score_loss(yts, pt),
        "best_iter":    int(m.best_iteration_ or 0),
        "n_features":   Xtr.shape[1],
    }


def main() -> None:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["home_win", "total_runs"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")

    # All features (drop targets/meta)
    feats_all = [c for c in df.columns
                 if c not in EXCLUDE_COLS and df[c].notna().any()]
    sets = _feature_sets(feats_all)
    for k, v in sets.items():
        print(f"  {k}: {len(v)} features")

    tr = df[df["game_date"] <= "2024-12-31"]
    vl = df[(df["game_date"] > "2024-12-31") & (df["game_date"] <= "2025-07-31")]
    ts = df[(df["game_date"] > "2025-07-31") & (df["game_date"] <= "2025-12-31")]
    print(f"\ntrain {len(tr):,}  val {len(vl):,}  test {len(ts):,}\n")

    ytr = tr["home_win"].astype(int)
    yvl = vl["home_win"].astype(int)
    yts = ts["home_win"].astype(int)

    # Reference: pure market_p_home (no model) on test
    if "market_p_home" in df.columns:
        ts_m = ts.dropna(subset=["market_p_home"])
        if len(ts_m):
            pm = ts_m["market_p_home"].clip(1e-6, 1-1e-6)
            print(f"REFERENCE — raw market_p_home (no model) on test "
                  f"(n={len(ts_m):,}):")
            print(f"  log_loss={log_loss(ts_m['home_win'], pm):.4f}  "
                  f"AUC={roc_auc_score(ts_m['home_win'], pm):.4f}  "
                  f"acc={((pm>0.5)==ts_m['home_win']).mean():.4f}\n")

    results = []
    for name, fs in sets.items():
        m = _fit_score(tr[fs], ytr, vl[fs], yvl, ts[fs], yts)
        m["model"] = name
        results.append(m)

    rdf = pd.DataFrame(results).set_index("model")[
        ["n_features", "best_iter",
         "val_logloss", "val_auc", "val_acc",
         "test_logloss", "test_auc", "test_acc", "test_brier"]
    ]
    print("\n=== ABLATION RESULTS ===")
    print(rdf.round(4).to_string())


if __name__ == "__main__":
    main()
