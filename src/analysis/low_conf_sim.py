"""Validate the low_conf_market rule on walkforward.

If model's p_pick is in [0.50, 0.58) AND market disagrees, swap to market's pick.
Measure accuracy gain vs baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def main() -> None:
    df = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    df = df.dropna(subset=["p_home", "market_p_home", "home_win"]).copy()

    df["model_home"] = df["p_home"] >= 0.5
    df["mkt_home"]   = df["market_p_home"] >= 0.5
    df["p_pick_mod"] = np.maximum(df["p_home"], 1 - df["p_home"])

    df["model_correct"] = np.where(df["model_home"], df["home_win"] == 1, df["home_win"] == 0)
    df["mkt_correct"]   = np.where(df["mkt_home"],   df["home_win"] == 1, df["home_win"] == 0)

    LO, HI = 0.50, 0.58
    is_low_conf = (df["p_pick_mod"] >= LO) & (df["p_pick_mod"] < HI)
    disagrees = df["model_home"] != df["mkt_home"]
    override = is_low_conf & disagrees

    df["hybrid_correct"] = np.where(override, df["mkt_correct"], df["model_correct"])

    n = len(df)
    n_override = override.sum()
    base_acc  = df["model_correct"].mean()
    hyb_acc   = df["hybrid_correct"].mean()
    mkt_acc   = df["mkt_correct"].mean()

    print(f"n={n:,}  (overrides applied: {n_override:,} = {n_override/n*100:.1f}%)")
    print(f"  baseline model     acc={base_acc:.4f}")
    print(f"  market only        acc={mkt_acc:.4f}")
    print(f"  hybrid (override)  acc={hyb_acc:.4f}")
    print(f"  delta over base    = {(hyb_acc - base_acc)*100:+.2f}pp")

    # Per-fold breakdown to check stability
    print()
    print("Per fold (delta = hybrid - baseline):")
    print(f"{'fold':>16} {'n':>5} {'base':>7} {'hyb':>7} {'delta':>8} {'n_ov':>6}")
    for fold, g in df.groupby("fold_month"):
        nf = len(g)
        b  = g["model_correct"].mean()
        ov = (g["p_pick_mod"].between(LO, HI, inclusive="left") &
              (g["model_home"] != g["mkt_home"]))
        h  = np.where(ov, g["mkt_correct"], g["model_correct"]).mean()
        print(f"{str(fold):>16} {nf:>5} {b*100:>6.2f}% {h*100:>6.2f}% "
              f"{(h-b)*100:>+7.2f}pp {int(ov.sum()):>5}")

    # Stability summary
    fold_deltas = []
    for fold, g in df.groupby("fold_month"):
        b  = g["model_correct"].mean()
        ov = (g["p_pick_mod"].between(LO, HI, inclusive="left") &
              (g["model_home"] != g["mkt_home"]))
        h  = np.where(ov, g["mkt_correct"], g["model_correct"]).mean()
        fold_deltas.append(h - b)
    fold_deltas = np.array(fold_deltas)
    n_better = int((fold_deltas > 0).sum())
    print()
    print(f"Folds where hybrid improved acc: {n_better}/{len(fold_deltas)}")
    print(f"Mean delta across folds: {fold_deltas.mean()*100:+.3f}pp  "
          f"(std {fold_deltas.std()*100:.3f}pp)")


if __name__ == "__main__":
    main()
