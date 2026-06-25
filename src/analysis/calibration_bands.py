"""Test the user's hypothesis: 'teams shown at 45–48% prob actually tend to win.'

If model is well-calibrated, a team at p=46% wins ~46% of the time.
If the user is right, that bucket wins notably MORE than 46% → flipping the
pick in that band would boost accuracy.

We split the analysis two ways:
  A) by p_home (home win prob bucket → actual home win rate)
  B) by p_pick (the side the model picks → actual pick correct rate)
  C) by p_dog  (the side the model AVOIDS → actual dog win rate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def _bin_label(lo, hi):
    return f"[{lo*100:>4.1f}, {hi*100:>4.1f})"


def main() -> None:
    df = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    df = df.dropna(subset=["p_home", "home_win"]).copy()

    df["p_pick"] = np.maximum(df["p_home"], 1 - df["p_home"])
    df["p_dog"]  = 1 - df["p_pick"]
    df["model_home"] = df["p_home"] >= 0.5
    df["pick_won"] = np.where(df["model_home"], df["home_win"] == 1, df["home_win"] == 0)
    df["dog_won"]  = ~df["pick_won"]
    df["mkt_home"] = df["market_p_home"] >= 0.5
    df["mkt_correct"] = np.where(df["mkt_home"], df["home_win"] == 1, df["home_win"] == 0)

    print(f"global n={len(df):,}  home_win_rate={df['home_win'].mean():.4f}  "
          f"model_acc={df['pick_won'].mean():.4f}  mkt_acc={df['mkt_correct'].mean():.4f}\n")

    # ------------------------------------------------------------------
    # A) p_home buckets
    # ------------------------------------------------------------------
    print("=" * 80)
    print("A) p_home BUCKETS — actual home-win rate vs predicted")
    print("=" * 80)
    print(f"{'bucket':>16} {'n':>5} {'pred':>7} {'actual':>7} {'delta':>8} {'flip_acc':>9}")
    bins = np.arange(0.25, 0.76, 0.025)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df["p_home"] >= lo) & (df["p_home"] < hi)
        g = df[mask]
        if len(g) < 30:
            continue
        actual = g["home_win"].mean()
        pred = g["p_home"].mean()
        # If we flipped the model pick for games in this bucket:
        flipped_pick_won = ~g["pick_won"]
        flip_acc = flipped_pick_won.mean()
        delta = actual - pred
        mark = ""
        if abs(delta) > 0.03 and len(g) >= 100:
            mark = " <<<" if delta > 0 else " (low)"
        print(f"{_bin_label(lo, hi):>16} {len(g):>5} "
              f"{pred*100:>6.1f}% {actual*100:>6.1f}% "
              f"{delta*100:>+7.2f}pp {flip_acc*100:>8.1f}%{mark}")

    # ------------------------------------------------------------------
    # B) p_pick buckets — does the model's preferred side actually win at the rate it predicts?
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("B) p_pick BUCKETS — model picks at this confidence level")
    print("=" * 80)
    print(f"{'bucket':>16} {'n':>5} {'pred_acc':>8} {'actual_acc':>10} {'delta':>8} {'flip_acc':>9}")
    bins = np.arange(0.50, 0.86, 0.025)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df["p_pick"] >= lo) & (df["p_pick"] < hi)
        g = df[mask]
        if len(g) < 30:
            continue
        pred = g["p_pick"].mean()
        actual = g["pick_won"].mean()
        flip_acc = 1 - actual
        delta = actual - pred
        mark = ""
        if delta < -0.04 and len(g) >= 100:
            mark = " <<< MODEL OVERSURE"
        print(f"{_bin_label(lo, hi):>16} {len(g):>5} "
              f"{pred*100:>7.1f}% {actual*100:>9.1f}% "
              f"{delta*100:>+7.2f}pp {flip_acc*100:>8.1f}%{mark}")

    # ------------------------------------------------------------------
    # C) The user's specific zone: model picks at 45-48% (i.e., model says
    #    PICK_SIDE has 52-55% prob, DOG_SIDE has 45-48%). Does the dog win
    #    more than expected?
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("C) USER HYPOTHESIS — 'teams shown at 45-48% prob actually win often'")
    print("=" * 80)
    # If the team's displayed prob is 45-48%, that's the DOG side. So
    # p_dog in [0.45, 0.48). Equivalently, p_pick in [0.52, 0.55).
    # Test: in that band, what's the actual dog win rate?
    bins = [(0.40, 0.42), (0.42, 0.44), (0.44, 0.46), (0.46, 0.48),
            (0.48, 0.50)]
    print(f"{'dog_prob_band':>16} {'n':>5} {'pred_dog_win':>13} "
          f"{'actual_dog_win':>15} {'delta':>8} {'flip_better?':>13}")
    for lo, hi in bins:
        mask = (df["p_dog"] >= lo) & (df["p_dog"] < hi)
        g = df[mask]
        if len(g) < 30:
            continue
        pred = g["p_dog"].mean()
        actual = g["dog_won"].mean()
        delta = actual - pred
        # If we flipped the model pick for these, accuracy would be `actual`.
        # If `actual > 0.50`, flipping beats keeping. Otherwise not.
        flip_better = "YES" if actual > 0.50 else "no"
        print(f"{_bin_label(lo, hi):>16} {len(g):>5} "
              f"{pred*100:>12.1f}% {actual*100:>14.1f}% "
              f"{delta*100:>+7.2f}pp {flip_better:>13}")

    # ------------------------------------------------------------------
    # D) Per p_pick band — would flipping beat keeping (raw accuracy + market check)?
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("D) DECISION TABLE — should we flip in any p_pick band?")
    print("=" * 80)
    print(f"{'p_pick_band':>16} {'n':>5} {'model_acc':>10} {'flip_acc':>9} {'mkt_acc':>9} {'best':>8}")
    bins = [(0.50, 0.52), (0.52, 0.55), (0.55, 0.58), (0.58, 0.62),
            (0.62, 0.66), (0.66, 0.70), (0.70, 0.78)]
    for lo, hi in bins:
        mask = (df["p_pick"] >= lo) & (df["p_pick"] < hi)
        g = df[mask]
        if len(g) < 30:
            continue
        mod = g["pick_won"].mean()
        flp = 1 - mod
        mkt = g["mkt_correct"].mean()
        best = max({"model": mod, "flip": flp, "market": mkt}.items(), key=lambda x: x[1])[0]
        print(f"{_bin_label(lo, hi):>16} {len(g):>5} "
              f"{mod*100:>9.2f}% {flp*100:>8.2f}% {mkt*100:>8.2f}% {best:>8}")


if __name__ == "__main__":
    main()
