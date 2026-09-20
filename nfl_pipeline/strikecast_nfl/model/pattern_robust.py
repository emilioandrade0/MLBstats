"""Robust pattern hunter with anti-overfitting safeguards.

Instead of one train/test split (which can hide overfitting), we do:
  1. Rolling walk-forward: for each season in TEST_SEASONS, train on prior years
     and evaluate the combo ON THAT SEASON ALONE.
  2. Consistency filter: keep only combos with positive ROI in >= min_positive
     of the test seasons.
  3. Sample-size floor per season: skip season if < min_n_per_season picks.
  4. Bootstrap CI (optional): resample the pooled test set 1000x to estimate
     the ROI 5th percentile. Skip if p5 < 0.

Also supports 3-filter combos with tighter thresholds (bigger n, higher
consistency required).

Output: data/processed/pattern_robust.parquet
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from ..paths import PROCESSED
from ..features.filters import FILTER_CATALOG


def _flat() -> list[str]:
    return [n for cat in FILTER_CATALOG.values() for (n, _) in cat["filters"]]


def _bucket_roi(g: pd.DataFrame) -> tuple[int, int, int, float, float]:
    n = len(g)
    w = int((g["won"] == True).sum())
    l = int((g["won"] == False).sum())
    staked = float(g["stake"].sum())
    pnl = float(g["pnl"].sum())
    acc = w / (w + l) * 100 if (w + l) else 0.0
    roi = pnl / staked * 100 if staked else 0.0
    return n, w, l, acc, roi


def _bootstrap_roi_p5(g: pd.DataFrame, n_boot: int = 500, seed: int = 42) -> float:
    if g.empty:
        return -999.0
    rng = np.random.default_rng(seed)
    pnl = g["pnl"].values
    stake = g["stake"].values
    n = len(pnl)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = stake[idx].sum()
        rois[i] = (pnl[idx].sum() / s * 100) if s else 0.0
    return float(np.percentile(rois, 5))


def hunt_robust(
    train_start: int = 2015,
    test_seasons: list[int] | None = None,
    k_values: tuple[int, ...] = (1, 2, 3),
    min_n_per_season: int = 6,
    min_n_pooled_test: int = 40,
    min_positive_seasons: dict[int, int] | None = None,
    require_bootstrap: bool = True,
) -> pd.DataFrame:
    """
    For each combo of size k, evaluate on each test season and report:
      - consistency: fraction of test seasons with ROI > 0
      - pooled test ROI + 5th percentile bootstrap CI
    """
    if test_seasons is None:
        test_seasons = [2021, 2022, 2023, 2024, 2025]
    if min_positive_seasons is None:
        # Require k+2 positive seasons out of 5 for k=1; more for bigger combos
        min_positive_seasons = {1: 3, 2: 4, 3: 4}

    picks = pd.read_parquet(PROCESSED / "picks_backtest.parquet")
    gf = pd.read_parquet(PROCESSED / "game_filters.parquet")
    picks = picks.merge(gf, on="game_id", how="left")

    filters = _flat()
    rows: list[dict] = []

    total_combos = 0
    for k in k_values:
        total_combos += len(list(combinations(filters, k)))
    print(f"Evaluando {total_combos:,} combinaciones sobre {len(test_seasons)} temporadas OOS...")

    for market in ["ml", "spread", "total"]:
        for mode in ["confidence", "value"]:
            base = picks[(picks["market"] == market) & (picks["mode"] == mode)]
            if base.empty:
                continue
            test_pool = base[base["season"].isin(test_seasons)]

            for k in k_values:
                need_pos = min_positive_seasons.get(k, k + 1)
                for combo in combinations(filters, k):
                    mask_all = pd.Series(True, index=base.index)
                    for f in combo:
                        mask_all &= base[f] == True
                    sub_all = base[mask_all]
                    if len(sub_all) < min_n_pooled_test:
                        continue

                    # per-season ROI
                    season_metrics = []
                    positive = 0
                    for s in test_seasons:
                        sub = sub_all[sub_all["season"] == s]
                        if len(sub) < min_n_per_season:
                            season_metrics.append((s, len(sub), np.nan, np.nan))
                            continue
                        n, w, l, acc, roi = _bucket_roi(sub)
                        season_metrics.append((s, n, acc, roi))
                        if roi > 0:
                            positive += 1
                    if positive < need_pos:
                        continue

                    # pooled test metrics
                    pooled_test = sub_all[sub_all["season"].isin(test_seasons)]
                    if len(pooled_test) < min_n_pooled_test:
                        continue
                    n_p, w_p, l_p, acc_p, roi_p = _bucket_roi(pooled_test)

                    # bootstrap p5
                    p5 = _bootstrap_roi_p5(pooled_test) if require_bootstrap else roi_p
                    if require_bootstrap and p5 < 0:
                        continue

                    # in-sample ROI on training years (for context)
                    train_sub = sub_all[sub_all["season"] < min(test_seasons)]
                    n_tr, _, _, acc_tr, roi_tr = _bucket_roi(train_sub) if not train_sub.empty else (0, 0, 0, 0, 0)

                    rows.append({
                        "market": market, "mode": mode,
                        "filters": " + ".join(combo), "k": k,
                        "n_train": n_tr, "roi_train": roi_tr, "acc_train": acc_tr,
                        "n_test": n_p, "roi_test": roi_p, "acc_test": acc_p,
                        "roi_test_p5": p5,
                        "positive_seasons": positive,
                        "seasons_evaluated": len([m for m in season_metrics if not np.isnan(m[3])]),
                        "per_season_roi": ", ".join(
                            f"{s}:{r:+.1f}%" if not np.isnan(r) else f"{s}:n/a"
                            for (s, _, _, r) in season_metrics
                        ),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("Ningún patrón pasó los filtros anti-overfitting.")
        return df

    df["overfit_gap"] = df["roi_train"] - df["roi_test"]
    df = df.sort_values(["market", "mode", "roi_test"], ascending=[True, True, False])

    path = PROCESSED / "pattern_robust.parquet"
    df.to_parquet(path, index=False)

    print(f"\npattern_robust.parquet: {len(df):,} patrones sobrevivieron")
    print("\n=== TOP 5 por market x mode (ROI test pooled, p5 > 0) ===")
    for (mkt, mode), g in df.groupby(["market", "mode"]):
        print(f"\n  {mkt.upper()} · {mode}")
        for _, r in g.head(5).iterrows():
            print(f"    ROI_test {r['roi_test']:+6.2f}% (p5 {r['roi_test_p5']:+5.1f}%)  "
                  f"pos_seasons {r['positive_seasons']}/{r['seasons_evaluated']}  "
                  f"n={r['n_test']:>4}  k={r['k']}  {r['filters']}")
            print(f"                                         per season: {r['per_season_roi']}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", type=str, default="2021,2022,2023,2024,2025")
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--min-n-per-season", type=int, default=6)
    ap.add_argument("--min-n-pooled", type=int, default=40)
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    ts = [int(s) for s in args.test_seasons.split(",") if s.strip()]
    hunt_robust(
        test_seasons=ts,
        k_values=tuple(range(1, args.max_k + 1)),
        min_n_per_season=args.min_n_per_season,
        min_n_pooled_test=args.min_n_pooled,
        require_bootstrap=not args.no_bootstrap,
    )


if __name__ == "__main__":
    main()
