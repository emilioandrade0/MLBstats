"""Brute-force pattern hunter: find filter combinations with highest ROI.

Iterates over all filters (and pairs, and triples) for each market × mode,
reporting the top-K subsets by ROI (with a minimum sample size to avoid noise).

Output: data/processed/pattern_leaderboard.parquet
"""
from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd

from ..paths import PROCESSED
from ..features.filters import FILTER_CATALOG


def _flatten_filters() -> list[str]:
    return [name for cat in FILTER_CATALOG.values() for (name, _) in cat["filters"]]


def hunt(min_n: int = 30, max_combo: int = 2) -> pd.DataFrame:
    picks = pd.read_parquet(PROCESSED / "picks_backtest.parquet")
    gf = pd.read_parquet(PROCESSED / "game_filters.parquet")
    picks = picks.merge(gf, on="game_id", how="left")

    filters = _flatten_filters()
    rows: list[dict] = []

    for market in ["ml", "spread", "total"]:
        for mode in ["confidence", "value"]:
            base = picks[(picks["market"] == market) & (picks["mode"] == mode)]
            if base.empty:
                continue

            for k in range(1, max_combo + 1):
                for combo in combinations(filters, k):
                    mask = pd.Series(True, index=base.index)
                    for f in combo:
                        mask &= base[f] == True
                    sub = base[mask]
                    settled = int(((sub["won"] == True) | (sub["won"] == False)).sum())
                    if settled < min_n:
                        continue
                    wins = int((sub["won"] == True).sum())
                    losses = int((sub["won"] == False).sum())
                    staked = float(sub["stake"].sum())
                    pnl = float(sub["pnl"].sum())
                    rows.append({
                        "market": market, "mode": mode,
                        "filters": " + ".join(combo),
                        "k": k,
                        "n": len(sub), "wins": wins, "losses": losses,
                        "acc_pct": wins / (wins + losses) * 100 if (wins + losses) else 0,
                        "roi_pct": pnl / staked * 100 if staked else 0,
                        "pnl": pnl,
                    })

    df = pd.DataFrame(rows).sort_values(["market", "mode", "roi_pct"], ascending=[True, True, False])
    path = PROCESSED / "pattern_leaderboard.parquet"
    df.to_parquet(path, index=False)

    print(f"\npattern_leaderboard.parquet: {len(df):,} combos evaluated (min_n={min_n})")
    print("\n=== TOP 5 por market × mode (ROI, min 30 picks) ===")
    for (market, mode), g in df.groupby(["market", "mode"]):
        top = g.head(5)
        print(f"\n  {market.upper()} · {mode}")
        for _, r in top.iterrows():
            print(f"    ROI {r['roi_pct']:+6.2f}%  acc {r['acc_pct']:5.1f}%  n={r['n']:>4}  {r['filters']}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=30,
                    help="Minimum settled bets to consider (default 30, filters noise)")
    ap.add_argument("--max-combo", type=int, default=2,
                    help="Max number of filters to AND (default 2; try 3 for deeper search)")
    args = ap.parse_args()
    hunt(min_n=args.min_n, max_combo=args.max_combo)


if __name__ == "__main__":
    main()
