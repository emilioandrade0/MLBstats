"""Out-of-sample validation for pattern_hunter results.

Splits history into TRAIN (early seasons) and TEST (recent seasons).
For each combo, computes ROI on TRAIN (in-sample) and ROI on TEST (out-of-sample).

A pattern is REAL if both are positive and reasonably close. Anything that shines
in TRAIN and dies in TEST was overfitting to noise.

Output: data/processed/pattern_oos.parquet + printed leaderboard.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd

from ..paths import PROCESSED
from ..features.filters import FILTER_CATALOG


def _flat() -> list[str]:
    return [n for cat in FILTER_CATALOG.values() for (n, _) in cat["filters"]]


def hunt_oos(train_end: int = 2020, min_n_train: int = 20, min_n_test: int = 15,
             max_combo: int = 2) -> pd.DataFrame:
    picks = pd.read_parquet(PROCESSED / "picks_backtest.parquet")
    gf = pd.read_parquet(PROCESSED / "game_filters.parquet")
    picks = picks.merge(gf, on="game_id", how="left")

    filters = _flat()
    rows: list[dict] = []

    def bucket(g: pd.DataFrame) -> tuple[int, int, int, float, float]:
        n = len(g)
        wins = int((g["won"] == True).sum())
        losses = int((g["won"] == False).sum())
        staked = float(g["stake"].sum())
        pnl = float(g["pnl"].sum())
        acc = wins / (wins + losses) * 100 if (wins + losses) else 0
        roi = pnl / staked * 100 if staked else 0
        return n, wins, losses, acc, roi

    for market in ["ml", "spread", "total"]:
        for mode in ["confidence", "value"]:
            base = picks[(picks["market"] == market) & (picks["mode"] == mode)]
            if base.empty:
                continue
            train = base[base["season"] <= train_end]
            test = base[base["season"] > train_end]

            for k in range(1, max_combo + 1):
                for combo in combinations(filters, k):
                    m_train = pd.Series(True, index=train.index)
                    m_test = pd.Series(True, index=test.index)
                    for f in combo:
                        m_train &= train[f] == True
                        m_test &= test[f] == True
                    tr = train[m_train]
                    te = test[m_test]
                    if len(tr) < min_n_train or len(te) < min_n_test:
                        continue
                    ntr, wtr, ltr, atr, rtr = bucket(tr)
                    nte, wte, lte, ate, rte = bucket(te)
                    rows.append({
                        "market": market, "mode": mode,
                        "filters": " + ".join(combo), "k": k,
                        "n_train": ntr, "acc_train": atr, "roi_train": rtr,
                        "n_test": nte, "acc_test": ate, "roi_test": rte,
                        "roi_delta": rte - rtr,
                        "stable": (rtr > 0) and (rte > 0),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No patterns met sample-size minimums.")
        return df
    df = df.sort_values(["market", "mode", "roi_test"], ascending=[True, True, False])

    path = PROCESSED / "pattern_oos.parquet"
    df.to_parquet(path, index=False)

    print(f"\n=== OOS Split: train <= {train_end}, test > {train_end} ===")
    print(f"pattern_oos.parquet: {len(df):,} combos evaluated")
    print(f"\n=== PATTERNS THAT SURVIVED OOS (ROI > 0 en train Y test) ===\n")

    for (mkt, mode), g in df.groupby(["market", "mode"]):
        stable = g[g["stable"]].head(5)
        if stable.empty:
            print(f"  {mkt.upper()} · {mode}: nada sobrevivió\n")
            continue
        print(f"  {mkt.upper()} · {mode}")
        print(f"    {'Filtros':<50} {'Train ROI':>10} {'Test ROI':>10} {'n_test':>8}")
        for _, r in stable.iterrows():
            print(f"    {r['filters'][:48]:<50} {r['roi_train']:>+9.2f}% {r['roi_test']:>+9.2f}% {r['n_test']:>8}")
        print()

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", type=int, default=2020,
                    help="Last season included in training set (default 2020 -> test 2021+)")
    ap.add_argument("--min-n-train", type=int, default=20)
    ap.add_argument("--min-n-test", type=int, default=15)
    ap.add_argument("--max-combo", type=int, default=2)
    args = ap.parse_args()
    hunt_oos(train_end=args.train_end, min_n_train=args.min_n_train,
             min_n_test=args.min_n_test, max_combo=args.max_combo)


if __name__ == "__main__":
    main()
