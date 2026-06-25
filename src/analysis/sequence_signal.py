"""Test whether per-team W/L binary sequences contain predictive info BEYOND
what the rolling win% already captures.

Hypothesis: two teams with the same 10-game win% can have very different
recent SEQUENCES. e.g.:
  Team A: W-W-W-W-W-L-L-L-L-L   (was hot, now cold)
  Team B: W-L-W-L-W-L-W-L-W-L   (consistently mediocre)
Both are 5-5 in L10. Current model treats them as identical.

We test if features derived from the SEQUENCE (streak length, encoded last
N, alternation) add predictive power over the count-based win_pct features.

Test procedure (per team, treating each game as a row):
  1. For each game, compute:
     - target: won_today (0/1)
     - baseline features: win_pct_l30, win_pct_l10
     - sequence features: won_yesterday, won_2_in_row, win_streak, lose_streak,
       alternating_l4, last_5_pattern (integer 0-31)
  2. Fit two logistic regressions:
     - Model A: target ~ baseline features only
     - Model B: target ~ baseline + sequence features
  3. Compare log-likelihood. If B > A by a statistically meaningful margin,
     sequence has independent signal.
  4. Use chronological train/test split to avoid leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from ..normalize.paths import PROCESSED


def _per_team_long(games: pd.DataFrame) -> pd.DataFrame:
    """Convert game-rows into team-rows (each game = 2 team-rows)."""
    g = games.dropna(subset=["home_score", "away_score", "game_date",
                              "home_team_abbrev", "away_team_abbrev"]).copy()
    g = g[g["game_type"] == "R"].copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g["home_won"] = (g["home_score"] > g["away_score"]).astype(int)
    h = g[["game_pk", "game_date", "season", "home_team_abbrev", "home_won"]].copy()
    h.columns = ["game_pk", "game_date", "season", "team", "won"]
    a = g[["game_pk", "game_date", "season", "away_team_abbrev", "home_won"]].copy()
    a["away_won"] = 1 - a["home_won"]
    a = a[["game_pk", "game_date", "season", "away_team_abbrev", "away_won"]]
    a.columns = ["game_pk", "game_date", "season", "team", "won"]
    out = pd.concat([h, a], ignore_index=True)
    return out.sort_values(["team", "season", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)


def _add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """All shifted/lagged to avoid leakage. Computed per (team, season)."""
    g = df.groupby(["team", "season"], sort=False)
    # Past games' results (shifted)
    df["won_l1"] = g["won"].shift(1)  # yesterday
    df["won_l2"] = g["won"].shift(2)
    df["won_l3"] = g["won"].shift(3)
    df["won_l4"] = g["won"].shift(4)
    df["won_l5"] = g["won"].shift(5)
    # Streak features (consecutive W's or L's BEFORE today)
    def streak_len(s):
        # For each row, length of consecutive 1s ending at row-1
        arr = s.values
        out = np.zeros(len(arr), dtype=int)
        cnt = 0
        for i, v in enumerate(arr):
            out[i] = cnt
            cnt = cnt + 1 if v == 1 else 0
        return out
    def lose_streak_len(s):
        arr = s.values
        out = np.zeros(len(arr), dtype=int)
        cnt = 0
        for i, v in enumerate(arr):
            out[i] = cnt
            cnt = cnt + 1 if v == 0 else 0
        return out
    df["win_streak"] = g["won"].transform(streak_len)
    df["lose_streak"] = g["won"].transform(lose_streak_len)
    # Rolling counts
    df["wins_l5"] = g["won"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).sum()).reset_index(level=[0,1], drop=True)
    df["wins_l10"] = g["won"].apply(lambda s: s.shift(1).rolling(10, min_periods=1).sum()).reset_index(level=[0,1], drop=True)
    df["wins_l30"] = g["won"].apply(lambda s: s.shift(1).rolling(30, min_periods=3).sum()).reset_index(level=[0,1], drop=True)
    df["games_l5"] = g["won"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).count()).reset_index(level=[0,1], drop=True)
    df["games_l10"] = g["won"].apply(lambda s: s.shift(1).rolling(10, min_periods=1).count()).reset_index(level=[0,1], drop=True)
    df["games_l30"] = g["won"].apply(lambda s: s.shift(1).rolling(30, min_periods=3).count()).reset_index(level=[0,1], drop=True)
    df["wpct_l10"] = df["wins_l10"] / df["games_l10"].replace(0, np.nan)
    df["wpct_l30"] = df["wins_l30"] / df["games_l30"].replace(0, np.nan)
    # Alternation in last 4 games
    df["alt_l4"] = ((df["won_l1"] != df["won_l2"]) &
                     (df["won_l2"] != df["won_l3"]) &
                     (df["won_l3"] != df["won_l4"])).astype(float)
    # Encoded last 5 (binary as integer 0-31, msb = oldest)
    df["pattern_l5"] = (
        df["won_l5"].fillna(0).astype(int) * 16 +
        df["won_l4"].fillna(0).astype(int) * 8 +
        df["won_l3"].fillna(0).astype(int) * 4 +
        df["won_l2"].fillna(0).astype(int) * 2 +
        df["won_l1"].fillna(0).astype(int)
    )
    return df


def main() -> None:
    games = pd.read_parquet(PROCESSED / "games.parquet", columns=[
        "game_pk", "game_date", "season", "game_type",
        "home_team_abbrev", "away_team_abbrev",
        "home_score", "away_score",
    ])
    print(f"loaded {len(games):,} games")
    tl = _per_team_long(games)
    tl = _add_sequence_features(tl)
    # Drop rows without enough history
    tl = tl.dropna(subset=["wpct_l30", "won_l1", "won_l2", "won_l3", "won_l4", "won_l5"])
    tl = tl[tl["games_l30"] >= 10].copy()
    print(f"team-games with full sequence history: {len(tl):,}")

    # Chronological split: train on pre-2025-04, test on 2025-04+
    tl["game_date"] = pd.to_datetime(tl["game_date"])
    train = tl[tl["game_date"] < "2025-04-01"]
    test = tl[tl["game_date"] >= "2025-04-01"]
    print(f"train {len(train):,}  test {len(test):,}\n")

    # Baseline features (just win pct count-based)
    baseline_feats = ["wpct_l10", "wpct_l30"]
    # Add sequence features
    seq_feats = baseline_feats + [
        "won_l1", "won_l2", "won_l3", "won_l4", "won_l5",
        "win_streak", "lose_streak", "alt_l4",
    ]
    y_train = train["won"].astype(int)
    y_test = test["won"].astype(int)

    def fit_eval(feats, label):
        Xtr = train[feats].fillna(0.5)
        Xts = test[feats].fillna(0.5)
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(Xtr, y_train)
        p = m.predict_proba(Xts)[:, 1]
        ll = log_loss(y_test, p.clip(1e-6, 1-1e-6))
        auc = roc_auc_score(y_test, p)
        acc = ((p > 0.5) == y_test).mean()
        print(f"  {label:<30} n_feats={len(feats):>2}  ll={ll:.4f}  AUC={auc:.4f}  acc={acc:.4f}")
        return ll, m

    print("Test 1: count features vs count+sequence features")
    ll_baseline, _ = fit_eval(baseline_feats, "baseline (win_pct only)")
    ll_seq, m_seq = fit_eval(seq_feats, "+ sequence features")
    delta = ll_baseline - ll_seq
    n_test = len(test)
    print(f"\n  log_loss delta:  {delta:+.4f}  (positive = sequence helps)")
    # Likelihood ratio test
    LR_stat = 2 * delta * n_test  # 2 * (LL_complex - LL_simple) * n
    df_extra = len(seq_feats) - len(baseline_feats)
    p_value = 1 - stats.chi2.cdf(LR_stat, df_extra)
    print(f"  Likelihood ratio test:  chi2={LR_stat:.2f}  df={df_extra}  p={p_value:.4f}")
    if delta > 0.001 and p_value < 0.05:
        print(f"  >> SIGNAL FOUND in sequences (significant, effect = {delta:.4f})")
    else:
        print(f"  >> No significant signal (or effect too small to matter)")

    # Show what the model learned
    print("\n  Sequence-model coefficients (per-feature):")
    for f, c in sorted(zip(seq_feats, m_seq.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"    {f:<18} {c:+.4f}")

    # Test 2: does the last-game outcome shift predictions?
    print("\nTest 2: P(win today | won yesterday) vs P(win today | lost yesterday)")
    # Conditional on having same win_pct_l30 bucket
    test_check = test.copy()
    test_check["wpct_bkt"] = pd.cut(test_check["wpct_l30"], [0, 0.4, 0.45, 0.50, 0.55, 0.60, 1])
    pivot = test_check.groupby(["wpct_bkt", "won_l1"], observed=True)["won"].agg(["mean", "count"]).reset_index()
    print(pivot.to_string())

    # Test 3: predict from pattern_l5 alone
    print("\nTest 3: per-pattern win rate over last-5 (frequency-based)")
    pat = test.groupby("pattern_l5")["won"].agg(["mean", "count"]).reset_index()
    pat = pat[pat["count"] >= 30].sort_values("mean", ascending=False)
    pat["pattern_bits"] = pat["pattern_l5"].apply(lambda v: f"{v:05b}")
    print("\n  Top 5 best patterns:")
    print(pat.head().to_string())
    print("\n  Top 5 worst patterns:")
    print(pat.tail().to_string())


if __name__ == "__main__":
    main()
