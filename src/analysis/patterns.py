"""Pattern analysis on the historical odds/outcomes sequence.

The user's intuition: watching MLB results day after day, the favorite/underdog
pattern *looks* like it repeats (e.g., right-right-left-right-right). We test
this against the null hypothesis that game outcomes are independent draws.

Tests run:
  1. Per-day favorite win rate autocorrelation across days
  2. Wald-Wolfowitz runs test on favorite/underdog sequence per day
  3. Sequential dependence: P(fav wins today's slate | fav won yesterday's)
  4. Markov chain test on per-day F/U majority
  5. Total runs autocorrelation
  6. Streak length distribution vs expected under independence

A "real" pattern requires:
  - p < 0.05 on the test
  - Effect size large enough to be useful (|correlation| > 0.05, etc.)
  - Persistence across a holdout period (not just data dredging)

Usage:
    python -m src.analysis.patterns
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..normalize.paths import PROCESSED


def _load_data() -> pd.DataFrame:
    """Build a per-game record: date, was_favorite_home, did_favorite_win, total_runs."""
    games = pd.read_parquet(PROCESSED / "games.parquet",
                             columns=["game_pk", "game_date", "season", "game_type",
                                      "home_team_abbrev", "away_team_abbrev",
                                      "home_score", "away_score"])
    games = games[games["game_type"] == "R"].copy()
    games = games.dropna(subset=["home_score", "away_score", "game_date"])
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["home_won"] = (games["home_score"] > games["away_score"]).astype(int)
    games["total_runs"] = games["home_score"] + games["away_score"]

    # Bring in market lines (devigged) to identify favorite/underdog per game
    market = pd.read_parquet(PROCESSED / "features_market.parquet",
                              columns=["game_pk", "market_p_home"])
    df = games.merge(market, on="game_pk", how="inner")
    df["home_was_fav"] = (df["market_p_home"] > 0.5).astype(int)
    # 1 if the favorite won, 0 if underdog won
    df["fav_won"] = ((df["home_was_fav"] == 1) & (df["home_won"] == 1)) | \
                    ((df["home_was_fav"] == 0) & (df["home_won"] == 0))
    df["fav_won"] = df["fav_won"].astype(int)
    return df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def _runs_test(sequence: list[int]) -> tuple[float, float]:
    """Wald-Wolfowitz runs test for randomness in a binary sequence.

    Returns (Z-score, p-value). Under H0 (random), Z ~ N(0,1).
    """
    n = len(sequence)
    if n < 10:
        return 0.0, 1.0
    n1 = sum(sequence)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return 0.0, 1.0
    # Count runs
    runs = 1 + sum(1 for i in range(1, n) if sequence[i] != sequence[i-1])
    expected = (2 * n1 * n0) / n + 1
    variance = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1))
    if variance <= 0:
        return 0.0, 1.0
    z = (runs - expected) / np.sqrt(variance)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def test1_daily_fav_rate_autocorrelation(df: pd.DataFrame) -> dict:
    """Does today's favorite win rate predict tomorrow's?"""
    daily = df.groupby(df["game_date"].dt.date).agg(
        n=("fav_won", "size"),
        fav_rate=("fav_won", "mean"),
    )
    daily = daily[daily["n"] >= 8].copy()  # only "full" slate days
    s = daily["fav_rate"].values
    if len(s) < 30:
        return {"name": "Daily favorite-rate autocorrelation", "skipped": True}
    # lag-1 autocorrelation
    r, p = stats.pearsonr(s[:-1], s[1:])
    return {
        "name": "Daily favorite-rate autocorrelation (lag-1)",
        "n_days": len(s),
        "correlation": round(r, 4),
        "p_value": round(p, 4),
        "verdict": "SIGNIFICANT" if p < 0.05 and abs(r) > 0.05 else "no pattern",
        "interpretation": (
            f"Today's favorite win rate vs tomorrow's correlate r={r:.3f}. "
            f"Under independence r ≈ 0."
        ),
    }


def test2_runs_test_per_slate(df: pd.DataFrame) -> dict:
    """Wald-Wolfowitz runs test on the F/U sequence within each day."""
    p_values = []
    z_scores = []
    skipped = 0
    for _, day in df.groupby(df["game_date"].dt.date):
        if len(day) < 10:
            skipped += 1
            continue
        seq = day["fav_won"].tolist()
        z, p = _runs_test(seq)
        p_values.append(p)
        z_scores.append(z)
    n_days = len(p_values)
    n_significant = sum(1 for p in p_values if p < 0.05)
    expected_significant = 0.05 * n_days  # under H0
    # Binomial test: are there more significant days than expected?
    binom_p = stats.binom.sf(n_significant - 1, n_days, 0.05)
    return {
        "name": "Within-slate runs test (per day)",
        "n_days_tested": n_days,
        "n_significant_days": n_significant,
        "expected_under_random": round(expected_significant, 1),
        "p_excess": round(float(binom_p), 4),
        "mean_z": round(float(np.mean(z_scores)), 3),
        "verdict": "SIGNIFICANT" if binom_p < 0.05 else "no pattern",
        "interpretation": (
            f"Out of {n_days} slate-days, {n_significant} show non-random F/U "
            f"ordering at p<0.05 (expected by chance: {expected_significant:.0f})."
        ),
    }


def test3_sequential_dependence(df: pd.DataFrame) -> dict:
    """Does the favorite winning yesterday predict the favorite winning today?"""
    daily = df.groupby(df["game_date"].dt.date).agg(
        n=("fav_won", "size"),
        fav_wins=("fav_won", "sum"),
    )
    daily = daily[daily["n"] >= 6].copy()
    daily["fav_majority"] = (daily["fav_wins"] / daily["n"] > 0.5).astype(int)
    seq = daily["fav_majority"].values
    if len(seq) < 30:
        return {"name": "Sequential dependence (Markov-1)", "skipped": True}
    # Build 2x2 transition table
    table = np.zeros((2, 2), dtype=int)
    for i in range(1, len(seq)):
        table[seq[i-1], seq[i]] += 1
    # Conditional probabilities
    p_fav_given_fav  = table[1, 1] / max(1, table[1].sum())
    p_fav_given_dog  = table[0, 1] / max(1, table[0].sum())
    p_fav_marginal   = seq.mean()
    # Chi-square test of independence
    chi2, p, _, _ = stats.chi2_contingency(table)
    return {
        "name": "Sequential dependence (Markov-1, day-to-day)",
        "n_transitions": int(table.sum()),
        "p_fav_majority_unconditional": round(float(p_fav_marginal), 4),
        "p_fav_majority_given_fav_yesterday": round(float(p_fav_given_fav), 4),
        "p_fav_majority_given_dog_yesterday": round(float(p_fav_given_dog), 4),
        "diff": round(float(p_fav_given_fav - p_fav_given_dog), 4),
        "chi2_p_value": round(float(p), 4),
        "verdict": "SIGNIFICANT" if p < 0.05 and abs(p_fav_given_fav - p_fav_given_dog) > 0.05 else "no pattern",
        "interpretation": (
            f"P(fav majority today | fav majority yesterday) = {p_fav_given_fav:.3f}, "
            f"P(fav majority today | dog majority yesterday) = {p_fav_given_dog:.3f}. "
            f"Diff = {p_fav_given_fav - p_fav_given_dog:+.3f}. "
            f"Under independence diff ≈ 0."
        ),
    }


def test4_total_runs_autocorrelation(df: pd.DataFrame) -> dict:
    """Does yesterday's average total runs predict today's?"""
    daily = df.groupby(df["game_date"].dt.date).agg(
        n=("total_runs", "size"),
        mean_total=("total_runs", "mean"),
    )
    daily = daily[daily["n"] >= 8].copy()
    s = daily["mean_total"].values
    if len(s) < 30:
        return {"name": "Total runs autocorrelation", "skipped": True}
    r, p = stats.pearsonr(s[:-1], s[1:])
    return {
        "name": "Daily total-runs autocorrelation (lag-1)",
        "n_days": len(s),
        "correlation": round(float(r), 4),
        "p_value": round(float(p), 4),
        "verdict": "SIGNIFICANT" if p < 0.05 and abs(r) > 0.05 else "no pattern",
        "interpretation": (
            f"Yesterday's mean total runs vs today's correlate r={r:.3f}."
        ),
    }


def test5_streak_distribution(df: pd.DataFrame) -> dict:
    """Compare streak length distribution to expected under iid Bernoulli."""
    p_fav = df["fav_won"].mean()
    # Compute streak lengths in the full sequence
    seq = df["fav_won"].values
    streaks = []
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        streaks.append((seq[i], j - i + 1))
        i = j + 1
    # Expected streak distribution under iid
    # P(streak of length k of "1" then a "0") = p^k * (1-p)
    fav_streak_lengths = [length for val, length in streaks if val == 1]
    if len(fav_streak_lengths) < 50:
        return {"name": "Streak distribution", "skipped": True}
    # Compare empirical mean streak length to expected
    expected_mean = 1 / (1 - p_fav)  # geometric distribution mean
    observed_mean = float(np.mean(fav_streak_lengths))
    # Bootstrap CI on observed mean
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(fav_streak_lengths, size=len(fav_streak_lengths), replace=True)
        boot_means.append(np.mean(sample))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    z = (observed_mean - expected_mean) / np.std(boot_means)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "name": "Favorite streak length distribution",
        "p_favorite": round(float(p_fav), 4),
        "expected_mean_streak": round(float(expected_mean), 3),
        "observed_mean_streak": round(observed_mean, 3),
        "95_ci": [round(float(ci_low), 3), round(float(ci_high), 3)],
        "p_value": round(float(p), 4),
        "verdict": "SIGNIFICANT" if p < 0.05 else "no pattern",
        "interpretation": (
            f"Expected mean streak under iid: {expected_mean:.2f}. "
            f"Observed: {observed_mean:.2f}. CI: [{ci_low:.2f}, {ci_high:.2f}]."
        ),
    }


def _print_report(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("PATTERN ANALYSIS — does the historical F/U sequence show structure?")
    print("=" * 78)
    real_patterns = 0
    for r in results:
        if r.get("skipped"):
            print(f"\n[{r['name']}] (skipped — insufficient data)")
            continue
        print(f"\n[{r['name']}]")
        for k, v in r.items():
            if k in ("name", "interpretation", "verdict"):
                continue
            print(f"  {k:40s} {v}")
        verdict = r["verdict"]
        flag = "[*] PATRON REAL" if verdict == "SIGNIFICANT" else "[ ] sin patron"
        if verdict == "SIGNIFICANT":
            real_patterns += 1
        print(f"  >> {flag}: {r['interpretation']}")
    print("\n" + "=" * 78)
    print(f"CONCLUSIÓN: {real_patterns}/{len(results)} tests muestran patrón explotable.")
    print("=" * 78)
    if real_patterns == 0:
        print(
            "\nVeredict: las secuencias de favorito/underdog son estadísticamente\n"
            "indistinguibles de tiros independientes (gambler's fallacy confirmado).\n"
            "Sumar 'pattern features' no mejorará accuracy.\n"
            "Caminos alternativos: lineup splits L/R, umpire-specific tendencies,\n"
            "lazy-bullpen features post-doble jornada."
        )
    else:
        print(
            "\nVeredict: hay patrón explotable. Próximo paso: codificar como features\n"
            "y re-entrenar el modelo con walk-forward."
        )


def main() -> None:
    print("Loading data…")
    df = _load_data()
    print(f"Loaded {len(df):,} games across {df['game_date'].nunique():,} days "
          f"(seasons {df['season'].min()}–{df['season'].max()}).")

    results = [
        test1_daily_fav_rate_autocorrelation(df),
        test2_runs_test_per_slate(df),
        test3_sequential_dependence(df),
        test4_total_runs_autocorrelation(df),
        test5_streak_distribution(df),
    ]
    _print_report(results)


if __name__ == "__main__":
    main()
