"""Daily pattern audit — do underdogs over-perform on specific days/cycles?

For every day in the historical data, computes:
  - # games
  - # underdog wins (decimal >= 2.0 winner)
  - # favorite wins
  - actual_dog_wr   = dog_wins / total_games (in that day)
  - expected_dog_wr = avg(book_implied_prob_dog) for that day
  - deviation_pp   = actual - expected  (positive = dogs over-performed)

Then looks for:
  1. Cycles — rolling 7d / 14d / 30d windows
  2. Day-of-week — Mon..Sun average deviation
  3. "Hot dog days" — days where deviation > +15pp
  4. Streaks — consecutive days of high dog WR
  5. Slate-size effect — small (≤8) vs big (≥13) slates

Output: data/processed/daily_patterns.parquet  + console findings.

Run:
  python -m src.analysis.daily_patterns
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "daily_patterns.parquet"


def _build():
    # Load games + closing odds
    games = pd.read_parquet(PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "season",
                 "home_score", "away_score",
                 "home_team_abbrev", "away_team_abbrev"])
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_score", "away_score"]).copy()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    oc = pd.read_parquet(PROCESSED / "odds_close.parquet")
    sharp = oc[oc["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    sharp["home_ml"] = (sharp["home_ml_close"].fillna(sharp["home_ml_current"])
                                              .fillna(sharp["home_ml_top"]))
    sharp["away_ml"] = (sharp["away_ml_close"].fillna(sharp["away_ml_current"])
                                              .fillna(sharp["away_ml_top"]))
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
    ).reset_index()
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    matched = xref[xref["_merge"]=="both"][["espn_event_id","game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    ml = agg.merge(matched, on="espn_event_id", how="inner") \
            .drop(columns=["espn_event_id"]).drop_duplicates("game_pk")
    df = games.merge(ml, on="game_pk", how="inner")

    def _am_to_dec(v):
        if pd.isna(v): return np.nan
        v = float(v)
        if v == 0: return np.nan
        return 1 + (v/100 if v > 0 else 100/abs(v))

    df["dec_home"] = df["home_ml"].apply(_am_to_dec)
    df["dec_away"] = df["away_ml"].apply(_am_to_dec)
    df = df.dropna(subset=["dec_home", "dec_away"]).copy()
    df["imp_home"] = 1/df["dec_home"]
    df["imp_away"] = 1/df["dec_away"]
    # De-vigged probabilities for "expected"
    s = df["imp_home"] + df["imp_away"]
    df["fair_p_home"] = df["imp_home"]/s
    df["fair_p_away"] = df["imp_away"]/s

    # Underdog = higher decimal side. Mark which side & if it won.
    df["dog_is_home"] = df["dec_home"] > df["dec_away"]
    df["dog_dec"]     = np.where(df["dog_is_home"], df["dec_home"], df["dec_away"])
    df["dog_won"]     = np.where(df["dog_is_home"], df["home_win"]==1, df["home_win"]==0)
    df["fair_p_dog"]  = np.where(df["dog_is_home"], df["fair_p_home"], df["fair_p_away"])
    return df


def main():
    df = _build()
    print(f"Audited {len(df):,} games  ({df['game_date'].min().date()} → {df['game_date'].max().date()})")
    print()

    # ── Per-day aggregation ─────────────────────────────────────────────────
    daily = df.groupby(df["game_date"].dt.date).agg(
        n_games=("game_pk", "size"),
        dog_wins=("dog_won", "sum"),
        actual_dog_wr=("dog_won", "mean"),
        expected_dog_wr=("fair_p_dog", "mean"),
        avg_dog_dec=("dog_dec", "mean"),
    ).reset_index()
    daily = daily.rename(columns={"game_date": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["dev_pp"] = (daily["actual_dog_wr"] - daily["expected_dog_wr"]) * 100
    daily["dow"]    = daily["date"].dt.day_name().str[:3]
    daily["dom"]    = daily["date"].dt.day
    daily["month"]  = daily["date"].dt.month
    daily["week"]   = daily["date"].dt.isocalendar().week
    daily["year"]   = daily["date"].dt.year

    daily = daily.sort_values("date").reset_index(drop=True)
    # Rolling windows
    daily["dev_7d"]  = daily["dev_pp"].rolling(7,  min_periods=3).mean()
    daily["dev_14d"] = daily["dev_pp"].rolling(14, min_periods=5).mean()
    daily["dev_30d"] = daily["dev_pp"].rolling(30, min_periods=10).mean()

    daily.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(daily):,} days)")
    print()

    # ── 1. Overall ───────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"Overall: actual dog WR = {df['dog_won'].mean()*100:.2f}%  "
          f"expected = {df['fair_p_dog'].mean()*100:.2f}%  "
          f"deviation = {(df['dog_won'].mean() - df['fair_p_dog'].mean())*100:+.2f}pp")
    print()

    # ── 2. Day of week ───────────────────────────────────────────────────────
    print("=" * 80)
    print("By day of week:")
    dow = daily.groupby("dow").agg(
        n_days=("date", "size"),
        n_games=("n_games", "sum"),
        avg_actual_wr=("actual_dog_wr", "mean"),
        avg_expected_wr=("expected_dog_wr", "mean"),
        avg_dev_pp=("dev_pp", "mean"),
    )
    dow["avg_actual_wr"] *= 100
    dow["avg_expected_wr"] *= 100
    dow = dow.round(2).sort_values("avg_dev_pp", ascending=False)
    print(dow.to_string())
    print()

    # ── 3. Day of month ──────────────────────────────────────────────────────
    print("=" * 80)
    print("Top dog-friendly days of month (avg deviation):")
    dom = daily.groupby("dom").agg(
        n_days=("date", "size"),
        avg_dev_pp=("dev_pp", "mean"),
    ).round(2)
    print("  TOP 5 (dogs over-perform):")
    print(dom.sort_values("avg_dev_pp", ascending=False).head(5).to_string())
    print("  BOTTOM 5 (dogs under-perform):")
    print(dom.sort_values("avg_dev_pp", ascending=True).head(5).to_string())
    print()

    # ── 4. Hot dog days (deviation > +15pp) ──────────────────────────────────
    print("=" * 80)
    print("'Hot dog days' (actual dog WR > expected + 15pp, slate >= 8 games):")
    hot = daily[(daily["dev_pp"] > 15) & (daily["n_games"] >= 8)].copy()
    print(f"  Count: {len(hot)} days  ({len(hot)/len(daily[daily['n_games']>=8])*100:.1f}% of qualifying days)")
    print("  Recent examples:")
    print(hot[["date","n_games","actual_dog_wr","expected_dog_wr","dev_pp","dow"]].tail(15).to_string(index=False))
    print()

    # ── 5. Cold dog days (deviation < -15pp) ─────────────────────────────────
    print("=" * 80)
    print("'Fav lock days' (actual dog WR < expected - 15pp, slate >= 8 games):")
    cold = daily[(daily["dev_pp"] < -15) & (daily["n_games"] >= 8)].copy()
    print(f"  Count: {len(cold)} days  ({len(cold)/len(daily[daily['n_games']>=8])*100:.1f}% of qualifying days)")
    print()

    # ── 6. Streaks — consecutive days with dev > 0 ──────────────────────────
    print("=" * 80)
    daily["dog_streak"] = (daily["dev_pp"] > 0).astype(int)
    # Compute run-length encoding of streaks
    streak_id = (daily["dog_streak"] != daily["dog_streak"].shift()).cumsum()
    streaks = daily.groupby(streak_id).agg(
        side=("dog_streak", "first"),
        days=("date", "size"),
        start=("date", "min"),
        end=("date", "max"),
        total_dev=("dev_pp", "sum"),
    )
    dog_streaks = streaks[streaks["side"]==1].sort_values("days", ascending=False).head(10)
    fav_streaks = streaks[streaks["side"]==0].sort_values("days", ascending=False).head(10)
    print("Longest DOG streaks (consecutive days with dogs over-performing):")
    print(dog_streaks.to_string())
    print()
    print("Longest FAV streaks (consecutive days with favs dominating):")
    print(fav_streaks.to_string())
    print()

    # ── 7. Slate-size effect ─────────────────────────────────────────────────
    print("=" * 80)
    print("Slate-size effect:")
    daily["slate_bucket"] = pd.cut(daily["n_games"], bins=[0,5,8,11,14,17,99],
        labels=["1-5","6-8","9-11","12-14","15-17","18+"])
    slate = daily.groupby("slate_bucket", observed=False).agg(
        n_days=("date", "size"),
        avg_actual_wr=("actual_dog_wr", "mean"),
        avg_dev_pp=("dev_pp", "mean"),
    ).round(2)
    slate["avg_actual_wr"] *= 100
    print(slate.to_string())
    print()

    # ── 8. Month effect ──────────────────────────────────────────────────────
    print("=" * 80)
    print("By month (regular season Apr-Oct):")
    mo = daily.groupby("month").agg(
        n_days=("date", "size"),
        n_games=("n_games", "sum"),
        avg_dev_pp=("dev_pp", "mean"),
    ).round(2)
    print(mo.to_string())
    print()

    # ── 9. Autocorrelation: does dev_pp predict next day's dev_pp? ──────────
    print("=" * 80)
    print("Day-to-day autocorrelation (does today's dog-day predict tomorrow's?):")
    for lag in [1, 2, 3, 7, 14]:
        ac = daily["dev_pp"].autocorr(lag=lag)
        print(f"  lag={lag:>2d} days: {ac:+.4f}  {'(strong)' if abs(ac) > 0.15 else '(weak)' if abs(ac) > 0.05 else '(none)'}")


if __name__ == "__main__":
    main()
