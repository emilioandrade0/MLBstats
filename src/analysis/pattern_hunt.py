"""Exhaustive pattern hunter across walk-forward predictions.

Tests dozens of slicers (day of week, month, team, odds band, weather, pitcher
rest, recent form, park factor, …) and surfaces the cuts where:
  - the MODEL beats market by a wide margin,
  - flipping the model beats keeping it,
  - market dominates (skip zones),
  - or a third strategy (always-bet-home, always-bet-dog, etc.) wins.

Each slice reports: n, model_acc, mkt_acc, ROI for {model, flip, market, dog,
home}, best strategy, and p-value vs the global baseline. We filter to slices
with n ≥ 100 and |ROI delta| ≥ 1.5pp and rank by combined effect size.

Output is intentionally noisy — read it as a hypothesis generator, not a list
of rules to deploy. The strong findings are the ones that survive Bonferroni
correction across the ~50 slicers tested.
"""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from ..normalize.paths import PROCESSED

warnings.filterwarnings("ignore")


def _american_to_decimal(a):
    if a is None or not np.isfinite(a) or a == 0:
        return None
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def load():
    wp = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    train = pd.read_parquet(PROCESSED / "train.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")

    sharp = odds[odds["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    for s in ("home", "away"):
        sharp[f"{s}_ml"] = (sharp[f"{s}_ml_close"]
                            .fillna(sharp[f"{s}_ml_current"])
                            .fillna(sharp[f"{s}_ml_top"]))
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
    ).reset_index()
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    ml = agg.merge(matched, on="espn_event_id", how="inner").drop_duplicates("game_pk")
    ml["dec_home"] = ml["home_ml"].apply(_american_to_decimal)
    ml["dec_away"] = ml["away_ml"].apply(_american_to_decimal)

    keep_train = [c for c in train.columns if c in (
        "game_pk", "game_date", "park_runs_factor", "weather_temp_f",
        "win_pct_l30_h", "win_pct_l30_a",
        "run_diff_l10_h", "run_diff_l10_a",
        "off_xwoba_l30_h", "off_xwoba_l30_a",
        "starter_days_rest_h", "starter_days_rest_a",
        "starter_xwoba_l15_h", "starter_xwoba_l15_a",
    )]
    # wp.game_date is datetime; train.game_date may be string. Normalize.
    train = train.copy()
    train["game_date"] = pd.to_datetime(train["game_date"])
    wp["game_date"] = pd.to_datetime(wp["game_date"])
    df = wp.merge(train[keep_train], on=["game_pk", "game_date"], how="left", suffixes=("", "_t"))
    df = df.merge(games[["game_pk", "first_pitch_utc", "day_night", "double_header"]],
                  on="game_pk", how="left")
    df = df.merge(ml[["game_pk", "dec_home", "dec_away"]], on="game_pk", how="left")
    df["first_pitch_utc"] = pd.to_datetime(df["first_pitch_utc"], utc=True, errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["p_home", "market_p_home", "home_win", "dec_home", "dec_away"]).copy()

    # Strategy outcomes
    df["model_home"] = df["p_home"] >= 0.5
    df["mkt_home"] = df["market_p_home"] >= 0.5
    df["dec_pick_model"] = np.where(df["model_home"], df["dec_home"], df["dec_away"])
    df["dec_pick_flip"]  = np.where(df["model_home"], df["dec_away"], df["dec_home"])
    df["dec_pick_mkt"]   = np.where(df["mkt_home"], df["dec_home"], df["dec_away"])
    won_home = df["home_win"] == 1
    df["mod_correct"]  = np.where(df["model_home"], won_home, ~won_home)
    df["flp_correct"]  = ~df["mod_correct"]
    df["mkt_correct"]  = np.where(df["mkt_home"], won_home, ~won_home)
    df["dog_correct"]  = np.where(df["mkt_home"], ~won_home, won_home)  # bet road/home dog
    df["home_correct"] = won_home
    df["away_correct"] = ~won_home

    df["pnl_model"] = np.where(df["mod_correct"], df["dec_pick_model"] - 1, -1.0)
    df["pnl_flip"]  = np.where(df["flp_correct"], df["dec_pick_flip"]  - 1, -1.0)
    df["pnl_mkt"]   = np.where(df["mkt_correct"], df["dec_pick_mkt"]   - 1, -1.0)
    df["pnl_dog"]   = np.where(df["dog_correct"], np.where(df["mkt_home"], df["dec_away"], df["dec_home"]) - 1, -1.0)
    df["pnl_home"]  = np.where(won_home, df["dec_home"] - 1, -1.0)
    df["pnl_away"]  = np.where(~won_home, df["dec_away"] - 1, -1.0)

    df["dow"] = df["game_date"].dt.day_name()
    df["month"] = df["game_date"].dt.month
    df["p_pick"] = np.maximum(df["p_home"], 1 - df["p_home"])
    df["p_dog"]  = 1 - df["p_pick"]
    return df


GLOBAL_BASELINE = {}  # filled in main


def test_slice(g, label):
    """Return a dict of stats for one slice."""
    n = len(g)
    if n < 50:
        return None
    won_home_rate = g["home_win"].mean()
    cands = {
        "model":  g["pnl_model"].mean(),
        "flip":   g["pnl_flip"].mean(),
        "market": g["pnl_mkt"].mean(),
        "dog":    g["pnl_dog"].mean(),
        "home":   g["pnl_home"].mean(),
        "away":   g["pnl_away"].mean(),
        "skip":   0.0,
    }
    best = max(cands, key=cands.get)
    best_roi = cands[best]
    # Delta vs baseline (using model as baseline)
    delta_vs_model = best_roi - GLOBAL_BASELINE.get("model_roi", 0)

    # Stability: paired t-test of best vs model
    if best != "model" and best != "skip":
        pnl_best = {
            "flip": g["pnl_flip"], "market": g["pnl_mkt"],
            "dog": g["pnl_dog"], "home": g["pnl_home"], "away": g["pnl_away"],
        }[best]
        _, p_val = stats.ttest_rel(pnl_best, g["pnl_model"])
    else:
        p_val = 1.0

    return {
        "slice": label, "n": n,
        "home_rate": won_home_rate,
        "mod_acc": g["mod_correct"].mean(),
        "mkt_acc": g["mkt_correct"].mean(),
        "mod_roi": cands["model"],
        "flip_roi": cands["flip"],
        "mkt_roi": cands["market"],
        "dog_roi": cands["dog"],
        "home_roi": cands["home"],
        "away_roi": cands["away"],
        "best": best,
        "best_roi": best_roi,
        "delta_vs_model": delta_vs_model,
        "p_val": p_val,
    }


def main() -> None:
    df = load()
    GLOBAL_BASELINE["model_roi"] = df["pnl_model"].mean()
    GLOBAL_BASELINE["mkt_roi"] = df["pnl_mkt"].mean()
    print(f"GLOBAL: n={len(df):,}  mod_acc={df['mod_correct'].mean():.4f}  "
          f"mkt_acc={df['mkt_correct'].mean():.4f}  "
          f"mod_roi={GLOBAL_BASELINE['model_roi']:+.4f}  mkt_roi={GLOBAL_BASELINE['mkt_roi']:+.4f}")
    print()

    rows: list[dict] = []

    # ── DAY OF WEEK ──
    for dow, g in df.groupby("dow"):
        r = test_slice(g, f"DoW={dow}")
        if r: rows.append({"dim": "day_of_week", **r})

    # ── MONTH ──
    for m, g in df.groupby("month"):
        r = test_slice(g, f"Month={m}")
        if r: rows.append({"dim": "month", **r})

    # ── DAY vs NIGHT game (from games.parquet) ──
    for dn, g in df.groupby("day_night"):
        if dn is None: continue
        r = test_slice(g, f"day_night={dn}")
        if r: rows.append({"dim": "day_night", **r})

    # ── DOUBLEHEADER ──
    if "double_header" in df.columns:
        for dh, g in df.groupby(df["double_header"].fillna("N")):
            r = test_slice(g, f"DH={dh}")
            if r: rows.append({"dim": "double_header", **r})

    # ── DECIMAL ODDS BAND (favorite side) ──
    df["fav_dec"] = np.minimum(df["dec_home"], df["dec_away"])
    band_edges = [1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        mask = (df["fav_dec"] >= lo) & (df["fav_dec"] < hi)
        g = df[mask]
        r = test_slice(g, f"fav_dec=[{lo:.2f},{hi:.2f})")
        if r: rows.append({"dim": "fav_odds_band", **r})

    # ── DOG ODDS BAND ──
    df["dog_dec"] = np.maximum(df["dec_home"], df["dec_away"])
    band_edges = [1.80, 2.00, 2.20, 2.50, 3.00, 3.50, 5.00]
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        mask = (df["dog_dec"] >= lo) & (df["dog_dec"] < hi)
        g = df[mask]
        r = test_slice(g, f"dog_dec=[{lo:.2f},{hi:.2f})")
        if r: rows.append({"dim": "dog_odds_band", **r})

    # ── HOME FAVORITE vs ROAD FAVORITE ──
    for label, mask in [
        ("road_favorite", df["dec_away"] < df["dec_home"]),
        ("home_favorite", df["dec_home"] < df["dec_away"]),
        ("pickem",        np.abs(df["dec_home"] - df["dec_away"]) < 0.10),
    ]:
        g = df[mask]
        r = test_slice(g, label)
        if r: rows.append({"dim": "fav_location", **r})

    # ── HEAVY FAV (model says >=65%) — does the model over-trust them? ──
    for lo, hi, name in [(0.50, 0.55, "model_lean"),
                         (0.55, 0.60, "model_modfav"),
                         (0.60, 0.65, "model_fav"),
                         (0.65, 0.75, "model_heavyfav"),
                         (0.75, 0.95, "model_lockfav")]:
        g = df[df["p_pick"].between(lo, hi, inclusive="left")]
        r = test_slice(g, f"{name}({lo:.2f}–{hi:.2f})")
        if r: rows.append({"dim": "model_confidence", **r})

    # ── PARK RUN ENVIRONMENT ──
    if "park_runs_factor" in df.columns:
        prf = df["park_runs_factor"].dropna()
        if len(prf) > 100:
            q33, q66 = prf.quantile(0.33), prf.quantile(0.66)
            for label, mask in [
                (f"park_pitcher (≤{q33:.2f})", df["park_runs_factor"] <= q33),
                (f"park_neutral ({q33:.2f}–{q66:.2f})", df["park_runs_factor"].between(q33, q66)),
                (f"park_hitter (>{q66:.2f})", df["park_runs_factor"] > q66),
            ]:
                g = df[mask]
                r = test_slice(g, label)
                if r: rows.append({"dim": "park", **r})

    # ── WEATHER (temp) ──
    if "weather_temp_f" in df.columns:
        for label, mask in [
            ("cold (≤55°F)",  df["weather_temp_f"] <= 55),
            ("mild (55–75°F)", df["weather_temp_f"].between(55, 75)),
            ("hot (≥85°F)",   df["weather_temp_f"] >= 85),
        ]:
            g = df[mask]
            r = test_slice(g, label)
            if r: rows.append({"dim": "weather", **r})

    # ── PITCHER REST (home & away) ──
    for side in ("h", "a"):
        col = f"starter_days_rest_{side}"
        if col not in df.columns: continue
        for label, mask in [
            (f"{side}_3day_rest",  df[col] == 3),
            (f"{side}_4day_rest",  df[col] == 4),
            (f"{side}_5+day_rest", df[col] >= 5),
        ]:
            g = df[mask]
            r = test_slice(g, label)
            if r: rows.append({"dim": "pitcher_rest", **r})

    # ── RECENT FORM: hot vs cold teams ──
    if "win_pct_l30_h" in df.columns and "win_pct_l30_a" in df.columns:
        df["form_gap"] = df["win_pct_l30_h"] - df["win_pct_l30_a"]
        for label, mask in [
            ("home_much_hotter (gap>0.15)",  df["form_gap"] > 0.15),
            ("away_much_hotter (gap<-0.15)", df["form_gap"] < -0.15),
            ("similar_form (|gap|<0.05)",    df["form_gap"].abs() < 0.05),
        ]:
            g = df[mask]
            r = test_slice(g, label)
            if r: rows.append({"dim": "form_gap", **r})

    # ── HOME TEAM BIAS BY RECENT RUN DIFF ──
    if "run_diff_l10_h" in df.columns and "run_diff_l10_a" in df.columns:
        df["rd_gap"] = df["run_diff_l10_h"] - df["run_diff_l10_a"]
        for label, mask in [
            ("home_rd_dominant (rd>2)", df["rd_gap"] > 2),
            ("away_rd_dominant (rd<-2)", df["rd_gap"] < -2),
        ]:
            g = df[mask]
            r = test_slice(g, label)
            if r: rows.append({"dim": "run_diff_gap", **r})

    # ── BY TEAM (home & away separately, filter n>=80) ──
    for side in ("home", "away"):
        col = f"{side}_team_abbrev"
        for team, g in df.groupby(col):
            if len(g) < 80: continue
            r = test_slice(g, f"{team}_as_{side}")
            if r: rows.append({"dim": f"team_{side}", **r})

    # ── BY SEASON ──
    for s, g in df.groupby("season"):
        r = test_slice(g, f"season={s}")
        if r: rows.append({"dim": "season", **r})

    # ── INTERACTIONS: model_heavyfav × home/road ──
    for side in ("home", "away"):
        col = "model_home" if side == "home" else (~df["model_home"]).rename(None)
        mask = (df["p_pick"] >= 0.60) & (df["model_home"] if side == "home" else ~df["model_home"])
        g = df[mask]
        r = test_slice(g, f"heavyfav_at_{side}")
        if r: rows.append({"dim": "interaction", **r})

    # ── INTERACTIONS: low_conf × disagreement ──
    df["agree"] = df["model_home"] == df["mkt_home"]
    for label, mask in [
        ("agree + low_conf",  df["agree"] & df["p_pick"].between(0.50, 0.58)),
        ("disagree + low_conf", ~df["agree"] & df["p_pick"].between(0.50, 0.58)),
        ("agree + high_conf",  df["agree"] & (df["p_pick"] >= 0.62)),
        ("disagree + high_conf", ~df["agree"] & (df["p_pick"] >= 0.62)),
    ]:
        g = df[mask]
        r = test_slice(g, label)
        if r: rows.append({"dim": "agreement", **r})

    res = pd.DataFrame(rows)

    # ── SUMMARY: top non-model strategies by delta vs model baseline ──
    print("=" * 110)
    print(f"GLOBAL MODEL ROI = {GLOBAL_BASELINE['model_roi']:+.4f}")
    print("TOP slices where a NON-MODEL strategy beats global model ROI by ≥2pp (n≥100):")
    print("=" * 110)
    top = res[(res["n"] >= 100) & (res["best"] != "model")
              & (res["delta_vs_model"] >= 0.020)].copy()
    top = top.sort_values("delta_vs_model", ascending=False).head(25)
    print(f"{'dim':<18} {'slice':<35} {'n':>5} {'best':>7} {'roi':>8} "
          f"{'Δ vs mod':>9} {'p':>6}")
    for _, r in top.iterrows():
        print(f"{r['dim']:<18} {r['slice'][:35]:<35} {r['n']:>5} {r['best']:>7} "
              f"{r['best_roi']:>+8.4f} {r['delta_vs_model']*100:>+8.2f}pp {r['p_val']:>6.3f}")

    # ── SUMMARY: where model SHINES (model is best, ROI ≥ +2%) ──
    print()
    print("=" * 110)
    print("SLICES WHERE THE MODEL ALREADY WINS (model is best AND ROI > +2%):")
    print("=" * 110)
    win = res[(res["n"] >= 100) & (res["best"] == "model")
              & (res["mod_roi"] >= 0.020)].copy()
    win = win.sort_values("mod_roi", ascending=False).head(20)
    print(f"{'dim':<18} {'slice':<35} {'n':>5} {'mod_roi':>8} {'mkt_roi':>8}")
    for _, r in win.iterrows():
        print(f"{r['dim']:<18} {r['slice'][:35]:<35} {r['n']:>5} {r['mod_roi']:>+8.4f} "
              f"{r['mkt_roi']:>+8.4f}")

    # ── SUMMARY: dog-bias slices (always-bet-dog wins) ──
    print()
    print("=" * 110)
    print("SLICES WHERE BLIND-BETTING THE DOG IS PROFITABLE (n≥100, dog_roi > +2%):")
    print("=" * 110)
    dog = res[(res["n"] >= 100) & (res["dog_roi"] >= 0.020)].copy()
    dog = dog.sort_values("dog_roi", ascending=False).head(20)
    print(f"{'dim':<18} {'slice':<35} {'n':>5} {'dog_roi':>8} {'home_roi':>8} {'away_roi':>8}")
    for _, r in dog.iterrows():
        print(f"{r['dim']:<18} {r['slice'][:35]:<35} {r['n']:>5} {r['dog_roi']:>+8.4f} "
              f"{r['home_roi']:>+8.4f} {r['away_roi']:>+8.4f}")

    # ── SUMMARY: home-bias / away-bias by team ──
    print()
    print("=" * 110)
    print("TEAM EXTREMES — home/away ROI (n≥80):")
    print("=" * 110)
    teams = res[(res["dim"].str.startswith("team_")) & (res["n"] >= 80)].copy()
    teams = teams.sort_values("best_roi", ascending=False).head(15)
    print(f"{'slice':<25} {'n':>5} {'best':>7} {'best_roi':>8} {'mod_roi':>8} {'home_roi':>8} {'away_roi':>8}")
    for _, r in teams.iterrows():
        print(f"{r['slice']:<25} {r['n']:>5} {r['best']:>7} {r['best_roi']:>+8.4f} "
              f"{r['mod_roi']:>+8.4f} {r['home_roi']:>+8.4f} {r['away_roi']:>+8.4f}")

    # ── Bonferroni note ──
    print()
    print(f"Tests performed: {len(res)}.  Bonferroni-corrected α=0.05 → p < {0.05/len(res):.5f}")
    surv = res[(res["p_val"] < 0.05/len(res)) & (res["best"] != "model")
               & (res["best"] != "skip") & (res["n"] >= 100)]
    print(f"Slices surviving Bonferroni (real signal vs ruido): {len(surv)}")
    if len(surv):
        for _, r in surv.iterrows():
            print(f"  · {r['dim']}/{r['slice']}: best={r['best']} roi={r['best_roi']:+.4f}  p={r['p_val']:.5f}")

    res.to_parquet(PROCESSED / "pattern_hunt.parquet", index=False)
    print(f"\nFull table written to {PROCESSED / 'pattern_hunt.parquet'} ({len(res)} rows)")


if __name__ == "__main__":
    main()
