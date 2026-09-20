"""Compute ROI (return on investment) and CLV (closing line value) for the
walk-forward predictions, in BOTH modes (confidence and value).

ROI = sum of net P/L / total staked. Positive = profitable.

Pricing assumptions:
  - ML: use actual home_moneyline / away_moneyline from the closing line.
  - Spread & Totals: standard -110 both sides ($1 stake pays $0.909).

CLV note: for BACKTEST, our picks are made at closing (nflverse gives us the
closing spread as the feature). So historical CLV is 0 by construction. CLV
only makes sense for LIVE picks where we record the price we took and later
compare to the close. This module emits per-pick CLV = null for backtest and
computes real CLV from odds snapshots when available.

Output: data/processed/roi_summary.parquet + roi_by_season.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import PROCESSED

STAKE = 1.0
VIG_PRICE = -110  # standard ATS / Total price both sides


def _american_payout(odds: float) -> float:
    """Return net profit on a $1 winning bet at these American odds."""
    if odds > 0:
        return odds / 100.0
    return 100.0 / -odds


def _american_to_prob(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def compute() -> pd.DataFrame:
    preds = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    train = pd.read_parquet(PROCESSED / "train.parquet",
                            columns=["game_id", "season", "home_team", "away_team",
                                     "home_score", "away_score",
                                     "spread_line", "total_line",
                                     "home_moneyline", "away_moneyline"])
    # preds ya trae `season`; usamos el de train (canónico) y evitamos colisión.
    preds = preds.drop(columns=[c for c in ("season",) if c in preds.columns])
    m = preds.merge(train, on="game_id", how="left")
    m = m.dropna(subset=["home_score", "model_home_win_prob"]).copy()

    m["imp_home_ml"] = m["home_moneyline"].apply(_american_to_prob)
    m["imp_away_ml"] = m["away_moneyline"].apply(_american_to_prob)

    ATS_IMPLIED = 0.524  # -110 vig

    rows = []
    for _, g in m.iterrows():
        hs, as_ = g["home_score"], g["away_score"]
        home_won = hs > as_
        push_ml = hs == as_

        # ---- MONEYLINE ----
        p_home = g["model_home_win_prob"]
        # Confidence mode: side w/ higher model prob
        conf_ml_home = p_home >= 0.5
        # Value mode: side w/ higher edge
        edge_h = (p_home - g["imp_home_ml"]) if not pd.isna(g["imp_home_ml"]) else np.nan
        edge_a = ((1 - p_home) - g["imp_away_ml"]) if not pd.isna(g["imp_away_ml"]) else np.nan
        if not (pd.isna(edge_h) or pd.isna(edge_a)):
            val_ml_home = edge_h >= edge_a
            val_ml_edge = max(edge_h, edge_a)
        else:
            val_ml_home = conf_ml_home
            val_ml_edge = np.nan

        for mode_name, side_home, edge in [
            ("confidence", conf_ml_home, edge_h if conf_ml_home else edge_a),
            ("value",      val_ml_home, val_ml_edge),
        ]:
            price = g["home_moneyline"] if side_home else g["away_moneyline"]
            if pd.isna(price):
                continue
            if push_ml:
                pnl = 0.0
            else:
                won = side_home == home_won
                pnl = _american_payout(price) * STAKE if won else -STAKE
            rows.append({
                "game_id": g["game_id"], "season": int(g["season"]), "market": "ml",
                "mode": mode_name, "edge": edge, "stake": STAKE, "pnl": pnl,
                "won": None if push_ml else bool(side_home == home_won),
                "push": push_ml,
            })

        # ---- SPREAD ----
        if not pd.isna(g["spread_line"]):
            margin_adj = hs - as_ - g["spread_line"]
            push_sp = margin_adj == 0
            home_covered = margin_adj > 0
            p_cov = g["model_cover_home_prob"]
            if not pd.isna(p_cov):
                conf_sp_home = p_cov >= 0.5
                val_sp_home = (p_cov - ATS_IMPLIED) >= ((1 - p_cov) - ATS_IMPLIED)
                for mode_name, side_home in [("confidence", conf_sp_home), ("value", val_sp_home)]:
                    if push_sp:
                        pnl = 0.0
                    else:
                        won = side_home == home_covered
                        pnl = _american_payout(VIG_PRICE) * STAKE if won else -STAKE
                    conf = p_cov if side_home else (1 - p_cov)
                    rows.append({
                        "game_id": g["game_id"], "season": int(g["season"]), "market": "spread",
                        "mode": mode_name, "edge": (conf - ATS_IMPLIED) * 100,
                        "stake": STAKE, "pnl": pnl,
                        "won": None if push_sp else bool(side_home == home_covered),
                        "push": push_sp,
                    })

        # ---- TOTAL ----
        if not pd.isna(g["total_line"]):
            total = hs + as_
            push_to = total == g["total_line"]
            went_over = total > g["total_line"]
            p_over = g["model_total_over_prob"]
            if not pd.isna(p_over):
                conf_to_over = p_over >= 0.5
                val_to_over = (p_over - ATS_IMPLIED) >= ((1 - p_over) - ATS_IMPLIED)
                for mode_name, side_over in [("confidence", conf_to_over), ("value", val_to_over)]:
                    if push_to:
                        pnl = 0.0
                    else:
                        won = side_over == went_over
                        pnl = _american_payout(VIG_PRICE) * STAKE if won else -STAKE
                    conf = p_over if side_over else (1 - p_over)
                    rows.append({
                        "game_id": g["game_id"], "season": int(g["season"]), "market": "total",
                        "mode": mode_name, "edge": (conf - ATS_IMPLIED) * 100,
                        "stake": STAKE, "pnl": pnl,
                        "won": None if push_to else bool(side_over == went_over),
                        "push": push_to,
                    })

    picks = pd.DataFrame(rows)
    picks.to_parquet(PROCESSED / "picks_backtest.parquet", index=False)

    # Aggregate by (season, market, mode)
    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        wins = int((g["won"] == True).sum())
        losses = int((g["won"] == False).sum())
        pushes = int(g["push"].sum())
        staked = float(g["stake"].sum())
        pnl = float(g["pnl"].sum())
        roi = pnl / staked * 100 if staked else 0
        settled = wins + losses
        acc = wins / settled * 100 if settled else 0
        return pd.Series({"n": n, "wins": wins, "losses": losses, "pushes": pushes,
                          "staked": staked, "pnl": pnl, "roi_pct": roi, "acc_pct": acc})

    by_season = picks.groupby(["season", "market", "mode"]).apply(agg).reset_index()
    by_season.to_parquet(PROCESSED / "roi_by_season.parquet", index=False)

    summary = picks.groupby(["market", "mode"]).apply(agg).reset_index()
    # Also emit filtered "value with edge >= 2pp"
    value_only = picks[(picks["mode"] == "value") & (picks["edge"] >= 2)]
    if not value_only.empty:
        v = value_only.groupby("market").apply(agg).reset_index()
        v["mode"] = "value_edge2pp"
        summary = pd.concat([summary, v], ignore_index=True)
    value_5 = picks[(picks["mode"] == "value") & (picks["edge"] >= 5)]
    if not value_5.empty:
        v5 = value_5.groupby("market").apply(agg).reset_index()
        v5["mode"] = "value_edge5pp"
        summary = pd.concat([summary, v5], ignore_index=True)

    summary.to_parquet(PROCESSED / "roi_summary.parquet", index=False)

    print("\n=== ROI SUMMARY (unit stake = 1) ===")
    print(summary.to_string(index=False,
        formatters={"staked": "{:.0f}".format, "pnl": "{:+.1f}".format,
                    "roi_pct": "{:+.2f}%".format, "acc_pct": "{:.1f}%".format}))
    print(f"\nSaved:")
    print(f"  {PROCESSED / 'picks_backtest.parquet'}")
    print(f"  {PROCESSED / 'roi_by_season.parquet'}")
    print(f"  {PROCESSED / 'roi_summary.parquet'}")
    return summary


if __name__ == "__main__":
    compute()
