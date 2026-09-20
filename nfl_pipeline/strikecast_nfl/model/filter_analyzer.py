"""Analyze accuracy + ROI on any subset of games matching a boolean AND of filters.

Given a list of filter names, returns per-market metrics computed on the
picks_backtest.parquet subset. Designed to be called from the API.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..paths import PROCESSED


def analyze(filters: Iterable[str], mode: str = "confidence",
            market: str | None = None, min_edge: float | None = None) -> dict:
    picks = pd.read_parquet(PROCESSED / "picks_backtest.parquet")
    gf = pd.read_parquet(PROCESSED / "game_filters.parquet")

    picks = picks.merge(gf, on="game_id", how="left")
    picks = picks[picks["mode"] == mode]
    if market:
        picks = picks[picks["market"] == market]
    if min_edge is not None:
        picks = picks[picks["edge"] >= min_edge]

    for f in filters:
        if f not in picks.columns:
            return {"error": f"unknown filter: {f}",
                    "available": [c for c in gf.columns if c != "game_id"]}
        picks = picks[picks[f] == True]

    def bucket(g: pd.DataFrame) -> dict:
        n = len(g)
        wins = int((g["won"] == True).sum())
        losses = int((g["won"] == False).sum())
        pushes = int(g["push"].sum())
        settled = wins + losses
        staked = float(g["stake"].sum())
        pnl = float(g["pnl"].sum())
        return {
            "n": n, "wins": wins, "losses": losses, "pushes": pushes,
            "acc_pct": (wins / settled * 100) if settled else 0.0,
            "roi_pct": (pnl / staked * 100) if staked else 0.0,
            "pnl": pnl,
        }

    by_market: dict[str, dict] = {}
    for m in ["ml", "spread", "total"]:
        sub = picks[picks["market"] == m]
        if not sub.empty:
            by_market[m] = bucket(sub)

    return {
        "filters": list(filters),
        "mode": mode,
        "min_edge": min_edge,
        "n_games": int(picks["game_id"].nunique()),
        "by_market": by_market,
    }
