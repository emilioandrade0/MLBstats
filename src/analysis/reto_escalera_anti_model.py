"""Test the 'predict badly, then flip it' Reto Escalera idea honestly."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .reto_escalera_backtest import (
    DEV_END,
    HOLDOUT_START,
    OUT,
    Config,
    _daily_picks,
    _universe,
)


def _period_row(picks: pd.DataFrame, period: str) -> dict:
    good = picks["won"].astype(bool).to_numpy()
    anti = ~good
    # The public pick flips the anti-model side, so it is exactly `good` again.
    flipped = ~anti
    assert np.array_equal(flipped, good)
    return {
        "period": period,
        "n_days": int(len(picks)),
        "normal_accuracy": float(good.mean()),
        "deliberately_bad_accuracy": float(anti.mean()),
        "flipped_anti_accuracy": float(flipped.mean()),
        "changed_final_picks": int(np.count_nonzero(flipped != good)),
    }


def main() -> None:
    # Exact production-like ranking: min_prob/agreement are soft penalties.
    cfg = Config("current_like_agree", 1.80, 2.20, "model", "confidence", .55, True, True)
    universe = _universe()
    dev = _daily_picks(universe[universe.game_date <= DEV_END], cfg)
    hold = _daily_picks(universe[universe.game_date >= HOLDOUT_START], cfg)
    rows = [_period_row(dev, "dev_through_2025"), _period_row(hold, "holdout_2026")]
    payload = {
        "method": "binary anti-model p_bad(home)=1-p_model(home), then publish opposite side",
        "conclusion": "double inversion is algebraically identical to the original final pick",
        "periods": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "anti_model_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
