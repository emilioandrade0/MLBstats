"""Model's rolling accuracy per team — a self-referential lagged feature.

Idea: some teams the model predicts well, others badly. If we compute the model's
LAGGED accuracy on team X (games strictly before today), and feed that as a
feature, the model can learn "when I've been bad on X lately, adjust my raw
p_home toward the market" (or in the opposite direction if the pattern reverses).

Legality of the feature:
  Source is walkforward_preds.parquet — those are OUT-OF-FOLD predictions
  (walkforward retrains monthly, each month's test set was NEVER in that month's
  train). So using preds[game_date < D] to build a feature for a game on date D
  is strictly lagged, no leakage.

Features per (game_pk, team):
  team_model_acc_l30   — model's accuracy on this team's last 30 games (LAGGED)
  team_model_n_l30     — n games used (0-30) — near-0 = feature not reliable

Output: data/processed/features_team_model_acc.parquet
Key: (game_pk, team)

Run standalone:
  python -m src.features.team_model_acc
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

OUT = PROCESSED / "features_team_model_acc.parquet"
WINDOW = 30


def _segmented_thr(mp: np.ndarray) -> np.ndarray:
    out = np.full_like(mp, 0.475, dtype=float)
    finite = ~np.isnan(mp)
    out[finite & (mp > 0)    & (mp <= 0.42)] = 0.495
    out[finite & (mp > 0.42) & (mp <= 0.50)] = 0.480
    out[finite & (mp > 0.50) & (mp <= 0.60)] = 0.495
    return out


def build() -> Path:
    src = PROCESSED / "walkforward_preds.parquet"
    if not src.exists():
        raise SystemExit(
            "walkforward_preds.parquet not found. "
            "Run `python -m src.model.walkforward` first to generate the OOF preds."
        )
    p = pd.read_parquet(src)
    p["game_date"] = pd.to_datetime(p["game_date"])
    # Right/wrong flag using the same segmented threshold logic the model uses
    p["_t"] = _segmented_thr(p["market_p_home"].to_numpy())
    p["pred_home"] = (p["p_home"] > p["_t"]).astype(int)
    p["correct"] = (p["pred_home"] == p["home_win"]).astype(int)

    # Long format: one row per (team, game) so we can rolling-per-team
    home = p[["game_pk", "game_date", "home_team_abbrev", "correct"]].copy()
    home.columns = ["game_pk", "game_date", "team", "correct"]
    away = p[["game_pk", "game_date", "away_team_abbrev", "correct"]].copy()
    away.columns = ["game_pk", "game_date", "team", "correct"]
    tg = pd.concat([home, away], ignore_index=True)
    tg = tg.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

    grp = tg.groupby("team", sort=False)
    # LAGGED rolling mean — shift(1) so game_pk's OWN correctness doesn't leak in
    tg["team_model_acc_l30"] = grp["correct"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=5).mean()
    )
    tg["team_model_n_l30"] = grp["correct"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=1).count()
    )

    out = tg[["game_pk", "team", "team_model_acc_l30", "team_model_n_l30"]].copy()
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out):,} rows)")
    print(f"  team_model_acc_l30  mean={out['team_model_acc_l30'].mean():.4f}  "
          f"std={out['team_model_acc_l30'].std():.4f}  "
          f"NaN={out['team_model_acc_l30'].isna().mean():.2%}")
    print(f"  team_model_n_l30    median={out['team_model_n_l30'].median():.0f}  "
          f"max={out['team_model_n_l30'].max():.0f}")

    # Signal check: does team_model_acc_l30 correlate with model correctness NOW?
    tg["correct_now"] = grp["correct"].transform(lambda s: s)  # current game's correctness
    chk = tg.dropna(subset=["team_model_acc_l30", "correct_now"])
    r = chk["team_model_acc_l30"].corr(chk["correct_now"])
    print(f"\nCorrelacion (team_model_acc_l30 lagged) vs (correct today): r={r:.4f}")
    print(f"  (r>0.05 = señal explotable, r<0.02 = ruido)")
    return OUT


if __name__ == "__main__":
    build()
