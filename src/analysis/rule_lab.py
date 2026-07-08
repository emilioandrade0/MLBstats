"""Rule lab: discover, select, and holdout-test prediction overrides.

This script answers a stricter question than signal_mine.py:

  Can we choose a small set of rules on past data, then improve accuracy on
  a truly untouched final holdout?

Splits:
  discovery:  walk-forward games <= 2024-12-31
  validation: 2025-03-01 .. 2025-07-31
  holdout:    >= 2025-08-01

Run:
  python -m src.analysis.rule_lab
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .signal_mine import (
    OUT_DIR,
    STRATEGIES,
    candidate_columns,
    iter_candidates,
    load_base,
    mask_for,
)


DISCOVERY_END = pd.Timestamp("2024-12-31")
VALIDATION_START = pd.Timestamp("2025-03-01")
VALIDATION_END = pd.Timestamp("2025-07-31")
HOLDOUT_START = pd.Timestamp("2025-08-01")

MIN_DISCOVERY_N = 70
MIN_VALIDATION_N = 45
MIN_HOLDOUT_N = 30
MAX_RULES = 8


@dataclass(frozen=True)
class Rule:
    label: str
    dims: tuple[str, ...]
    values: tuple[object, ...]
    strategy: str
    n_disc: int
    n_val: int
    n_holdout: int
    val_acc_lift: float
    val_roi_lift: float
    holdout_acc_lift: float
    holdout_roi_lift: float


def split_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "discovery": df["game_date"] <= DISCOVERY_END,
        "validation": (df["game_date"] >= VALIDATION_START) & (df["game_date"] <= VALIDATION_END),
        "holdout": df["game_date"] >= HOLDOUT_START,
    }


def pick_home_for_strategy(df: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "model":
        return df["model_home"].copy()
    if strategy == "market":
        return df["market_home"].copy()
    if strategy == "flip_model":
        return ~df["model_home"]
    if strategy == "dog":
        return ~df["favorite_home"]
    if strategy == "favorite":
        return df["favorite_home"].copy()
    if strategy == "home":
        return pd.Series(True, index=df.index)
    if strategy == "away":
        return pd.Series(False, index=df.index)
    raise ValueError(f"unknown strategy: {strategy}")


def score_pick(df: pd.DataFrame, pick_home: pd.Series) -> dict[str, float]:
    correct = pick_home == df["home_won"]
    dec = np.where(pick_home, df["dec_home"], df["dec_away"])
    pnl = np.where(correct, dec - 1.0, -1.0)
    return {
        "n": int(len(df)),
        "acc": float(correct.mean()) if len(df) else np.nan,
        "roi": float(pnl.mean()) if len(df) else np.nan,
        "wins": int(correct.sum()) if len(df) else 0,
    }


def best_strategy_on(df: pd.DataFrame) -> tuple[str, dict[str, dict[str, float]]]:
    stats = {s: score_pick(df, pick_home_for_strategy(df, s)) for s in STRATEGIES}
    baseline = stats["model"]
    def key(strategy: str) -> tuple[float, float, float]:
        st = stats[strategy]
        return (st["acc"] - baseline["acc"], st["roi"] - baseline["roi"], st["acc"])
    return max(STRATEGIES, key=key), stats


def evaluate_rule(df: pd.DataFrame, cand, masks: dict[str, pd.Series]) -> dict | None:
    mask = mask_for(df, cand)
    disc = df[mask & masks["discovery"]]
    val = df[mask & masks["validation"]]
    holdout = df[mask & masks["holdout"]]
    if len(disc) < MIN_DISCOVERY_N or len(val) < MIN_VALIDATION_N or len(holdout) < MIN_HOLDOUT_N:
        return None

    strategy, disc_stats = best_strategy_on(disc)
    if strategy == "model":
        return None

    val_strategy = score_pick(val, pick_home_for_strategy(val, strategy))
    val_model = score_pick(val, pick_home_for_strategy(val, "model"))
    hold_strategy = score_pick(holdout, pick_home_for_strategy(holdout, strategy))
    hold_model = score_pick(holdout, pick_home_for_strategy(holdout, "model"))

    return {
        "label": cand.label,
        "dims": cand.dims,
        "values": cand.values,
        "strategy": strategy,
        "n_disc": len(disc),
        "n_val": len(val),
        "n_holdout": len(holdout),
        "disc_acc_lift": disc_stats[strategy]["acc"] - disc_stats["model"]["acc"],
        "disc_roi_lift": disc_stats[strategy]["roi"] - disc_stats["model"]["roi"],
        "val_acc": val_strategy["acc"],
        "val_model_acc": val_model["acc"],
        "val_acc_lift": val_strategy["acc"] - val_model["acc"],
        "val_roi_lift": val_strategy["roi"] - val_model["roi"],
        "holdout_acc": hold_strategy["acc"],
        "holdout_model_acc": hold_model["acc"],
        "holdout_acc_lift": hold_strategy["acc"] - hold_model["acc"],
        "holdout_roi_lift": hold_strategy["roi"] - hold_model["roi"],
    }


def discover_candidates(df: pd.DataFrame) -> pd.DataFrame:
    masks = split_masks(df)
    discovery = df[masks["discovery"]]
    cols = lab_candidate_columns(df)
    rows = []
    seen = set()
    for cand in iter_candidates(discovery, cols):
        key = (cand.dims, cand.values)
        if key in seen:
            continue
        seen.add(key)
        row = evaluate_rule(df, cand, masks)
        if row is not None:
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["dims"] = out["dims"].map(lambda v: "|".join(v))
    out["values"] = out["values"].map(lambda v: "|".join(map(str, v)))
    out["validation_score"] = (
        out["val_acc_lift"] * 100
        + out["val_roi_lift"] * 8
        + np.log1p(out["n_val"]) / 10
        - (out["disc_acc_lift"] - out["val_acc_lift"]).clip(lower=0) * 5
    )
    out = out.sort_values(["validation_score", "val_acc_lift", "n_val"], ascending=False).reset_index(drop=True)
    return out


def lab_candidate_columns(df: pd.DataFrame) -> list[str]:
    """Curated high-signal columns for the strict holdout lab.

    signal_mine.py can stay broad and noisy. This lab needs to finish quickly
    and avoid redundant cuts that only add multiple-testing risk.
    """
    preferred = [
        "month", "season_part", "slot_bucket", "day_night",
        "fav_loc", "model_loc",
        "fav_dec_band", "dog_dec_band", "model_conf_band", "market_conf_band",
        "gap_band", "gap_sign", "blend_shift_band",
        "model_market_disagree", "model_picks_favorite",
        "total_band", "temp_bucket", "series_game",
        "market_std_band",
        "team_win_l30_edge_model_bucket", "team_run_l10_edge_model_bucket",
        "off_xwoba_edge_model_bucket",
        "starter_xwoba_edge_bucket", "starter_xwoba_edge_model_bucket",
        "starter_rest_edge_bucket", "team_rest_edge_model_bucket",
        "bullpen_fresh_edge_model_bucket",
        "lineup_xwoba_edge_model_bucket", "lineup_top4_xwoba_edge_model_bucket",
        "elo_edge_model_bucket",
    ]
    available = set(candidate_columns(df))
    return [c for c in preferred if c in available]


def materialize_mask(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    dims = tuple(str(row["dims"]).split("|"))
    values = tuple(str(row["values"]).split("|"))
    mask = pd.Series(True, index=df.index)
    for col, value in zip(dims, values):
        mask &= df[col].astype("object").astype(str) == value
    return mask


def apply_rule_set(df: pd.DataFrame, rules: pd.DataFrame) -> pd.Series:
    pick = df["model_home"].copy()
    for _, row in rules.iterrows():
        mask = materialize_mask(df, row)
        pick.loc[mask] = pick_home_for_strategy(df.loc[mask], row["strategy"])
    return pick


def greedy_select(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    masks = split_masks(df)
    pool = candidates[
        (candidates["val_acc_lift"] > 0.035)
        & (candidates["val_roi_lift"] > 0.0)
        & (candidates["n_val"] >= MIN_VALIDATION_N)
    ].head(250).copy()

    selected_rows = []
    selected = pd.DataFrame()
    current_pick = df["model_home"].copy()
    current_val = score_pick(df[masks["validation"]], current_pick.loc[masks["validation"]])
    current_roi = current_val["roi"]

    for _ in range(MAX_RULES):
        best_idx = None
        best_stats = None
        best_gain = 0.0
        for idx, row in pool.iterrows():
            if any(row["label"] == r["label"] for r in selected_rows):
                continue
            test_rules = pd.concat([selected, pd.DataFrame([row])], ignore_index=True)
            pick = apply_rule_set(df, test_rules)
            stats = score_pick(df[masks["validation"]], pick.loc[masks["validation"]])
            acc_gain = stats["acc"] - current_val["acc"]
            roi_gain = stats["roi"] - current_roi
            gain = acc_gain * 100 + roi_gain * 6
            if acc_gain > 0.0015 and roi_gain > -0.01 and gain > best_gain:
                best_idx = idx
                best_stats = stats
                best_gain = gain
        if best_idx is None or best_stats is None:
            break
        row = pool.loc[best_idx].copy()
        row["combo_val_acc_after"] = best_stats["acc"]
        row["combo_val_roi_after"] = best_stats["roi"]
        selected_rows.append(row.to_dict())
        selected = pd.DataFrame(selected_rows)
        current_pick = apply_rule_set(df, selected)
        current_val = score_pick(df[masks["validation"]], current_pick.loc[masks["validation"]])
        current_roi = current_val["roi"]

    return pd.DataFrame(selected_rows)


def write_outputs(df: pd.DataFrame, candidates: pd.DataFrame, selected: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUT_DIR / "rule_lab_candidates.csv", index=False)
    selected.to_csv(OUT_DIR / "rule_lab_selected.csv", index=False)

    masks = split_masks(df)
    lines = [
        "# Rule lab",
        "",
        f"Rows: {len(df):,}",
        f"Discovery: <= {DISCOVERY_END.date()}",
        f"Validation: {VALIDATION_START.date()} to {VALIDATION_END.date()}",
        f"Holdout: >= {HOLDOUT_START.date()}",
        "",
        "## Baselines",
        "",
    ]
    for split, mask in masks.items():
        lines.append(f"### {split}")
        for strategy in STRATEGIES:
            st = score_pick(df[mask], pick_home_for_strategy(df[mask], strategy))
            lines.append(f"- {strategy}: acc={st['acc']:.3f}, roi={st['roi']:+.3f}, n={st['n']:,}")
        lines.append("")

    def add_table(title: str, frame: pd.DataFrame, cols: list[str], n: int = 20) -> None:
        lines.extend([f"## {title}", ""])
        if frame.empty:
            lines.append("No rows.")
            lines.append("")
            return
        view = frame.head(n)[cols].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: f"{x:.3f}")
            else:
                view[col] = view[col].astype(str)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in view.itertuples(index=False):
            lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
        lines.append("")

    add_table(
        "Top Candidates",
        candidates,
        ["label", "strategy", "n_val", "n_holdout", "val_acc_lift", "val_roi_lift", "holdout_acc_lift", "holdout_roi_lift"],
        30,
    )
    add_table(
        "Greedy Selected Rules",
        selected,
        ["label", "strategy", "n_val", "n_holdout", "combo_val_acc_after", "combo_val_roi_after", "holdout_acc_lift", "holdout_roi_lift"],
        12,
    )

    if not selected.empty:
        pick = apply_rule_set(df, selected)
        lines.extend(["## Combined Rule Set", ""])
        for split, mask in masks.items():
            base = score_pick(df[mask], df.loc[mask, "model_home"])
            combo = score_pick(df[mask], pick.loc[mask])
            lines.append(
                f"- {split}: acc {base['acc']:.3f} -> {combo['acc']:.3f} "
                f"({(combo['acc'] - base['acc']) * 100:+.2f}pp), "
                f"roi {base['roi']:+.3f} -> {combo['roi']:+.3f}"
            )
    path = OUT_DIR / "rule_lab_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    df = load_base()
    candidates = discover_candidates(df)
    selected = greedy_select(df, candidates)
    report = write_outputs(df, candidates, selected)

    masks = split_masks(df)
    print(f"Loaded {len(df):,} walk-forward games")
    print(f"Candidates: {len(candidates):,}")
    print(f"Selected: {len(selected):,}")
    print(f"Report: {report}")
    print()
    print("BASELINE vs COMBO")
    combo_pick = apply_rule_set(df, selected) if not selected.empty else df["model_home"]
    for split, mask in masks.items():
        base = score_pick(df[mask], df.loc[mask, "model_home"])
        combo = score_pick(df[mask], combo_pick.loc[mask])
        print(
            f"  {split:10s} acc {base['acc']:.3f}->{combo['acc']:.3f} "
            f"({(combo['acc'] - base['acc']) * 100:+.2f}pp) "
            f"roi {base['roi']:+.3f}->{combo['roi']:+.3f}"
        )
    if not selected.empty:
        print()
        print("SELECTED RULES")
        cols = ["label", "strategy", "n_val", "n_holdout", "holdout_acc_lift", "holdout_roi_lift"]
        print(selected[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
