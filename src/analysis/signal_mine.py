"""Leakage-aware pattern mining over walk-forward MLB predictions.

This is a hypothesis generator, not production logic.

Design:
  1. Use only walk-forward predictions and pregame/lagged columns.
  2. Discover candidate slices on early months.
  3. Validate the exact same slice/strategy on later months.
  4. Rank by validation accuracy lift, ROI lift, sample size, and stability.

Run:
  python -m src.analysis.signal_mine
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


OUT_DIR = PROCESSED / "signal_mine"
MIN_DISCOVERY_N = 80
MIN_VALIDATION_N = 40
MIN_DAILY_N = 4
DISCOVERY_END = pd.Timestamp("2025-03-31")
STRATEGIES = ("model", "market", "flip_model", "dog", "favorite", "home", "away")


def american_to_decimal(odds: float | int | None) -> float | None:
    if odds is None or not np.isfinite(odds) or odds == 0:
        return None
    return float(1 + (odds / 100.0 if odds > 0 else 100.0 / -odds))


def bin_num(s: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    unique_bins = []
    for value in bins:
        if not unique_bins or value != unique_bins[-1]:
            unique_bins.append(value)
    if len(unique_bins) < 3:
        return pd.Series("NA", index=s.index, dtype="object")
    use_labels = labels[: len(unique_bins) - 1]
    return pd.cut(s, bins=unique_bins, labels=use_labels, include_lowest=True).astype("object").fillna("NA")


def side_value(home_value: pd.Series, away_value: pd.Series, pick_home: pd.Series) -> pd.Series:
    return pd.Series(np.where(pick_home, home_value, away_value), index=home_value.index)


def load_moneylines() -> pd.DataFrame:
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    sharp = odds[odds["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    for side in ("home", "away"):
        sharp[f"{side}_ml"] = (
            sharp[f"{side}_ml_close"]
            .fillna(sharp[f"{side}_ml_current"])
            .fillna(sharp[f"{side}_ml_top"])
        )
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id", as_index=False).agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
    )
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    out = agg.merge(matched, on="espn_event_id", how="inner").drop_duplicates("game_pk")
    out["dec_home"] = out["home_ml"].apply(american_to_decimal)
    out["dec_away"] = out["away_ml"].apply(american_to_decimal)
    return out[["game_pk", "home_ml", "away_ml", "dec_home", "dec_away"]]


def load_base() -> pd.DataFrame:
    wp = pd.read_parquet(PROCESSED / "walkforward_preds.parquet").copy()
    train = pd.read_parquet(PROCESSED / "train.parquet").copy()
    games = pd.read_parquet(PROCESSED / "games.parquet").copy()
    ml = load_moneylines()

    wp["game_date"] = pd.to_datetime(wp["game_date"])
    train["game_date"] = pd.to_datetime(train["game_date"])
    games["game_date"] = pd.to_datetime(games["game_date"])

    keep_train = [
        c for c in train.columns
        if c not in {
            "status", "home_score", "away_score", "home_win", "total_runs",
            "f5_home_score", "f5_away_score", "f5_home_won", "f5_tied",
            "f5_total_runs", "f5_run_diff",
        }
    ]
    df = wp.merge(
        train[keep_train],
        on=["game_pk", "game_date", "season", "home_team_abbrev", "away_team_abbrev"],
        how="left",
        suffixes=("", "_train"),
    )
    df = df.merge(
        games[[
            "game_pk", "first_pitch_utc", "day_night", "double_header",
            "game_number", "venue_name", "venue_city", "roof_type", "turf_type",
            "weather_condition", "weather_temp_f",
        ]],
        on="game_pk",
        how="left",
        suffixes=("", "_game"),
    )
    df = df.merge(ml, on="game_pk", how="left")
    df = df.dropna(subset=["p_home", "p_home_model_raw", "market_p_home", "home_win", "dec_home", "dec_away"])
    df = df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    df["first_pitch_utc"] = pd.to_datetime(df["first_pitch_utc"], utc=True, errors="coerce")
    df["et_date"] = df["first_pitch_utc"].dt.tz_convert("America/New_York").dt.date
    df["slot"] = df.groupby("et_date", dropna=False).cumcount() + 1
    df["dow"] = df["game_date"].dt.day_name()
    df["month"] = df["game_date"].dt.month
    df["season_part"] = np.select(
        [
            df["month"].isin([3, 4]),
            df["month"].isin([5, 6]),
            df["month"].isin([7, 8]),
            df["month"].isin([9, 10]),
        ],
        ["early", "pre_summer", "summer", "stretch"],
        default="other",
    )

    df["home_won"] = df["home_win"].astype(int) == 1
    df["model_home"] = df["p_home"] >= 0.5
    df["raw_model_home"] = df["p_home_model_raw"] >= 0.5
    df["market_home"] = df["market_p_home"] >= 0.5
    df["favorite_home"] = df["dec_home"] < df["dec_away"]
    df["favorite_team"] = np.where(df["favorite_home"], df["home_team_abbrev"], df["away_team_abbrev"])
    df["dog_team"] = np.where(df["favorite_home"], df["away_team_abbrev"], df["home_team_abbrev"])
    df["model_team"] = np.where(df["model_home"], df["home_team_abbrev"], df["away_team_abbrev"])
    df["market_team"] = np.where(df["market_home"], df["home_team_abbrev"], df["away_team_abbrev"])
    df["fav_dec"] = np.minimum(df["dec_home"], df["dec_away"])
    df["dog_dec"] = np.maximum(df["dec_home"], df["dec_away"])
    df["p_model_pick"] = np.maximum(df["p_home"], 1 - df["p_home"])
    df["p_raw_model_pick"] = np.maximum(df["p_home_model_raw"], 1 - df["p_home_model_raw"])
    df["p_market_pick"] = np.maximum(df["market_p_home"], 1 - df["market_p_home"])
    df["model_market_gap"] = df["p_home"] - df["market_p_home"]
    df["abs_model_market_gap"] = df["model_market_gap"].abs()
    df["model_market_disagree"] = df["model_home"] != df["market_home"]
    df["model_picks_favorite"] = df["model_home"] == df["favorite_home"]
    df["raw_model_market_disagree"] = df["raw_model_home"] != df["market_home"]
    df["blend_shift"] = df["p_home"] - df["p_home_model_raw"]
    df["gap_sign"] = np.select(
        [df["model_market_gap"] <= -0.03, df["model_market_gap"] >= 0.03],
        ["market_home_gap", "model_home_gap"],
        default="small_gap",
    )

    add_engineered_columns(df)
    add_strategy_columns(df)
    add_pattern_columns(df)
    return df


def add_engineered_columns(df: pd.DataFrame) -> None:
    pairs = {
        "team_win_l30_edge": ("win_pct_l30_h", "win_pct_l30_a", 1),
        "team_run_l10_edge": ("run_diff_l10_h", "run_diff_l10_a", 1),
        "off_xwoba_edge": ("off_xwoba_l30_h", "off_xwoba_l30_a", 1),
        "off_barrel_edge": ("off_barrel_rate_l30_h", "off_barrel_rate_l30_a", 1),
        "off_k_edge": ("off_k_pct_l30_h", "off_k_pct_l30_a", -1),
        "off_bb_edge": ("off_bb_pct_l30_h", "off_bb_pct_l30_a", 1),
        "def_xwoba_edge": ("def_xwoba_l30_h", "def_xwoba_l30_a", -1),
        "starter_xwoba_edge": ("starter_xwoba_l15_a", "starter_xwoba_l15_h", 1),
        "starter_barrel_edge": ("starter_barrel_against_l15_a", "starter_barrel_against_l15_h", 1),
        "starter_k_edge": ("starter_k_pct_l15_h", "starter_k_pct_l15_a", 1),
        "starter_bb_edge": ("starter_bb_pct_l15_a", "starter_bb_pct_l15_h", 1),
        "starter_velo_edge": ("starter_velo_mean_l15_h", "starter_velo_mean_l15_a", 1),
        "starter_pa_edge": ("starter_pa_l15_h", "starter_pa_l15_a", 1),
        "starter_rest_edge": ("starter_days_rest_h", "starter_days_rest_a", 1),
        "team_rest_edge": ("days_since_last_game_h", "days_since_last_game_a", 1),
        "team_runs_scored_edge": ("runs_scored_l10_h", "runs_scored_l10_a", 1),
        "team_runs_allowed_edge": ("runs_allowed_l10_a", "runs_allowed_l10_h", 1),
        "bullpen_1d_fresh_edge": ("bullpen_ip_l1d_a", "bullpen_ip_l1d_h", 1),
        "bullpen_fresh_edge": ("bullpen_ip_l3d_a", "bullpen_ip_l3d_h", 1),
        "bullpen_5d_fresh_edge": ("bullpen_ip_l5d_a", "bullpen_ip_l5d_h", 1),
        "bullpen_apps_edge": ("bullpen_apps_l3d_a", "bullpen_apps_l3d_h", 1),
        "bullpen_runs_edge": ("bullpen_runs_l5_a", "bullpen_runs_l5_h", 1),
        "lineup_xwoba_edge": ("lineup_xwoba_vs_sp_hand_h", "lineup_xwoba_vs_sp_hand_a", 1),
        "lineup_top4_xwoba_edge": ("lineup_top4_xwoba_vs_sp_hand_h", "lineup_top4_xwoba_vs_sp_hand_a", 1),
        "lineup_barrel_edge": ("lineup_barrel_vs_sp_hand_h", "lineup_barrel_vs_sp_hand_a", 1),
        "lineup_k_edge": ("lineup_k_pct_vs_sp_hand_a", "lineup_k_pct_vs_sp_hand_h", 1),
        "lineup_bb_edge": ("lineup_bb_pct_vs_sp_hand_h", "lineup_bb_pct_vs_sp_hand_a", 1),
        "park_off_xwoba_edge": ("park_off_xwoba_h", "park_off_xwoba_a", 1),
        "park_def_xwoba_edge": ("park_def_xwoba_a", "park_def_xwoba_h", 1),
        "park_starter_xwoba_edge": ("park_starter_xwoba_a", "park_starter_xwoba_h", 1),
        "park_run_edge": ("park_run_diff_h", "park_run_diff_a", 1),
        "elo_edge": ("home_elo_pre", "away_elo_pre", 1),
        "pyth_l30_edge": ("pyth_wpct_l30_h", "pyth_wpct_l30_a", 1),
        "luck_babip_off_edge": ("babip_off_dev_h", "babip_off_dev_a", 1),
        "luck_lob_def_edge": ("lob_def_dev_a", "lob_def_dev_h", 1),
    }
    for out_col, (home_col, away_col, sign) in pairs.items():
        if home_col in df.columns and away_col in df.columns:
            df[out_col] = sign * (pd.to_numeric(df[home_col], errors="coerce") - pd.to_numeric(df[away_col], errors="coerce"))

    if "series_game_number" in df.columns:
        df["series_game"] = df["series_game_number"].fillna(1).clip(1, 5).astype(int).astype(str)
    if "series_run_diff_so_far" in df.columns:
        df["series_run_diff_bucket"] = bin_num(
            pd.to_numeric(df["series_run_diff_so_far"], errors="coerce"),
            [-99, -5, -1, 1, 5, 99],
            ["home_trailing_big", "home_trailing", "even", "home_leading", "home_leading_big"],
        )
    if "ump_run_impact" in df.columns:
        df["ump_run_bucket"] = bin_num(
            pd.to_numeric(df["ump_run_impact"], errors="coerce"),
            [-99, -0.35, -0.10, 0.10, 0.35, 99],
            ["ump_pitcher_big", "ump_pitcher", "ump_neutral", "ump_hitter", "ump_hitter_big"],
        )
    if "ump_consistency" in df.columns:
        df["ump_consistency_bucket"] = bin_num(
            pd.to_numeric(df["ump_consistency"], errors="coerce"),
            [-99, 0.88, 0.91, 0.94, 99],
            ["ump_loose", "ump_mid", "ump_consistent", "ump_tight"],
        )


def add_strategy_columns(df: pd.DataFrame) -> None:
    pick_home = {
        "model": df["model_home"],
        "market": df["market_home"],
        "flip_model": ~df["model_home"],
        "dog": ~df["favorite_home"],
        "favorite": df["favorite_home"],
        "home": pd.Series(True, index=df.index),
        "away": pd.Series(False, index=df.index),
    }
    for strategy, home_pick in pick_home.items():
        correct = home_pick == df["home_won"]
        dec = np.where(home_pick, df["dec_home"], df["dec_away"])
        df[f"{strategy}_correct"] = correct.astype(float)
        df[f"{strategy}_pnl"] = np.where(correct, dec - 1.0, -1.0)


def add_pattern_columns(df: pd.DataFrame) -> None:
    df["slot_bucket"] = np.select(
        [df["slot"] <= 3, df["slot"].between(4, 7), df["slot"].between(8, 11), df["slot"] >= 12],
        ["early_slate", "mid_slate", "late_slate", "nightcap"],
        default="NA",
    )
    df["fav_loc"] = np.where(df["favorite_home"], "home_fav", "road_fav")
    df["model_loc"] = np.where(df["model_home"], "model_home", "model_away")
    df["dog_loc"] = np.where(df["favorite_home"], "road_dog", "home_dog")
    df["fav_dec_band"] = bin_num(
        df["fav_dec"], [1.0, 1.35, 1.45, 1.55, 1.65, 1.75, 1.90, 2.10],
        ["fav_1.00_1.35", "fav_1.35_1.45", "fav_1.45_1.55", "fav_1.55_1.65", "fav_1.65_1.75", "fav_1.75_1.90", "near_pickem"],
    )
    df["dog_dec_band"] = bin_num(
        df["dog_dec"], [1.0, 1.95, 2.10, 2.30, 2.60, 3.00, 3.60, 9.0],
        ["dog_lt1.95", "dog_1.95_2.10", "dog_2.10_2.30", "dog_2.30_2.60", "dog_2.60_3.00", "dog_3.00_3.60", "dog_3.60_plus"],
    )
    df["model_conf_band"] = bin_num(
        df["p_model_pick"], [0.50, 0.52, 0.55, 0.58, 0.62, 0.66, 0.72, 0.90],
        ["m50_52", "m52_55", "m55_58", "m58_62", "m62_66", "m66_72", "m72_plus"],
    )
    df["market_conf_band"] = bin_num(
        df["p_market_pick"], [0.50, 0.52, 0.55, 0.58, 0.62, 0.66, 0.72, 0.90],
        ["mk50_52", "mk52_55", "mk55_58", "mk58_62", "mk62_66", "mk66_72", "mk72_plus"],
    )
    df["gap_band"] = bin_num(
        df["abs_model_market_gap"], [0.0, 0.015, 0.03, 0.05, 0.08, 0.15, 0.50],
        ["gap_0_1.5", "gap_1.5_3", "gap_3_5", "gap_5_8", "gap_8_15", "gap_15_plus"],
    )
    df["raw_model_conf_band"] = bin_num(
        df["p_raw_model_pick"], [0.50, 0.52, 0.55, 0.58, 0.62, 0.66, 0.72, 0.90],
        ["raw50_52", "raw52_55", "raw55_58", "raw58_62", "raw62_66", "raw66_72", "raw72_plus"],
    )
    df["blend_shift_band"] = bin_num(
        df["blend_shift"], [-1.0, -0.08, -0.03, -0.01, 0.01, 0.03, 0.08, 1.0],
        ["blend_down_big", "blend_down", "blend_down_small", "blend_flat", "blend_up_small", "blend_up", "blend_up_big"],
    )
    if "market_p_home_std" in df.columns:
        df["market_std_band"] = bin_num(
            pd.to_numeric(df["market_p_home_std"], errors="coerce"),
            [-1, 0.005, 0.015, 0.03, 1],
            ["mkt_consensus_tight", "mkt_consensus_mid", "mkt_consensus_wide", "mkt_consensus_chaos"],
        )
    if "market_n_providers" in df.columns:
        df["market_depth_bucket"] = bin_num(
            pd.to_numeric(df["market_n_providers"], errors="coerce"),
            [-1, 2, 4, 8, 99],
            ["few_books", "some_books", "many_books", "deep_market"],
        )
    if "market_over_under" in df.columns:
        df["total_band"] = bin_num(
            pd.to_numeric(df["market_over_under"], errors="coerce"),
            [0, 7.0, 8.0, 8.5, 9.0, 10.0, 20.0],
            ["total_low", "total_7_8", "total_8_8.5", "total_8.5_9", "total_9_10", "total_high"],
        )
    if "weather_temp_f" in df.columns:
        df["temp_bucket"] = bin_num(
            pd.to_numeric(df["weather_temp_f"], errors="coerce"),
            [-99, 55, 65, 75, 85, 140],
            ["cold", "cool", "mild", "warm", "hot"],
        )
    for col in [
        "team_win_l30_edge", "team_run_l10_edge", "off_xwoba_edge",
        "off_barrel_edge", "def_xwoba_edge", "starter_xwoba_edge",
        "starter_barrel_edge", "starter_k_edge", "starter_rest_edge",
        "team_rest_edge", "team_runs_scored_edge", "team_runs_allowed_edge",
        "bullpen_1d_fresh_edge", "bullpen_fresh_edge", "bullpen_5d_fresh_edge",
        "bullpen_apps_edge", "bullpen_runs_edge", "lineup_xwoba_edge",
        "lineup_top4_xwoba_edge", "lineup_barrel_edge", "lineup_k_edge",
        "park_off_xwoba_edge", "park_def_xwoba_edge", "park_run_edge",
        "elo_edge", "pyth_l30_edge",
    ]:
        if col in df.columns:
            q = df[col].quantile([0.25, 0.45, 0.55, 0.75]).to_list()
            df[f"{col}_bucket"] = bin_num(
                df[col],
                [-np.inf, q[0], q[1], q[2], q[3], np.inf],
                [f"{col}_away_big", f"{col}_away", f"{col}_even", f"{col}_home", f"{col}_home_big"],
            )
            df[f"{col}_for_model"] = np.where(df["model_home"], df[col], -df[col])
            q2 = df[f"{col}_for_model"].quantile([0.25, 0.45, 0.55, 0.75]).to_list()
            df[f"{col}_model_bucket"] = bin_num(
                df[f"{col}_for_model"],
                [-np.inf, q2[0], q2[1], q2[2], q2[3], np.inf],
                [f"{col}_against_model_big", f"{col}_against_model", f"{col}_neutral", f"{col}_with_model", f"{col}_with_model_big"],
            )


@dataclass(frozen=True)
class Candidate:
    dims: tuple[str, ...]
    values: tuple[object, ...]
    label: str


def candidate_columns(df: pd.DataFrame) -> list[str]:
    base_cols = [
        "dow", "month", "season_part", "slot", "slot_bucket", "day_night",
        "double_header", "game_number", "roof_type", "turf_type",
        "weather_condition", "fav_loc", "dog_loc", "model_loc",
        "fav_dec_band", "dog_dec_band", "model_conf_band", "market_conf_band",
        "raw_model_conf_band", "gap_band", "gap_sign", "blend_shift_band",
        "model_market_disagree", "raw_model_market_disagree", "model_picks_favorite",
        "total_band", "temp_bucket", "series_game", "series_run_diff_bucket",
        "market_std_band", "market_depth_bucket", "ump_run_bucket", "ump_consistency_bucket",
        "favorite_team", "dog_team", "model_team", "home_team_abbrev", "away_team_abbrev",
    ]
    engineered = [
        c for c in df.columns
        if c.endswith("_bucket") and c not in base_cols
    ]
    cols = [c for c in base_cols + engineered if c in df.columns]
    return cols


def iter_candidates(df: pd.DataFrame, cols: list[str]) -> Iterable[Candidate]:
    for col in cols:
        counts = df[col].astype("object").value_counts(dropna=False)
        for value, count in counts.items():
            if count >= MIN_DISCOVERY_N:
                yield Candidate((col,), (value,), f"{col}={value}")

    priority_cols = [
        "slot", "slot_bucket", "dow", "month", "season_part", "day_night",
        "double_header", "fav_loc", "dog_loc", "model_loc",
        "fav_dec_band", "dog_dec_band", "model_conf_band", "market_conf_band",
        "raw_model_conf_band", "gap_band", "gap_sign", "blend_shift_band",
        "model_market_disagree", "raw_model_market_disagree", "model_picks_favorite",
        "total_band", "temp_bucket", "series_game", "series_run_diff_bucket",
        "market_std_band", "market_depth_bucket", "ump_run_bucket", "ump_consistency_bucket",
        "team_win_l30_edge_bucket", "team_win_l30_edge_model_bucket",
        "team_run_l10_edge_bucket", "team_run_l10_edge_model_bucket",
        "off_xwoba_edge_model_bucket", "off_barrel_edge_model_bucket",
        "def_xwoba_edge_model_bucket",
        "starter_xwoba_edge_bucket", "starter_xwoba_edge_model_bucket",
        "starter_barrel_edge_model_bucket", "starter_k_edge_model_bucket",
        "starter_rest_edge_bucket", "team_rest_edge_model_bucket",
        "bullpen_1d_fresh_edge_model_bucket", "bullpen_fresh_edge_model_bucket",
        "bullpen_5d_fresh_edge_model_bucket", "bullpen_apps_edge_model_bucket",
        "lineup_xwoba_edge_model_bucket", "lineup_top4_xwoba_edge_model_bucket",
        "lineup_barrel_edge_model_bucket", "park_run_edge_model_bucket",
        "elo_edge_model_bucket", "pyth_l30_edge_model_bucket",
    ]
    compact_cols = [
        c for c in priority_cols
        if c not in {"favorite_team", "dog_team", "model_team", "home_team_abbrev", "away_team_abbrev", "weather_condition"}
        and c in cols
        and df[c].nunique(dropna=False) <= 12
    ]
    for left, right in combinations(compact_cols, 2):
        counts = df.groupby([left, right], dropna=False, observed=True).size()
        for values, count in counts.items():
            if count >= MIN_DISCOVERY_N:
                label = f"{left}={values[0]} & {right}={values[1]}"
                yield Candidate((left, right), tuple(values), label)

    anchor_cols = [
        "model_conf_band", "market_conf_band", "model_market_disagree",
        "fav_dec_band", "dog_dec_band", "total_band",
    ]
    context_cols = [
        c for c in compact_cols
        if c not in anchor_cols
        and not c.endswith("_bucket") or c.endswith("_model_bucket")
    ]
    triple_pairs = [
        ("model_conf_band", "model_market_disagree"),
        ("model_conf_band", "fav_dec_band"),
        ("model_conf_band", "market_conf_band"),
        ("fav_dec_band", "total_band"),
        ("dog_dec_band", "market_conf_band"),
    ]
    for first, second in triple_pairs:
        if first not in compact_cols or second not in compact_cols:
            continue
        for third in context_cols:
            if third in {first, second} or third not in compact_cols:
                continue
            counts = df.groupby([first, second, third], dropna=False, observed=True).size()
            for values, count in counts.items():
                if count >= max(MIN_DISCOVERY_N, 90):
                    label = f"{first}={values[0]} & {second}={values[1]} & {third}={values[2]}"
                    yield Candidate((first, second, third), tuple(values), label)


def mask_for(df: pd.DataFrame, cand: Candidate) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in zip(cand.dims, cand.values):
        if pd.isna(val):
            mask &= df[col].isna()
        else:
            mask &= df[col].astype("object") == val
    return mask


def strategy_stats(g: pd.DataFrame, strategy: str) -> dict[str, float]:
    return {
        "acc": float(g[f"{strategy}_correct"].mean()),
        "roi": float(g[f"{strategy}_pnl"].mean()),
        "wins": int(g[f"{strategy}_correct"].sum()),
    }


def evaluate_candidate(df: pd.DataFrame, cand: Candidate) -> dict | None:
    mask = mask_for(df, cand)
    disc = df[mask & (df["game_date"] <= DISCOVERY_END)]
    val = df[mask & (df["game_date"] > DISCOVERY_END)]
    if len(disc) < MIN_DISCOVERY_N or len(val) < MIN_VALIDATION_N:
        return None

    disc_stats = {s: strategy_stats(disc, s) for s in STRATEGIES}
    best = max(STRATEGIES, key=lambda s: (disc_stats[s]["acc"] - disc_stats["model"]["acc"], disc_stats[s]["roi"]))
    val_best = strategy_stats(val, best)
    val_model = strategy_stats(val, "model")
    val_market = strategy_stats(val, "market")
    full_best = strategy_stats(df[mask], best)
    full_model = strategy_stats(df[mask], "model")

    return {
        "label": cand.label,
        "dims": "|".join(cand.dims),
        "values": "|".join(map(str, cand.values)),
        "strategy": best,
        "n_disc": len(disc),
        "n_val": len(val),
        "disc_acc": disc_stats[best]["acc"],
        "disc_model_acc": disc_stats["model"]["acc"],
        "disc_roi": disc_stats[best]["roi"],
        "disc_model_roi": disc_stats["model"]["roi"],
        "val_acc": val_best["acc"],
        "val_model_acc": val_model["acc"],
        "val_market_acc": val_market["acc"],
        "val_roi": val_best["roi"],
        "val_model_roi": val_model["roi"],
        "val_market_roi": val_market["roi"],
        "val_acc_lift": val_best["acc"] - val_model["acc"],
        "val_roi_lift": val_best["roi"] - val_model["roi"],
        "full_acc": full_best["acc"],
        "full_model_acc": full_model["acc"],
        "full_roi": full_best["roi"],
        "full_model_roi": full_model["roi"],
    }


def mine_rules(df: pd.DataFrame) -> pd.DataFrame:
    cols = candidate_columns(df)
    discovery = df[df["game_date"] <= DISCOVERY_END]
    rows = []
    seen = set()
    for cand in iter_candidates(discovery, cols):
        key = (cand.dims, cand.values)
        if key in seen:
            continue
        seen.add(key)
        row = evaluate_candidate(df, cand)
        if row:
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["score"] = (
        out["val_acc_lift"] * 100
        + out["val_roi_lift"] * 8
        + np.log1p(out["n_val"]) / 12
        - (out["disc_acc"] - out["val_acc"]).clip(lower=0) * 4
    )
    out = out.sort_values(["score", "val_acc_lift", "n_val"], ascending=False).reset_index(drop=True)
    return out


def summarize_bad_good(df: pd.DataFrame) -> pd.DataFrame:
    """Find circumstances where favorites/dogs overperform out of sample."""
    rows = []
    cols = [
        c for c in candidate_columns(df)
        if c not in {"favorite_team", "dog_team", "model_team", "home_team_abbrev", "away_team_abbrev"}
    ]
    for cand in iter_candidates(df[df["game_date"] <= DISCOVERY_END], cols):
        mask = mask_for(df, cand)
        val = df[mask & (df["game_date"] > DISCOVERY_END)]
        if len(val) < MIN_VALIDATION_N:
            continue
        dog_acc = val["dog_correct"].mean()
        fav_acc = val["favorite_correct"].mean()
        rows.append({
            "label": cand.label,
            "n_val": len(val),
            "dog_win_rate": dog_acc,
            "fav_win_rate": fav_acc,
            "dog_roi": val["dog_pnl"].mean(),
            "fav_roi": val["favorite_pnl"].mean(),
            "home_dog_share": ((~val["favorite_home"]).mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["dog_score"] = (out["dog_win_rate"] - 0.5) * 100 + out["dog_roi"] * 12 + np.log1p(out["n_val"]) / 10
    return out.sort_values("dog_score", ascending=False).reset_index(drop=True)


def daily_context(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("et_date", dropna=False).agg(
        games=("game_pk", "size"),
        fav_win_rate=("favorite_correct", "mean"),
        dog_win_rate=("dog_correct", "mean"),
        home_win_rate=("home_correct", "mean"),
        avg_total=("market_over_under", "mean"),
        avg_gap=("abs_model_market_gap", "mean"),
        model_acc=("model_correct", "mean"),
        market_acc=("market_correct", "mean"),
    ).reset_index()
    daily = daily[daily["games"] >= MIN_DAILY_N].copy()
    daily["dog_day"] = daily["dog_win_rate"] >= 0.5
    daily["chaos_day"] = daily["avg_gap"] >= daily["avg_gap"].quantile(0.70)
    return daily.sort_values("dog_win_rate", ascending=False)


def write_report(df: pd.DataFrame, rules: pd.DataFrame, dogs: pd.DataFrame, daily: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rules.to_csv(OUT_DIR / "validated_pattern_rules.csv", index=False)
    dogs.to_csv(OUT_DIR / "dog_favorite_contexts.csv", index=False)
    daily.to_csv(OUT_DIR / "daily_contexts.csv", index=False)

    base_lines = [
        "# Signal mine report",
        "",
        f"Rows: {len(df):,}",
        f"Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}",
        f"Discovery <= {DISCOVERY_END.date()}, validation > {DISCOVERY_END.date()}",
        "",
        "## Global validation metrics",
        "",
    ]
    val = df[df["game_date"] > DISCOVERY_END]
    for strategy in STRATEGIES:
        st = strategy_stats(val, strategy)
        base_lines.append(f"- {strategy}: acc={st['acc']:.3f}, roi={st['roi']:+.3f}, n={len(val):,}")

    def markdown_table(frame: pd.DataFrame, cols: list[str]) -> list[str]:
        view = frame[cols].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
            else:
                view[col] = view[col].astype(str)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in view.itertuples(index=False):
            lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
        return lines

    def table_block(title: str, frame: pd.DataFrame, cols: list[str], n: int = 20) -> list[str]:
        lines = ["", f"## {title}", ""]
        if frame.empty:
            return lines + ["No rows."]
        lines += markdown_table(frame.head(n), cols)
        return lines

    base_lines += table_block(
        "Validated rule candidates",
        rules,
        ["label", "strategy", "n_disc", "n_val", "val_acc", "val_model_acc", "val_acc_lift", "val_roi", "val_roi_lift"],
        25,
    )
    base_lines += table_block(
        "Dog contexts",
        dogs,
        ["label", "n_val", "dog_win_rate", "dog_roi", "fav_win_rate", "fav_roi"],
        20,
    )
    base_lines += table_block(
        "Dog-heavy days",
        daily,
        ["et_date", "games", "dog_win_rate", "fav_win_rate", "home_win_rate", "avg_total", "avg_gap"],
        15,
    )

    path = OUT_DIR / "report.md"
    path.write_text("\n".join(base_lines), encoding="utf-8")
    return path


def main() -> None:
    df = load_base()
    rules = mine_rules(df)
    dogs = summarize_bad_good(df)
    daily = daily_context(df)
    report = write_report(df, rules, dogs, daily)

    print(f"Loaded {len(df):,} walk-forward games")
    print(f"Report: {report}")
    print()
    print("GLOBAL VALIDATION")
    val = df[df["game_date"] > DISCOVERY_END]
    for strategy in STRATEGIES:
        st = strategy_stats(val, strategy)
        print(f"  {strategy:10s} acc={st['acc']:.3f} roi={st['roi']:+.3f} n={len(val):,}")
    print()
    print("TOP VALIDATED RULE CANDIDATES")
    if rules.empty:
        print("  none")
    else:
        cols = ["label", "strategy", "n_val", "val_acc", "val_model_acc", "val_acc_lift", "val_roi_lift"]
        print(rules.head(20)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print("TOP DOG CONTEXTS")
    if dogs.empty:
        print("  none")
    else:
        cols = ["label", "n_val", "dog_win_rate", "dog_roi", "fav_win_rate"]
        print(dogs.head(15)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
