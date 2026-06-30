"""Assemble train.parquet — one row per game with all features (home/away diff) + targets.

Output: data/processed/train.parquet
Targets:
  home_win  (0/1)
  total_runs  (regression)
Features (per game, diffed home minus away where it makes sense):
  is_home_advantage (always 1)
  rest_diff = home_days_rest - away_days_rest
  run_diff_l10_h, run_diff_l10_a, run_diff_l10_diff
  off_xwoba_l30_h/_a/_diff   (home minus away offense)
  def_xwoba_l30_h/_a/_diff
  starter_xwoba_l15_h/_a/_diff
  starter_k_pct_l15_h/_a/_diff
  starter_bb_pct_l15_h/_a/_diff
  starter_velo_h/_a
  park_runs_factor
  weather_temp_f
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    f5_path = PROCESSED / "f5_targets.parquet"
    f5_targets = pd.read_parquet(f5_path) if f5_path.exists() else None
    team_feat = pd.read_parquet(PROCESSED / "features_team.parquet")
    pit_feat = pd.read_parquet(PROCESSED / "features_pitcher.parquet")
    # Pitcher L5 (last 5 starts) tested — -0.35pp acc, -1.9 AUC vs baseline.
    # The 5-start window is too noisy: pitcher variance star-to-start is high,
    # so L15 already captures the stable signal. Kept on disk, disabled here.
    park = pd.read_parquet(PROCESSED / "features_park.parquet")
    market = pd.read_parquet(PROCESSED / "features_market.parquet")
    lineup_path = PROCESSED / "features_lineup.parquet"
    lineup = pd.read_parquet(lineup_path) if lineup_path.exists() else None
    elo_path = PROCESSED / "features_elo.parquet"
    elo = pd.read_parquet(elo_path) if elo_path.exists() else None
    bp_path = PROCESSED / "features_bullpen.parquet"
    bullpen = pd.read_parquet(bp_path) if bp_path.exists() else None
    sc_path = PROCESSED / "features_series.parquet"
    series_ctx = pd.read_parquet(sc_path) if sc_path.exists() else None
    ump_path = PROCESSED / "features_umpire.parquet"
    ump = pd.read_parquet(ump_path) if ump_path.exists() else None
    # Arsenal features were tested (lineup × pitcher pitch-type matchup) but
    # degraded walk-forward performance — overfitting noise. Kept on disk but
    # not merged into train. To re-enable, set arsenal = pd.read_parquet(...).
    arsenal = None
    tanking = None
    # Pythagorean + cluster luck tested but degraded walk-forward (collinear).
    # Kept on disk; uncomment to re-enable individually for further experiments.
    pythag = None
    cluster_luck = None
    cb_path = PROCESSED / "features_comeback.parquet"
    comeback = pd.read_parquet(cb_path) if cb_path.exists() else None
    # lineup_recent (L15 top-4 batter form): improves model AUC by +7pts and
    # log_loss by -0.0024 in walk-forward. High-conf band gains +0.99pp acc.
    # Operationally shifts the edge threshold so the production 5pp filter
    # needs re-calibration to compensate (handled in _value_block).
    lr_path = PROCESSED / "features_lineup_recent.parquet"
    lineup_recent = pd.read_parquet(lr_path) if lr_path.exists() else None
    # burn features (yesterday's bullpen/extra-innings — "carne al asador") tested
    # walk-forward (-0.49pp acc, -5.5 AUC). Hangover hypothesis didn't hold; the
    # parquet is kept for the web visualization but excluded from the model.
    burn = None
    # game_flow features (fi_score, lead_hold_6, runs_std) tested but degraded
    # walk-forward AUC and edge calibration — high collinearity with existing
    # bullpen/win_pct features. Kept on disk; set to None to disable.
    game_flow = None
    # situational features (1-run games, blowouts, timezone travel) tested but
    # degraded walk-forward — collinear with win_pct/run_diff/xwoba already in model.
    situational = None
    # error features (error_rate, resilience) tested but degraded walk-forward —
    # likely captured by def_xwoba + win_pct. Kept on disk; disabled here.
    errors = None

    g = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    g = g.dropna(subset=["home_team_abbrev", "away_team_abbrev", "game_date"])

    # --- team features (home + away) ---
    home_t = team_feat[team_feat["is_home"] == 1].drop(columns=["is_home", "game_date"])
    away_t = team_feat[team_feat["is_home"] == 0].drop(columns=["is_home", "game_date"])

    tcols = [c for c in home_t.columns if c not in ("game_pk", "team")]
    home_t = home_t.rename(columns={c: f"{c}_h" for c in tcols}).rename(columns={"team": "home_team_abbrev"})
    away_t = away_t.rename(columns={c: f"{c}_a" for c in tcols}).rename(columns={"team": "away_team_abbrev"})

    g = g.merge(home_t, on=["game_pk", "home_team_abbrev"], how="left")
    g = g.merge(away_t, on=["game_pk", "away_team_abbrev"], how="left")

    # --- pitcher features — match probable starter per side ---
    p_h = pit_feat.rename(columns={"pitcher": "probable_home_pitcher_id"})
    p_a = pit_feat.rename(columns={"pitcher": "probable_away_pitcher_id"})
    pcols = [c for c in pit_feat.columns if c not in ("game_pk", "pitcher", "game_date")]
    p_h = p_h[["game_pk", "probable_home_pitcher_id"] + pcols].rename(
        columns={c: f"{c}_h" for c in pcols}
    )
    p_a = p_a[["game_pk", "probable_away_pitcher_id"] + pcols].rename(
        columns={c: f"{c}_a" for c in pcols}
    )
    g = g.merge(p_h, on=["game_pk", "probable_home_pitcher_id"], how="left")
    g = g.merge(p_a, on=["game_pk", "probable_away_pitcher_id"], how="left")

    # --- park factor (prior-season for this venue) ---
    g = g.merge(park, on=["season", "venue_id"], how="left")

    # --- market features (devigged consensus probability + totals + spread) ---
    market_cols = [c for c in market.columns if c != "espn_event_id"]
    g = g.merge(market[market_cols], on="game_pk", how="left")

    # --- ELO ratings ---
    if elo is not None:
        g = g.merge(elo, on="game_pk", how="left")

    # --- Cluster luck (BABIP / LOB%) ---
    if cluster_luck is not None:
        ccols = [c for c in cluster_luck.columns if c not in ("game_pk", "team")]
        home_C = cluster_luck.rename(columns={"team": "home_team_abbrev"})
        away_C = cluster_luck.rename(columns={"team": "away_team_abbrev"})
        home_C = home_C.rename(columns={c: f"{c}_h" for c in ccols})
        away_C = away_C.rename(columns={c: f"{c}_a" for c in ccols})
        g = g.merge(home_C, on=["game_pk", "home_team_abbrev"], how="left")
        g = g.merge(away_C, on=["game_pk", "away_team_abbrev"], how="left")
        for c in ccols:
            h = f"{c}_h"; a = f"{c}_a"
            if h in g.columns and a in g.columns:
                g[f"{c}_diff"] = g[h] - g[a]

    # --- Pythagorean expectation (luck regression signal) ---
    if pythag is not None:
        pcols = [c for c in pythag.columns if c not in ("game_pk", "team")]
        home_P = pythag.rename(columns={"team": "home_team_abbrev"})
        away_P = pythag.rename(columns={"team": "away_team_abbrev"})
        home_P = home_P.rename(columns={c: f"{c}_h" for c in pcols})
        away_P = away_P.rename(columns={c: f"{c}_a" for c in pcols})
        g = g.merge(home_P, on=["game_pk", "home_team_abbrev"], how="left")
        g = g.merge(away_P, on=["game_pk", "away_team_abbrev"], how="left")
        for c in pcols:
            h = f"{c}_h"; a = f"{c}_a"
            if h in g.columns and a in g.columns:
                g[f"{c}_diff"] = g[h] - g[a]

    # --- tanking / late-season features (home + away) ---
    if tanking is not None:
        tcols = [c for c in tanking.columns if c not in ("game_pk", "team")]
        home_T = tanking.rename(columns={"team": "home_team_abbrev"})
        away_T = tanking.rename(columns={"team": "away_team_abbrev"})
        home_T = home_T.rename(columns={c: f"{c}_h" for c in tcols})
        away_T = away_T.rename(columns={c: f"{c}_a" for c in tcols})
        g = g.merge(home_T, on=["game_pk", "home_team_abbrev"], how="left")
        g = g.merge(away_T, on=["game_pk", "away_team_abbrev"], how="left")
        for c in tcols:
            h = f"{c}_h"; a = f"{c}_a"
            if h in g.columns and a in g.columns:
                g[f"{c}_diff"] = g[h] - g[a]

    # --- series context ---
    if series_ctx is not None:
        g = g.merge(series_ctx, on="game_pk", how="left")

    # --- umpire features (HP ump prior-season aggregates) ---
    if ump is not None:
        g = g.merge(ump, on="game_pk", how="left")

    # --- bullpen features (home + away) ---
    if bullpen is not None:
        bp_cols = [c for c in bullpen.columns if c not in ("game_pk", "side")]
        home_bp = bullpen[bullpen["side"] == "home"].drop(columns=["side"])
        away_bp = bullpen[bullpen["side"] == "away"].drop(columns=["side"])
        home_bp = home_bp.rename(columns={c: f"{c}_h" for c in bp_cols})
        away_bp = away_bp.rename(columns={c: f"{c}_a" for c in bp_cols})
        g = g.merge(home_bp, on="game_pk", how="left")
        g = g.merge(away_bp, on="game_pk", how="left")
        for c in bp_cols:
            h = f"{c}_h"; a = f"{c}_a"
            if h in g.columns and a in g.columns:
                g[f"{c}_diff"] = g[h] - g[a]

    # --- comeback features (rolling comeback rate/depth per team) ---
    if comeback is not None:
        cb_cols = [c for c in comeback.columns if c not in ("game_pk", "side")]
        home_cb = comeback[comeback["side"] == "home"].drop(columns=["side"])
        away_cb = comeback[comeback["side"] == "away"].drop(columns=["side"])
        home_cb = home_cb.rename(columns={c: f"{c}_h" for c in cb_cols})
        away_cb = away_cb.rename(columns={c: f"{c}_a" for c in cb_cols})
        g = g.merge(home_cb, on="game_pk", how="left")
        g = g.merge(away_cb, on="game_pk", how="left")
        for c in cb_cols:
            h = f"{c}_h"; a = f"{c}_a"
            if h in g.columns and a in g.columns:
                g[f"{c}_diff"] = g[h] - g[a]

    # --- lineup_recent features (top-4 batters L15 form) ---
    if lineup_recent is not None:
        lr_cols = [c for c in lineup_recent.columns if c not in ("game_pk", "side")]
        home_lr = lineup_recent[lineup_recent["side"] == "home"].drop(columns=["side"])
        away_lr = lineup_recent[lineup_recent["side"] == "away"].drop(columns=["side"])
        home_lr = home_lr.rename(columns={c: f"{c}_h" for c in lr_cols})
        away_lr = away_lr.rename(columns={c: f"{c}_a" for c in lr_cols})
        g = g.merge(home_lr, on="game_pk", how="left")
        g = g.merge(away_lr, on="game_pk", how="left")
        for c in lr_cols:
            h = f"{c}_h"; a = f"{c}_a"
            if h in g.columns and a in g.columns:
                g[f"{c}_diff"] = g[h] - g[a]

    # --- burn features (yesterday's "carne al asador" — bullpen burn, extras) ---
    if burn is not None:
        bn_cols = [c for c in burn.columns if c not in ("game_pk", "side")]
        home_bn = burn[burn["side"] == "home"].drop(columns=["side"])
        away_bn = burn[burn["side"] == "away"].drop(columns=["side"])
        home_bn = home_bn.rename(columns={c: f"{c}_h" for c in bn_cols})
        away_bn = away_bn.rename(columns={c: f"{c}_a" for c in bn_cols})
        g = g.merge(home_bn, on="game_pk", how="left")
        g = g.merge(away_bn, on="game_pk", how="left")
        for c in bn_cols:
            h = f"{c}_h"; a = f"{c}_a"
            if h in g.columns and a in g.columns:
                g[f"{c}_diff"] = g[h] - g[a]

    # --- arsenal matchup features (lineup vs opposing starter's specific arsenal) ---
    if arsenal is not None:
        L = arsenal.copy()
        lcols = [c for c in L.columns if c not in ("game_pk", "side")]
        home_A = L[L["side"] == "home"].drop(columns=["side"])
        away_A = L[L["side"] == "away"].drop(columns=["side"])
        home_A = home_A.rename(columns={c: f"{c}_h" for c in lcols})
        away_A = away_A.rename(columns={c: f"{c}_a" for c in lcols})
        g = g.merge(home_A, on="game_pk", how="left")
        g = g.merge(away_A, on="game_pk", how="left")
        for c in lcols:
            h = f"{c}_h"; a = f"{c}_a"
            if h in g.columns and a in g.columns:
                g[f"{c}_diff"] = g[h] - g[a]

    # --- lineup features (home + away lineup vs opposing starter's hand) ---
    if lineup is not None:
        L = lineup.copy()
        lcols = [c for c in L.columns if c not in ("game_pk", "side")]
        home_L = L[L["side"] == "home"].drop(columns=["side"])
        away_L = L[L["side"] == "away"].drop(columns=["side"])
        home_L = home_L.rename(columns={c: f"{c}_h" for c in lcols})
        away_L = away_L.rename(columns={c: f"{c}_a" for c in lcols})
        g = g.merge(home_L, on="game_pk", how="left")
        g = g.merge(away_L, on="game_pk", how="left")
        # Diffs
        for c in lcols:
            h = f"{c}_h"; a = f"{c}_a"
            if h in g.columns and a in g.columns:
                g[f"{c}_diff"] = g[h] - g[a]
    # logit transform — gradient boosters do better with logit than raw prob
    p = g["market_p_home"].clip(0.01, 0.99)
    g["market_logit_p_home"] = np.log(p / (1 - p))

    # --- park × offense/pitching interactions ---
    # Hitter-friendly parks magnify good offenses and hurt pitchers more.
    # Multiplicative features help LightGBM see the interaction explicitly.
    from .park_offense import INTERACTION_PAIRS
    for base_a, base_b, name in INTERACTION_PAIRS:
        if base_a not in g.columns:
            continue
        for side in ("h", "a"):
            colb = f"{base_b}_{side}"
            if colb in g.columns:
                g[f"{name}_{side}"] = g[base_a] * g[colb]
        # diff
        h, a = f"{name}_h", f"{name}_a"
        if h in g.columns and a in g.columns:
            g[f"{name}_diff"] = g[h] - g[a]

    # --- targets ---
    g["home_win"] = (g["home_score"] > g["away_score"]).astype("Int64")
    g["total_runs"] = g["home_score"] + g["away_score"]
    # F5 (first-5-innings) targets — separate betting market with lower variance
    if f5_targets is not None:
        g = g.merge(f5_targets, on="game_pk", how="left")

    # --- engineered diffs ---
    diff_pairs = [
        "runs_scored_l10", "runs_allowed_l10", "run_diff_l10", "win_pct_l30",
        "off_xwoba_l30", "off_barrel_rate_l30", "off_k_pct_l30", "off_bb_pct_l30",
        "def_xwoba_l30", "def_barrel_rate_l30", "def_k_pct_l30", "def_bb_pct_l30",
        "starter_xwoba_l15", "starter_barrel_against_l15",
        "starter_k_pct_l15", "starter_bb_pct_l15",
        "starter_velo_mean_l15", "starter_spin_mean_l15",
        "starter_pa_l15", "starter_days_rest", "days_since_last_game",
    ]
    for c in diff_pairs:
        h = f"{c}_h"; a = f"{c}_a"
        if h in g.columns and a in g.columns:
            g[f"{c}_diff"] = g[h] - g[a]

    keep_base = [
        "game_pk", "game_date", "season", "game_type", "status",
        "home_team_abbrev", "away_team_abbrev",
        "home_team_id", "away_team_id",
        "venue_id", "venue_name", "roof_type",
        "probable_home_pitcher_id", "probable_away_pitcher_id",
        "home_score", "away_score",
        "weather_temp_f",
        "park_runs_factor",
        "market_p_home", "market_logit_p_home", "market_over_under",
        "market_spread", "market_n_providers", "market_p_home_std",
        "home_win", "total_runs",
        "f5_home_score", "f5_away_score", "f5_home_won", "f5_tied",
        "f5_total_runs", "f5_run_diff",
    ]
    feature_cols = [c for c in g.columns
                    if c.endswith(("_h", "_a", "_diff"))]
    # Add features explicitly that don't follow the _h/_a/_diff convention.
    explicit_features = [
        "market_logit_p_home", "market_over_under", "market_spread",
        "market_p_home_std", "market_n_providers",
        "home_elo_pre", "away_elo_pre", "elo_diff_pre", "expected_home_win_elo",
        "series_game_number", "series_run_diff_so_far",
        "did_home_win_previous_game_in_series",
        "ump_acc", "ump_acc_above_x", "ump_consistency",
        "ump_run_impact", "ump_games",
    ]
    feature_cols = feature_cols + [c for c in explicit_features if c in g.columns]
    seen: set[str] = set()
    keep: list[str] = []
    for c in keep_base + feature_cols:
        if c in g.columns and c not in seen:
            seen.add(c)
            keep.append(c)
    out_df = g[keep].copy()

    # Drop low-signal noise features (|corr with home_win| < 0.01 across all seasons).
    # Walk-forward showed pruning these 20 features improves ACC +3.8pp and ROI +3.2pp.
    _DROP = {
        "days_since_last_game_h", "days_since_last_game_a",
        "def_barrel_rate_l30_h", "off_barrel_rate_l30_a",
        "starter_days_rest_h", "starter_days_rest_diff",
        "starter_spin_mean_l15_a", "starter_spin_mean_l15_diff",
        "bullpen_ip_l1d_diff", "bullpen_ip_l3d_diff",
        "cb_max_def_l30_h", "cb_win_rate_l30_a",
        "lineup_n_with_data_h", "lineup_n_with_data_a", "lineup_n_with_data_diff",
        "park_off_xwoba_h", "park_def_xwoba_a",
        "series_game_number",
        "ump_consistency", "ump_games",
    }
    drop_present = [c for c in _DROP if c in out_df.columns]
    if drop_present:
        out_df = out_df.drop(columns=drop_present)
        print(f"pruned {len(drop_present)} low-signal features")

    out = PROCESSED / "train.parquet"
    out_df.to_parquet(out, index=False)
    actual_feat_cols = [c for c in out_df.columns if c not in keep_base]
    print(f"wrote {out} ({len(out_df):,} rows, {len(actual_feat_cols)} features)")
    return out


if __name__ == "__main__":
    build()
