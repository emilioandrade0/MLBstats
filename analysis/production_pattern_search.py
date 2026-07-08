import itertools
import json

import numpy as np
import pandas as pd


WF_PATH = "data/processed/walkforward_preds.parquet"
TRAIN_PATH = "data/processed/train.parquet"


TRAP_PICK_TEAMS = {"LAA"}
TRAP_FADE_TEAMS = {"MIL"}
DAY_TRAP_FADE = {("WSH", 5)}


def accuracy(y, pick_home):
    return float((np.asarray(y, dtype=int) == np.asarray(pick_home, dtype=int)).mean())


def auc_rank(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and p[order[j]] == p[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def logloss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def winner_threshold(mp, ph, ht, at, season):
    if not np.isfinite(mp):
        return 0.5
    modern = pd.notna(season) and int(season) >= 2026
    if mp <= 0.42:
        thr = 0.500 if modern else 0.440
    elif mp <= 0.50:
        thr = 0.500 if modern else 0.485
    elif mp <= 0.53:
        thr = 0.500 if modern else 0.4975
    elif mp <= 0.56:
        thr = 0.500 if modern else 0.485
    elif mp <= 0.60:
        thr = 0.500 if modern else 0.515
    else:
        thr = 0.440

    if np.isfinite(ph) and 0.42 <= mp <= 0.60:
        if 0.46 <= ph < 0.47:
            thr -= 0.025
        elif 0.46 <= (1 - ph) < 0.47:
            thr += 0.025
        elif 0.53 <= ph < 0.54:
            thr += 0.025
        if ht == "AZ" and 0.55 <= ph < 0.60:
            thr += 0.015
        elif at == "AZ" and 0.55 <= (1 - ph) < 0.60:
            thr -= 0.015
        if ht == "NYY" and 0.53 <= ph < 0.55:
            thr += 0.015
        elif at == "NYY" and 0.53 <= (1 - ph) < 0.55:
            thr -= 0.015
        if ht == "HOU" and 0.55 <= ph < 0.60:
            thr += 0.015
        elif at == "HOU" and 0.55 <= (1 - ph) < 0.60:
            thr -= 0.015
        if ht == "MIN" and 0.53 <= ph < 0.55:
            thr += 0.010
        elif at == "MIN" and 0.53 <= (1 - ph) < 0.55:
            thr -= 0.010
    return thr


def production_pick(df):
    out = []
    for r in df.itertuples(index=False):
        raw = bool(
            np.isfinite(r.p_home)
            and r.p_home >= winner_threshold(
                r.market_p_home,
                r.p_home,
                r.home_team_abbrev,
                r.away_team_abbrev,
                r.season,
            )
        )
        picked = r.home_team_abbrev if raw else r.away_team_abbrev
        opp = r.away_team_abbrev if raw else r.home_team_abbrev
        dow = pd.Timestamp(r.game_date).dayofweek
        if picked in TRAP_PICK_TEAMS or opp in TRAP_FADE_TEAMS or (opp, dow) in DAY_TRAP_FADE:
            raw = not raw
        out.append(raw)
    return np.asarray(out, dtype=bool)


def probability_for_pick(df, pick_home):
    p = df.p_home.to_numpy(dtype=float).copy()
    return np.where(pick_home, np.maximum(p, 1 - p), np.minimum(p, 1 - p))


def metric_table(df, pick_home, p_score):
    rows = []
    for season, idx in df.groupby("season").groups.items():
        g = df.loc[idx]
        ids = g.index.to_numpy()
        rows.append(
            {
                "season": str(int(season)),
                "n": len(g),
                "base_acc": accuracy(g.home_win, g.base_pick_home),
                "new_acc": accuracy(g.home_win, pick_home[ids]),
                "lift": accuracy(g.home_win, pick_home[ids]) - accuracy(g.home_win, g.base_pick_home),
                "auc": auc_rank(g.home_win, p_score[ids]),
                "log_loss": logloss(g.home_win, p_score[ids]),
            }
        )
    rows.append(
        {
            "season": "ALL",
            "n": len(df),
            "base_acc": accuracy(df.home_win, df.base_pick_home),
            "new_acc": accuracy(df.home_win, pick_home[df.index.to_numpy()]),
            "lift": accuracy(df.home_win, pick_home[df.index.to_numpy()]) - accuracy(df.home_win, df.base_pick_home),
            "auc": auc_rank(df.home_win, p_score[df.index.to_numpy()]),
            "log_loss": logloss(df.home_win, p_score[df.index.to_numpy()]),
        }
    )
    return pd.DataFrame(rows)


def load_data():
    wanted = [
        "game_pk",
        "market_spread",
        "market_n_providers",
        "market_p_home_std",
        "market_line_shift_home_pp",
        "market_line_shift_abs_pp",
        "market_total_shift",
        "lineup_xwoba_vs_sp_hand_diff",
        "lineup_k_pct_vs_sp_hand_diff",
        "lineup_bb_pct_vs_sp_hand_diff",
        "lineup_barrel_vs_sp_hand_diff",
        "lineup_top4_xwoba_vs_sp_hand_diff",
        "park_runs_factor",
        "park_off_xwoba_diff",
        "park_starter_xwoba_diff",
        "park_run_diff_diff",
        "series_run_diff_so_far",
        "did_home_win_previous_game_in_series",
        "ump_acc",
        "ump_acc_above_x",
        "ump_run_impact",
        "bullpen_ip_l5d_diff",
        "bullpen_apps_l3d_diff",
        "bullpen_runs_l5_diff",
        "bullpen_ip_min_l3_diff",
        "starter_days_rest_a",
        "days_since_last_game_diff",
        "weather_temp_f",
        "expected_home_win_elo",
        "elo_diff_pre",
        "pyth_wpct_l30_diff",
        "off_xwoba_l30_diff",
        "starter_xwoba_l15_diff",
        "lineup_hot_bats_count_l15_diff",
        "lineup_cold_bats_count_l15_diff",
    ]
    wf = pd.read_parquet(WF_PATH)
    schema = pd.read_parquet(TRAIN_PATH).columns.tolist()
    train = pd.read_parquet(TRAIN_PATH, columns=[c for c in wanted if c in schema])
    wf["game_date"] = pd.to_datetime(wf.game_date)
    wf = wf.dropna(subset=["home_win", "p_home"]).copy()
    wf["home_win"] = wf.home_win.astype(int)
    df = wf.merge(train, on="game_pk", how="left").sort_values("game_date").reset_index(drop=True)
    df["base_pick_home"] = production_pick(df)
    df["raw_pick_home"] = df.p_home_model_raw.fillna(df.p_home).to_numpy(dtype=float) >= 0.5
    df["blend_pick_home"] = df.p_home.to_numpy(dtype=float) >= 0.5
    df["market_pick_home"] = df.market_p_home.fillna(df.p_home).to_numpy(dtype=float) >= 0.5
    df["dow"] = df.game_date.dt.dayofweek
    df["month"] = df.game_date.dt.month
    df["picked_team"] = np.where(df.base_pick_home, df.home_team_abbrev, df.away_team_abbrev)
    df["opp_team"] = np.where(df.base_pick_home, df.away_team_abbrev, df.home_team_abbrev)
    df["base_correct"] = df.base_pick_home.astype(int) == df.home_win.astype(int)
    return df


def quantile_specs(df, cols):
    specs = []
    for col in cols:
        if col not in df.columns:
            continue
        x = df[col]
        if x.notna().sum() < 250 or x.nunique(dropna=True) < 4:
            continue
        for v in sorted(set(float(q) for q in x.quantile([0.2, 0.35, 0.5, 0.65, 0.8]).dropna())):
            specs.append((f"{col}<={v:.4g}", lambda z, col=col, v=v: (z[col] <= v).fillna(False).to_numpy(), col))
            specs.append((f"{col}>={v:.4g}", lambda z, col=col, v=v: (z[col] >= v).fillna(False).to_numpy(), col))
    return specs


def context_specs(df, include_teams=False):
    specs = [
        ("base_picks_home", lambda z: z.base_pick_home.to_numpy(), "side"),
        ("base_picks_away", lambda z: (~z.base_pick_home).to_numpy(), "side"),
        ("model_market_disagree", lambda z: (z.blend_pick_home != z.market_pick_home).to_numpy(), "agree"),
        ("model_market_agree", lambda z: (z.blend_pick_home == z.market_pick_home).to_numpy(), "agree"),
        ("raw_differs_from_prod", lambda z: (z.raw_pick_home != z.base_pick_home).to_numpy(), "source"),
        ("blend_differs_from_prod", lambda z: (z.blend_pick_home != z.base_pick_home).to_numpy(), "source"),
        ("market_differs_from_prod", lambda z: (z.market_pick_home != z.base_pick_home).to_numpy(), "source"),
        ("low_conf_35bp", lambda z: ((z.p_home - 0.5).abs() <= 0.035).to_numpy(), "conf"),
        ("high_conf_80bp", lambda z: ((z.p_home - 0.5).abs() >= 0.08).to_numpy(), "conf"),
        ("day_game", lambda z: z.day_night.astype(str).str.lower().eq("day").to_numpy(), "daynight"),
        ("night_game", lambda z: z.day_night.astype(str).str.lower().eq("night").to_numpy(), "daynight"),
    ]
    for m in sorted(df.month.dropna().unique()):
        specs.append((f"month={int(m)}", lambda z, m=m: (z.month == m).to_numpy(), "month"))
    if "did_home_win_previous_game_in_series" in df.columns:
        specs += [
            ("home_won_prev_series_game", lambda z: (z.did_home_win_previous_game_in_series == 1).fillna(False).to_numpy(), "series_prev"),
            ("home_lost_prev_series_game", lambda z: (z.did_home_win_previous_game_in_series == 0).fillna(False).to_numpy(), "series_prev"),
        ]
    if include_teams:
        for t in sorted(set(df.home_team_abbrev.dropna()) | set(df.away_team_abbrev.dropna())):
            specs.append((f"picked_team={t}", lambda z, t=t: (z.picked_team == t).to_numpy(), "picked_team"))
            specs.append((f"opp_team={t}", lambda z, t=t: (z.opp_team == t).to_numpy(), "opp_team"))
    return specs


def build_masks(df, cols, include_teams=False):
    primary = quantile_specs(df, cols)
    ctx = context_specs(df, include_teams=include_teams)
    masks = []
    masks.extend(primary)
    masks.extend(ctx)
    for a in primary:
        for b in ctx:
            masks.append((f"{a[0]} & {b[0]}", lambda z, a=a, b=b: a[1](z) & b[1](z), f"{a[2]}+{b[2]}"))
    return masks


def best_rule(df, cols, include_teams=False, min_n=85):
    masks = build_masks(df, cols, include_teams=include_teams)
    sources = {
        "flip": ~df.base_pick_home.to_numpy(dtype=bool),
        "raw": df.raw_pick_home.to_numpy(dtype=bool),
        "blend": df.blend_pick_home.to_numpy(dtype=bool),
        "market": df.market_pick_home.to_numpy(dtype=bool),
    }
    y = df.home_win.to_numpy(dtype=int)
    base = df.base_pick_home.to_numpy(dtype=bool)
    base_correct = base.astype(int) == y
    rows = []
    for name, fn, _family in masks:
        mask = fn(df)
        n = int(mask.sum())
        if n < min_n or n > len(df) * 0.55:
            continue
        base_c = int(base_correct[mask].sum())
        sub_base_acc = base_c / n
        for action, candidate_pick in sources.items():
            if action == "flip" and sub_base_acc >= 0.495:
                continue
            cand_c = int((candidate_pick.astype(int) == y)[mask].sum())
            gain = (cand_c - base_c) / len(df)
            if gain > 0:
                rows.append((gain, name, action, n, sub_base_acc))
    if not rows:
        return None, len(masks) * len(sources), 0
    gain, name, action, n, sub_base_acc = sorted(rows, reverse=True, key=lambda r: r[0])[0]
    return (
        {
            "name": name,
            "action": action,
            "n": n,
            "sub_base_acc": sub_base_acc,
            "design_lift": gain,
            "cols": cols,
            "include_teams": include_teams,
        },
        len(masks) * len(sources),
        len(rows),
    )


def apply_rule(df, rule, threshold_source):
    fn = None
    for name, candidate_fn, _family in build_masks(threshold_source, rule["cols"], rule["include_teams"]):
        if name == rule["name"]:
            fn = candidate_fn
            break
    pick = df.base_pick_home.to_numpy(dtype=bool).copy()
    p_score = probability_for_pick(df, pick)
    if fn is None:
        return pick, p_score, np.zeros(len(df), dtype=bool)
    mask = fn(df)
    if rule["action"] == "flip":
        pick[mask] = ~pick[mask]
        p_score[mask] = 1 - p_score[mask]
    elif rule["action"] == "raw":
        pick[mask] = df.raw_pick_home.to_numpy(dtype=bool)[mask]
        p_score[mask] = df.p_home_model_raw.fillna(df.p_home).to_numpy(dtype=float)[mask]
    elif rule["action"] == "blend":
        pick[mask] = df.blend_pick_home.to_numpy(dtype=bool)[mask]
        p_score[mask] = df.p_home.to_numpy(dtype=float)[mask]
    elif rule["action"] == "market":
        pick[mask] = df.market_pick_home.to_numpy(dtype=bool)[mask]
        p_score[mask] = df.market_p_home.fillna(df.p_home).to_numpy(dtype=float)[mask]
    return pick, p_score, mask


def temporal_kfold(df, cols, include_teams=False, k=6):
    d = df.sort_values("game_date").reset_index(drop=True)
    folds = np.array_split(np.arange(len(d)), k)
    lifts = []
    chosen = []
    mask_n = []
    for fold in folds:
        is_test = np.zeros(len(d), dtype=bool)
        is_test[fold] = True
        train = d.loc[~is_test].copy().reset_index(drop=True)
        test = d.loc[is_test].copy().reset_index(drop=True)
        rule, _tested, _hits = best_rule(train, cols, include_teams=include_teams)
        if rule is None:
            lifts.append(0.0)
            chosen.append("NONE")
            mask_n.append(0)
            continue
        pick, _p_score, mask = apply_rule(test, rule, train)
        lifts.append(accuracy(test.home_win, pick) - accuracy(test.home_win, test.base_pick_home))
        chosen.append(f"{rule['name']} | {rule['action']}")
        mask_n.append(int(mask.sum()))
    return {
        "wins": int(sum(x > 0 for x in lifts)),
        "folds": k,
        "win_rate": float(np.mean([x > 0 for x in lifts])),
        "mean_lift": float(np.mean(lifts)),
        "lifts": lifts,
        "chosen": chosen,
        "mask_n": mask_n,
    }


def main():
    df = load_data()
    df = df.reset_index(drop=True)
    cut = df.game_date.quantile(0.60)
    design = df[df.game_date <= cut].copy().reset_index(drop=True)
    holdout = df[df.game_date > cut].copy().reset_index(drop=True)

    base_p = probability_for_pick(df, df.base_pick_home.to_numpy(dtype=bool))
    print("PRODUCTION BASELINE")
    print(metric_table(df, df.base_pick_home.to_numpy(dtype=bool), base_p).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("SPLIT", design.game_date.min().date(), design.game_date.max().date(), len(design), "|", holdout.game_date.min().date(), holdout.game_date.max().date(), len(holdout))

    angles = [
        ("Source stability: raw/blend/market only where they disagree", [], False),
        ("Market liquidity and late steam", ["market_p_home_std", "market_n_providers", "market_spread", "market_line_shift_home_pp", "market_line_shift_abs_pp", "market_total_shift"], False),
        ("Lineup hand split plus park", ["lineup_xwoba_vs_sp_hand_diff", "lineup_k_pct_vs_sp_hand_diff", "lineup_bb_pct_vs_sp_hand_diff", "lineup_barrel_vs_sp_hand_diff", "lineup_top4_xwoba_vs_sp_hand_diff", "park_runs_factor", "park_off_xwoba_diff", "park_starter_xwoba_diff", "park_run_diff_diff"], False),
        ("Series state and schedule pocket", ["series_run_diff_so_far", "did_home_win_previous_game_in_series", "days_since_last_game_diff"], False),
        ("Umpire and run environment", ["ump_acc", "ump_acc_above_x", "ump_run_impact", "market_over_under", "pred_total", "weather_temp_f"], False),
        ("Bullpen/rest stress", ["bullpen_ip_l5d_diff", "bullpen_apps_l3d_diff", "bullpen_runs_l5_diff", "bullpen_ip_min_l3_diff", "starter_days_rest_a", "days_since_last_game_diff"], False),
        ("Natural team/opponent residual pockets", ["market_p_home_std", "market_line_shift_abs_pp", "expected_home_win_elo", "elo_diff_pre"], True),
        ("Team quality deltas: ELO/pyth/offense/starter", ["expected_home_win_elo", "elo_diff_pre", "pyth_wpct_l30_diff", "off_xwoba_l30_diff", "starter_xwoba_l15_diff", "lineup_hot_bats_count_l15_diff", "lineup_cold_bats_count_l15_diff"], False),
    ]

    summary = []
    for angle, cols, include_teams in angles:
        cols = [c for c in cols if c in df.columns]
        rule, tested, hits = best_rule(design, cols, include_teams=include_teams)
        print(f"\nANGLE {angle}")
        print("tested", tested, "eligible_design_hits", hits)
        if rule is None:
            print("NO DESIGN SIGNAL")
            summary.append({"angle": angle, "tested": tested, "status": "rejected_no_design"})
            continue
        hold_pick, hold_p, _hold_mask = apply_rule(holdout, rule, design)
        all_pick, all_p, _all_mask = apply_rule(df, rule, design)
        hold_table = metric_table(holdout, hold_pick, hold_p)
        all_table = metric_table(df, all_pick, all_p)
        kfold = temporal_kfold(df, cols, include_teams=include_teams)
        hold_lift = float(hold_table[hold_table.season == "ALL"].lift.iloc[0])
        all_lift = float(all_table[all_table.season == "ALL"].lift.iloc[0])
        nonnegative_all_seasons = int((all_table[all_table.season != "ALL"].lift >= 0).sum())
        status = "survived" if hold_lift > 0 and nonnegative_all_seasons >= 3 and kfold["win_rate"] >= 0.70 else "rejected"

        print("best", {k: rule[k] for k in ["name", "action", "n", "sub_base_acc", "design_lift"]})
        print("HOLDOUT")
        print(hold_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("ALL")
        print(all_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("KFOLD", json.dumps(kfold))
        summary.append(
            {
                "angle": angle,
                "tested": tested,
                "eligible_design_hits": hits,
                "best": rule["name"],
                "action": rule["action"],
                "design_lift": float(rule["design_lift"]),
                "holdout_lift": hold_lift,
                "all_lift": all_lift,
                "all_seasons_nonnegative": nonnegative_all_seasons,
                "kfold_win_rate": kfold["win_rate"],
                "kfold_wins": kfold["wins"],
                "kfold_mean_lift": kfold["mean_lift"],
                "status": status,
            }
        )

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
