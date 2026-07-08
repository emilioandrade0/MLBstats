import itertools
import json

import numpy as np
import pandas as pd


WF_PATH = "data/processed/walkforward_preds.parquet"
TRAIN_PATH = "data/processed/train.parquet"


def accuracy(y, p):
    return float(((np.asarray(p) > 0.5).astype(int) == np.asarray(y, dtype=int)).mean())


def logloss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


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


def metric_table(df, p):
    rows = []
    for season, idx in df.groupby("season").groups.items():
        g = df.loc[idx]
        pp = p[g.index.to_numpy()]
        rows.append(
            {
                "season": str(int(season)),
                "n": len(g),
                "base_acc": accuracy(g.home_win, g.p_home),
                "new_acc": accuracy(g.home_win, pp),
                "lift": accuracy(g.home_win, pp) - accuracy(g.home_win, g.p_home),
                "auc": auc_rank(g.home_win, pp),
                "log_loss": logloss(g.home_win, pp),
            }
        )
    rows.append(
        {
            "season": "ALL",
            "n": len(df),
            "base_acc": accuracy(df.home_win, df.p_home),
            "new_acc": accuracy(df.home_win, p[df.index.to_numpy()]),
            "lift": accuracy(df.home_win, p[df.index.to_numpy()]) - accuracy(df.home_win, df.p_home),
            "auc": auc_rank(df.home_win, p[df.index.to_numpy()]),
            "log_loss": logloss(df.home_win, p[df.index.to_numpy()]),
        }
    )
    return pd.DataFrame(rows)


def build_specs(df, cols):
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

    context = [
        ("model_fav_home", lambda z: (z.p_home > 0.5).to_numpy(), "side"),
        ("model_fav_away", lambda z: (z.p_home <= 0.5).to_numpy(), "side"),
        ("market_fav_home", lambda z: (z.market_p_home > 0.5).fillna(False).to_numpy(), "market_side"),
        ("market_fav_away", lambda z: (z.market_p_home <= 0.5).fillna(False).to_numpy(), "market_side"),
        ("model_market_disagree", lambda z: ((z.p_home > 0.5) != (z.market_p_home > 0.5)).fillna(False).to_numpy(), "agree"),
        ("low_conf_35bp", lambda z: ((z.p_home - 0.5).abs() <= 0.035).to_numpy(), "conf"),
        ("high_conf_80bp", lambda z: ((z.p_home - 0.5).abs() >= 0.08).to_numpy(), "conf"),
        ("day_game", lambda z: z.day_night.astype(str).str.lower().eq("day").to_numpy(), "day"),
        ("night_game", lambda z: z.day_night.astype(str).str.lower().eq("night").to_numpy(), "day"),
    ]
    if "did_home_win_previous_game_in_series" in df.columns:
        context += [
            ("home_won_prev_series_game", lambda z: (z.did_home_win_previous_game_in_series == 1).fillna(False).to_numpy(), "series_prev"),
            ("home_lost_prev_series_game", lambda z: (z.did_home_win_previous_game_in_series == 0).fillna(False).to_numpy(), "series_prev"),
        ]

    masks = list(specs)
    for primary in specs:
        for ctx in context:
            masks.append(
                (
                    f"{primary[0]} & {ctx[0]}",
                    lambda z, primary=primary, ctx=ctx: primary[1](z) & ctx[1](z),
                    f"{primary[2]}+{ctx[2]}",
                )
            )
    return masks


def best_rule(df, cols, min_n=90):
    y = df.home_win.to_numpy(dtype=int)
    p = df.p_home.to_numpy(dtype=float)
    base_correct = ((p > 0.5) == (y == 1))
    market_p = df.market_p_home.fillna(df.p_home).to_numpy(dtype=float)
    raw_p = df.p_home_model_raw.fillna(df.p_home).to_numpy(dtype=float)
    rows = []
    specs = build_specs(df, cols)

    for name, fn, _family in specs:
        mask = fn(df)
        n = int(mask.sum())
        if n < min_n or n > len(df) * 0.55:
            continue
        base_c = int(base_correct[mask].sum())
        sub_base = base_c / n

        if sub_base < 0.495:
            rows.append(((n - 2 * base_c) / len(df), name, "flip", n, sub_base))

        for action, alt_p in [("market", market_p), ("raw", raw_p)]:
            alt_correct = ((alt_p > 0.5) == (y == 1))
            gain = (int(alt_correct[mask].sum()) - base_c) / len(df)
            if gain > 0:
                rows.append((gain, name, action, n, sub_base))

    if not rows:
        return None, len(specs) * 3, 0

    gain, name, action, n, sub_base = sorted(rows, reverse=True, key=lambda r: r[0])[0]
    return (
        {
            "name": name,
            "action": action,
            "n": n,
            "sub_base_acc": sub_base,
            "design_lift": gain,
            "tested": len(specs) * 3,
            "eligible_hits": len(rows),
            "cols": cols,
        },
        len(specs) * 3,
        len(rows),
    )


def apply_named(df, rule, threshold_source):
    fn = None
    for name, candidate_fn, _family in build_specs(threshold_source, rule["cols"]):
        if name == rule["name"]:
            fn = candidate_fn
            break
    p = df.p_home.to_numpy(dtype=float).copy()
    if fn is None:
        return p, np.zeros(len(df), dtype=bool)
    mask = fn(df)
    if rule["action"] == "flip":
        p[mask] = 1 - p[mask]
    elif rule["action"] == "market":
        p[mask] = df.market_p_home.fillna(df.p_home).to_numpy(dtype=float)[mask]
    elif rule["action"] == "raw":
        p[mask] = df.p_home_model_raw.fillna(df.p_home).to_numpy(dtype=float)[mask]
    return p, mask


def temporal_kfold(df, cols, k=6):
    df = df.sort_values("game_date").reset_index(drop=True)
    folds = np.array_split(np.arange(len(df)), k)
    lifts = []
    chosen = []
    mask_n = []
    for fold in folds:
        is_test = np.zeros(len(df), dtype=bool)
        is_test[fold] = True
        train = df.loc[~is_test].copy().reset_index(drop=True)
        test = df.loc[is_test].copy().reset_index(drop=True)
        rule, _tested, _hits = best_rule(train, cols)
        if rule is None:
            lifts.append(0.0)
            chosen.append("NONE")
            mask_n.append(0)
            continue
        p, mask = apply_named(test, rule, train)
        lifts.append(accuracy(test.home_win, p) - accuracy(test.home_win, test.p_home))
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


def load_data():
    train_cols = [
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
    ]
    wf = pd.read_parquet(WF_PATH)
    schema_cols = pd.read_parquet(TRAIN_PATH).columns.tolist()
    train = pd.read_parquet(TRAIN_PATH, columns=[c for c in train_cols if c in schema_cols])
    wf["game_date"] = pd.to_datetime(wf.game_date)
    wf = wf.dropna(subset=["home_win", "p_home"]).copy()
    wf["home_win"] = wf.home_win.astype(int)
    return wf.merge(train, on="game_pk", how="left").sort_values("game_date").reset_index(drop=True)


def main():
    df = load_data()
    cut = df.game_date.quantile(0.60)
    design = df[df.game_date <= cut].copy().reset_index(drop=True)
    holdout = df[df.game_date > cut].copy().reset_index(drop=True)

    angles = [
        ("Liquidez mercado: dispersion/proveedores", ["market_p_home_std", "market_n_providers", "market_spread"]),
        ("Steam tardio: line shift y desacuerdo", ["market_line_shift_home_pp", "market_line_shift_abs_pp", "market_total_shift", "market_p_home_std"]),
        ("Lineup vs mano amplificado por park", ["lineup_xwoba_vs_sp_hand_diff", "lineup_k_pct_vs_sp_hand_diff", "lineup_bb_pct_vs_sp_hand_diff", "lineup_barrel_vs_sp_hand_diff", "lineup_top4_xwoba_vs_sp_hand_diff", "park_runs_factor", "park_off_xwoba_diff", "park_starter_xwoba_diff", "park_run_diff_diff"]),
        ("Estado de serie / revenge local", ["series_run_diff_so_far", "did_home_win_previous_game_in_series"]),
        ("Umpire + entorno de carreras", ["ump_acc", "ump_acc_above_x", "ump_run_impact", "market_over_under", "pred_total", "weather_temp_f"]),
        ("Fatiga bullpen/rest diferencial", ["bullpen_ip_l5d_diff", "bullpen_apps_l3d_diff", "bullpen_runs_l5_diff", "bullpen_ip_min_l3_diff", "starter_days_rest_a", "days_since_last_game_diff"]),
    ]

    print("BASELINE")
    print(metric_table(df, df.p_home.to_numpy()).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("SPLIT", design.game_date.min().date(), design.game_date.max().date(), len(design), "|", holdout.game_date.min().date(), holdout.game_date.max().date(), len(holdout))

    summary = []
    for angle, cols in angles:
        cols = [c for c in cols if c in df.columns]
        rule, tested, hits = best_rule(design, cols)
        print(f"\nANGLE {angle}")
        print("tested", tested, "eligible_design_hits", hits)
        if rule is None:
            print("NO DESIGN SIGNAL")
            summary.append({"angle": angle, "tested": tested, "status": "rejected_no_design"})
            continue

        holdout_p, _ = apply_named(holdout, rule, design)
        all_p, _ = apply_named(df, rule, design)
        holdout_table = metric_table(holdout, holdout_p)
        all_table = metric_table(df, all_p)
        kfold = temporal_kfold(df, cols)
        holdout_lift = float(holdout_table[holdout_table.season == "ALL"].lift.iloc[0])
        nonnegative_seasons = int((holdout_table[holdout_table.season != "ALL"].lift >= 0).sum())
        status = "survived" if holdout_lift > 0 and nonnegative_seasons >= 3 and kfold["win_rate"] >= 0.70 else "rejected"

        print("best", {k: rule[k] for k in ["name", "action", "n", "sub_base_acc", "design_lift"]})
        print("HOLDOUT")
        print(holdout_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
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
                "holdout_lift": holdout_lift,
                "all_lift": float(all_table[all_table.season == "ALL"].lift.iloc[0]),
                "holdout_seasons_nonnegative": nonnegative_seasons,
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
