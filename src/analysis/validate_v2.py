"""Round 2: rules that swap to MARKET pick (not blind-flip to dog).

Flipping trades accuracy for longer odds. Switching to market in suspect zones
might preserve accuracy by leveraging the market's calibration.

Rules tested:
  v1 — fav_band 1.60-1.70: use market pick (instead of model)
  v2 — fav_band 1.60-1.70 + market_disagrees: use market pick
  v3 — July: use market pick (instead of model)
  v4 — Wide low_conf [0.50, 0.62] + disagrees: use market pick
  v5 — Combine: any of (v2, v3, v4) → use market pick
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validate_new_rules import load, stats_for


def main() -> None:
    df = load()
    baseline = stats_for(df["model_home"], df)
    print(f"BASELINE: n={baseline['n']:,}  "
          f"acc={baseline['acc']:.4f}  roi={baseline['roi']:+.4f}\n")

    rules = []

    def test(name, mask, swap_to_mkt=True):
        pick = df["model_home"].copy()
        if swap_to_mkt:
            pick.loc[mask] = df.loc[mask, "mkt_home"]
        else:
            pick.loc[mask] = ~pick.loc[mask]
        s = stats_for(pick, df)
        d_acc = (s["acc"] - baseline["acc"]) * 100
        d_roi = (s["roi"] - baseline["roi"]) * 100
        ok_acc = s["acc"] > baseline["acc"]
        ok_roi = s["roi"] > baseline["roi"]
        verdict = "✅" if (ok_acc and ok_roi) else ("⚠" if ok_acc or ok_roi else "❌")
        print(f"  {name:<48} n_change={mask.sum():>5}  "
              f"acc={s['acc']:.4f} ({d_acc:+.2f}pp)  "
              f"roi={s['roi']:+.4f} ({d_roi:+.2f}pp) {verdict}")
        rules.append({"name": name, "ok_acc": ok_acc, "ok_roi": ok_roi,
                      "mask": mask, "acc": s["acc"], "roi": s["roi"]})
        return ok_acc and ok_roi

    print("=" * 90)
    print("DEFER-TO-MARKET RULES (swap model's pick to market's pick)")
    print("=" * 90)

    fav_band = (df["fav_dec"] >= 1.60) & (df["fav_dec"] < 1.70)
    disagree = df["model_home"] != df["mkt_home"]

    test("v1: fav_band[1.60,1.70) → use market", fav_band)
    test("v2: fav_band[1.60,1.70) + disagree → use market", fav_band & disagree)
    test("v3: July → use market", df["month"] == 7)
    test("v4: p_pick∈[0.50,0.62] + disagree → use market",
         df["p_home"].between(0.50, 0.62).fillna(False) | (1-df["p_home"]).between(0.50, 0.62).fillna(False))
    p_pick = np.maximum(df["p_home"], 1 - df["p_home"])
    test("v4b: p_pick∈[0.50,0.62] + disagree → use market",
         (p_pick.between(0.50, 0.62)) & disagree)
    test("v5: p_pick∈[0.50,0.65] + disagree → use market",
         (p_pick.between(0.50, 0.65)) & disagree)
    test("v6: p_pick∈[0.50,0.58] + disagree → use market (current rule)",
         (p_pick.between(0.50, 0.58)) & disagree)

    # Wider band tests
    print()
    print("=" * 90)
    print("FAV BAND TESTS (different ranges)")
    print("=" * 90)
    for lo, hi in [(1.40, 1.60), (1.50, 1.70), (1.60, 1.80), (1.70, 1.90),
                   (1.55, 1.75), (1.50, 1.80)]:
        band = (df["fav_dec"] >= lo) & (df["fav_dec"] < hi)
        test(f"fav_band[{lo:.2f},{hi:.2f}) + disagree → mkt",
             band & disagree)

    # Team-fade variations
    print()
    print("=" * 90)
    print("TEAM FADE → use market instead of model")
    print("=" * 90)
    for teams in [{"ATL"}, {"MIN"}, {"BAL"}, {"ATL", "MIN"}, {"ATL", "MIN", "BAL"}]:
        mask = df["home_team_abbrev"].isin(teams) & df["model_home"]
        test(f"fade {teams} @ home → mkt", mask)

    # Best combined
    print()
    print("=" * 90)
    print("STACK: combine all rules with ✅ verdict")
    print("=" * 90)
    good_rules = [r for r in rules if r["ok_acc"] and r["ok_roi"]]
    if good_rules:
        combined_mask = pd.Series(False, index=df.index)
        for r in good_rules:
            combined_mask |= r["mask"]
        pick = df["model_home"].copy()
        pick.loc[combined_mask] = df.loc[combined_mask, "mkt_home"]
        s = stats_for(pick, df)
        print(f"  rules adopted: {len(good_rules)}")
        for r in good_rules:
            print(f"    · {r['name']}")
        print(f"  COMBINED: n_change={combined_mask.sum():,}  "
              f"acc={s['acc']:.4f} ({(s['acc']-baseline['acc'])*100:+.2f}pp)  "
              f"roi={s['roi']:+.4f} ({(s['roi']-baseline['roi'])*100:+.2f}pp)")
    else:
        print("  No rules with ✅ verdict — nothing to combine.")


if __name__ == "__main__":
    main()
