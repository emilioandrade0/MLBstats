from __future__ import annotations

import json
from pathlib import Path

import analyze_opponent_adjusted_form as common
import build_data as bd
from swept_next_series import attach_swept_next_series_features


OUTPUT = bd.ROOT / "work" / "swept_next_series_walkforward_20260825.json"


def main():
    frame, coverage = attach_swept_next_series_features(bd.ROOT, common.base_frame())
    base_features = bd.features_for_mask(bd.DEFAULT_FACTOR_MASK)
    variants = {
        "symmetric_edge": ["next_series_after_swept_edge"],
        "split_by_location": ["home_next_series_after_swept", "away_next_series_after_swept"],
        "edge_with_progress": [
            "next_series_after_swept_edge",
            "next_series_after_swept_progress_edge",
        ],
        "full_context": [
            "home_next_series_after_swept",
            "away_next_series_after_swept",
            "next_series_after_swept_progress_edge",
            "both_next_series_after_swept",
        ],
        "visitor_vulnerability": ["away_next_series_after_swept"],
        "home_recovery": ["home_next_series_after_swept"],
    }
    baseline = common.evaluate(frame, base_features)
    candidate_frames = {
        name: common.evaluate(frame, base_features + features)
        for name, features in variants.items()
    }
    results = [
        common.summarize(candidate_frames[name], baseline, name, features)
        for name, features in variants.items()
    ]
    eligible_design = [
        item
        for item in results
        if item["seasons"]["2024"]["deltaPoints"] >= 0
        and item["seasons"]["2025"]["deltaPoints"] >= 0
    ]
    selected = sorted(
        eligible_design or results,
        key=lambda item: (-item["designDeltaPoints"], -item["accuracy"], len(item["features"])),
    )[0]
    selected["passesGate"] = bool(
        selected["designDeltaPoints"] > 0
        and selected["holdoutDeltaPoints"] > 0
        and selected["h1_2026"]["deltaPoints"] >= 0
        and selected["h2_2026"]["deltaPoints"] >= 0
    )
    selected_frame = candidate_frames[selected["name"]]
    changed = selected_frame["probability"].ge(0.5).ne(baseline["probability"].ge(0.5))
    selected["decisionChanges"] = {
        "games": int(changed.sum()),
        "baselineCorrect": int(baseline.loc[changed, "correct"].sum()),
        "candidateCorrect": int(selected_frame.loc[changed, "correct"].sum()),
        "netCorrect": int(
            selected_frame.loc[changed, "correct"].sum() - baseline.loc[changed, "correct"].sum()
        ),
    }
    results.sort(key=lambda item: (-item["designDeltaPoints"], -item["accuracy"]))
    report = bd.clean({
        "methodology": {
            "mode": "Monthly expanding walk-forward",
            "selection": "Elegir la variante sólo con 2024-2025 entre opciones no negativas en ambos años.",
            "gate": "Exigir mejora en diseño, 2026, H1 2026 y H2 2026 sin retrocesos.",
            "leakageRule": "La señal identifica la serie inmediata posterior a una barrida 0-3 y no usa su resultado futuro.",
        },
        "coverage": coverage,
        "baseline": common.summarize(baseline, baseline, "baseline", []),
        "selected": selected,
        "results": results,
    })
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
