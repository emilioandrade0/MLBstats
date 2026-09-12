from __future__ import annotations

import json

import numpy as np

import analyze_opponent_adjusted_form as common
import build_data as bd
from sustainable_offense import SIGNAL_COLUMNS, attach_sustainable_offense


def main():
    frame, audit = attach_sustainable_offense(bd.ROOT, common.base_frame(), "2026-08-23")
    base_features = bd.features_for_mask(bd.DEFAULT_FACTOR_MASK)
    groups = {
        "process_core": ["edge_wobaL5", "edge_wobaL10", "edge_xbhRateL10"],
        "discipline": ["edge_disciplineL5", "edge_disciplineL10"],
        "conversion_regression": ["edge_conversionGapL5", "edge_conversionGapL10", "edge_lobRateL5", "edge_lobRateL10"],
        "multiwindow_process": ["edge_wobaL5", "edge_wobaL10", "edge_xbhRateL5", "edge_xbhRateL10", "edge_processTrend"],
        "runs_vs_process": ["edge_runRateL5", "edge_runRateL10", "edge_wobaL10", "edge_conversionGapL10"],
        "compact": ["edge_wobaL10", "edge_disciplineL10", "edge_conversionGapL10", "edge_processTrend"],
        "all_sustainable_offense": [f"edge_{name}" for name in SIGNAL_COLUMNS],
    }
    baseline = common.evaluate(frame, base_features)
    results = [common.summarize(common.evaluate(frame, base_features + features), baseline, name, features) for name, features in groups.items()]
    eligible_design = [item for item in results if item["seasons"]["2024"]["deltaPoints"] >= 0 and item["seasons"]["2025"]["deltaPoints"] >= 0]
    selected = sorted(eligible_design or results, key=lambda item: (-item["designDeltaPoints"], -item["accuracy"]))[0]
    selected["passesGate"] = bool(
        selected["designDeltaPoints"] > 0 and selected["holdoutDeltaPoints"] > 0
        and selected["h1_2026"]["deltaPoints"] >= 0 and selected["h2_2026"]["deltaPoints"] >= 0
    )
    results.sort(key=lambda item: (-item["designDeltaPoints"], -item["accuracy"]))
    print(json.dumps(bd.clean({
        "selectionRule": "Elegir con 2024-2025 entre variantes no negativas en ambos años; exigir mejora 2026 y no retroceder en H1/H2.",
        "coverage": audit["coverage"], "baseline": common.summarize(baseline, baseline, "baseline", []),
        "selected": selected, "results": results,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
