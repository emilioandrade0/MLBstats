from __future__ import annotations

import json

import analyze_opponent_adjusted_form as common
import build_data as bd
from lineup_fatigue import TEST_FEATURES as FATIGUE_FEATURES
from lineup_fatigue import attach_lineup_fatigue_features
from opponent_adjusted_form import attach_opponent_adjusted_form
from rotation_quality import TEST_FEATURES as QUALITY_FEATURES
from rotation_quality import attach_rotation_quality_features
from sustainable_offense import attach_sustainable_offense


OPPONENT_SELECTED = ["edge_opponentStrengthL10", "edge_opponentStrengthL20"]
OFFENSE_SELECTED = ["edge_conversionGapL5", "edge_conversionGapL10", "edge_lobRateL5", "edge_lobRateL10"]


def main():
    frame = common.base_frame()
    frame, _ = attach_lineup_fatigue_features(bd.ROOT, frame, "2026-08-23")
    frame, _ = attach_rotation_quality_features(bd.ROOT, frame, "2026-08-23")
    frame, _ = attach_opponent_adjusted_form(frame, "2026-08-23")
    frame, _ = attach_sustainable_offense(bd.ROOT, frame, "2026-08-23")
    base = bd.features_for_mask(bd.DEFAULT_FACTOR_MASK)
    baselines = {
        "base": base,
        "fatigue": base + FATIGUE_FEATURES,
        "quality": base + QUALITY_FEATURES,
        "fatigue_quality": base + FATIGUE_FEATURES + QUALITY_FEATURES,
    }
    output = {}
    for baseline_name, baseline_features in baselines.items():
        baseline_predictions = common.evaluate(frame, baseline_features)
        variants = {}
        for signal_name, signal_features in (("opponent_context", OPPONENT_SELECTED), ("sustainable_offense", OFFENSE_SELECTED)):
            candidate = common.evaluate(frame, baseline_features + signal_features)
            result = common.summarize(candidate, baseline_predictions, f"{baseline_name}+{signal_name}", signal_features)
            result["passesGate"] = bool(
                result["designDeltaPoints"] > 0 and result["holdoutDeltaPoints"] > 0
                and result["h1_2026"]["deltaPoints"] >= 0 and result["h2_2026"]["deltaPoints"] >= 0
            )
            variants[signal_name] = result
        output[baseline_name] = variants
    print(json.dumps(bd.clean(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
