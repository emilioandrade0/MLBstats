"""Multi-seed ensemble wrapper — averages N seed models to reduce variance.

Used by both walkforward.py (evaluation) and train.py (production).
Exposes the same sklearn-like API as the underlying models so it slots into
existing pickle bundles at api.py load time transparently:

    cls_bundle["model"].predict_proba(X)[:, 1]
    cls_bundle["calibrator"].predict_proba(X)[:, 1]
    reg_bundle["model"].predict(X)

Walk-forward measured gain on n=5634 (2024-05 to 2026-06): +0.06pp acc,
+0.07pp AUC, better log_loss vs single-seed baseline.
"""
from __future__ import annotations

import numpy as np


SEEDS = (42, 43, 44, 45, 46)


class SeedEnsemble:
    """Averages predictions from N seed-trained models.

    Attributes:
        models: list of trained sklearn-compatible estimators
        kind: "cls" (predict_proba returns 2D) or "reg" (predict returns 1D)
    """
    def __init__(self, models: list, kind: str):
        self.models = models
        self.kind = kind

    def predict_proba(self, X) -> np.ndarray:
        assert self.kind == "cls", "predict_proba only valid for classifiers"
        p1 = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        if self.kind == "cls":
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
        return np.mean([m.predict(X) for m in self.models], axis=0)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average feature importances across seeds — for interpretability."""
        fis = [m.feature_importances_ for m in self.models
               if hasattr(m, "feature_importances_")]
        if not fis:
            return np.array([])
        return np.mean(fis, axis=0)
