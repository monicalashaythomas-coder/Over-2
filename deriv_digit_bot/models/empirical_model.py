"""Empirical distribution model - spec section 5, applied per window."""
from __future__ import annotations

from features.feature_engine import FeatureSnapshot
from models.base import ModelOutput

MIN_SAMPLE = 20


class EmpiricalModel:
    """Uses the mid-length window (100) as its primary read, since it
    balances responsiveness against sample-size stability. Confidence
    scales with how tight the Wilson CI is."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def predict(self, snap: FeatureSnapshot) -> ModelOutput:
        dist = snap.distributions.get(self.window_size)
        if dist is None or dist.n < MIN_SAMPLE:
            return ModelOutput("empirical", float("nan"), 0.0, dist.n if dist else 0, float("nan"))
        width = dist.wilson_high - dist.wilson_low
        confidence = max(0.0, 1.0 - width / 0.5)  # width of 0.5 -> conf 0; width of 0 -> conf 1
        return ModelOutput(
            model_name="empirical",
            probability=dist.p_over2,
            confidence=confidence,
            sample_size=dist.n,
            uncertainty=width / 2,
        )
