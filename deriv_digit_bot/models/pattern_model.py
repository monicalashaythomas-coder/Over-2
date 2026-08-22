"""
Pattern model - spec section 8, condensed.

Rather than a separate pattern-mining structure, this reuses the
run-length model's (type, length) states, which already encode the
W/W, W/L, L/L... pattern space up to the tracked depth, and applies a
stricter minimum-support requirement than the run-length model itself
(spec: "reject patterns with low sample size... implement minimum
support requirements"). Kept as a distinct ensemble member so it can
be weighted/disabled independently if it adds nothing.
"""
from __future__ import annotations

from features.feature_engine import FeatureSnapshot
from models.base import ModelOutput

MIN_SUPPORT = 40


class PatternModel:
    def predict(self, snap: FeatureSnapshot) -> ModelOutput:
        if snap.run_sample_size < MIN_SUPPORT:
            return ModelOutput("pattern", float("nan"), 0.0, snap.run_sample_size, float("nan"))
        conf = min(1.0, snap.run_sample_size / 250)
        return ModelOutput(
            "pattern", snap.run_p_win, conf, snap.run_sample_size,
            uncertainty=1.0 / max(1, snap.run_sample_size) ** 0.5,
        )
