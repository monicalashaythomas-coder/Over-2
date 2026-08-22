"""Run-length model - spec section 9. Must be empirically supported,
not assumed; abstains below a minimum sample size for the observed
run state."""
from __future__ import annotations

from features.feature_engine import FeatureSnapshot
from models.base import ModelOutput

MIN_SAMPLE = 25


class RunLengthModel:
    def predict(self, snap: FeatureSnapshot) -> ModelOutput:
        if snap.run_sample_size < MIN_SAMPLE:
            return ModelOutput("run_length", float("nan"), 0.0, snap.run_sample_size, float("nan"))
        conf = min(1.0, snap.run_sample_size / 150)
        return ModelOutput(
            "run_length", snap.run_p_win, conf, snap.run_sample_size,
            uncertainty=1.0 / max(1, snap.run_sample_size) ** 0.5,
        )
