"""
Bayesian Beta-Binomial model - spec section 10.

Prior centered on the theoretical baseline (0.70) with configurable
strength (pseudo-observations). Posterior updates on the recent
window's observed win/loss counts.
"""
from __future__ import annotations

from features.feature_engine import FeatureSnapshot
from models.base import ModelOutput


class BayesianModel:
    def __init__(self, baseline_p: float = 0.70, prior_strength: float = 10.0, window_size: int = 250):
        self.alpha0 = baseline_p * prior_strength
        self.beta0 = (1 - baseline_p) * prior_strength
        self.window_size = window_size

    def predict(self, snap: FeatureSnapshot) -> ModelOutput:
        dist = snap.distributions.get(self.window_size)
        if dist is None or dist.n == 0:
            return ModelOutput("bayesian", float("nan"), 0.0, 0, float("nan"))
        wins = round(dist.p_over2 * dist.n)
        losses = dist.n - wins
        alpha = self.alpha0 + wins
        beta = self.beta0 + losses
        mean = alpha / (alpha + beta)
        # posterior std of a Beta(alpha, beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = var ** 0.5
        confidence = max(0.0, 1.0 - std / 0.25)
        return ModelOutput(
            model_name="bayesian",
            probability=mean,
            confidence=confidence,
            sample_size=dist.n,
            uncertainty=std,
        )
