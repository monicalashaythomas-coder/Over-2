"""
Statistical significance model.

Wraps the chi-square and z-score hypothesis tests as an ensemble
member that only contributes a prediction when the deviation from
uniform is actually statistically significant at the configured
alpha - otherwise it abstains rather than reporting the empirical
estimate anyway. This is what makes "layers agreeing" mean something:
this model can only ever agree when there's a formally significant
deviation behind it, not merely a raw probability that happens to
look high.

Important, and worth restating: on truly random digits this WILL
still fire at approximately the alpha false-positive rate (5% of
windows by default). That is not a bug - it is what a correctly
calibrated significance test is supposed to do. Whether flagged
windows actually predict anything is exactly what the calibration
report in validation/quick_validation.py checks.
"""
from __future__ import annotations

from features.feature_engine import FeatureSnapshot
from models.base import ModelOutput
from validation.statistical_tests import chi_square_uniform_test, z_test_over2


class SignificanceModel:
    def __init__(self, window_size: int = 250, alpha: float = 0.05, baseline_p: float = 0.70):
        self.window_size = window_size
        self.alpha = alpha
        self.baseline_p = baseline_p

    def predict(self, snap: FeatureSnapshot) -> ModelOutput:
        dist = snap.distributions.get(self.window_size)
        if dist is None or dist.n < 50:
            return ModelOutput("significance", float("nan"), 0.0, dist.n if dist else 0, float("nan"))

        counts = [d.count for d in dist.per_digit]
        chi2 = chi_square_uniform_test(counts, alpha=self.alpha)
        over2 = round(dist.p_over2 * dist.n)
        z = z_test_over2(over2, dist.n, p0=self.baseline_p, alpha=self.alpha)

        if not (chi2.reject_uniform or z.reject_baseline):
            # No significant deviation detected - abstain rather than
            # report the raw empirical probability as if it meant something.
            return ModelOutput("significance", float("nan"), 0.0, dist.n, float("nan"))

        # Significant by at least one test: report the empirical P(over2)
        # for this window, with confidence scaled by how significant the
        # more informative (over2-specific) test is.
        import math
        p_value = z.p_value if not math.isnan(z.p_value) else chi2.p_value
        confidence = 0.5 if math.isnan(p_value) else max(0.0, min(1.0, 1.0 - p_value))
        return ModelOutput(
            model_name="significance",
            probability=dist.p_over2,
            confidence=confidence,
            sample_size=dist.n,
            uncertainty=p_value,
        )
