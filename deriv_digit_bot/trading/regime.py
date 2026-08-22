"""
Lightweight regime classifier - spec section 12, condensed.

Rather than a large learned regime model (which itself would need
walk-forward validation to trust), this applies transparent,
inspectable thresholds on features we already compute: model
dispersion and entropy. This is intentionally conservative - its main
job is to say UNSTABLE / MODEL_CONFLICT and block trading, not to
claim it has found a profitable regime.
"""
from __future__ import annotations

from dataclasses import dataclass

NORMAL = "NORMAL"
MODEL_CONFLICT = "MODEL_CONFLICT"
UNSTABLE = "UNSTABLE"
LOW_SAMPLE = "LOW_SAMPLE"


@dataclass
class RegimeResult:
    regime: str
    tradeable: bool
    reason: str


def classify(std_probability: float, n_models_used: int, entropy_norm_100: float,
             max_dispersion: float) -> RegimeResult:
    if n_models_used < 2:
        return RegimeResult(LOW_SAMPLE, False, "Fewer than 2 models produced a usable estimate")
    if std_probability > max_dispersion:
        return RegimeResult(MODEL_CONFLICT, False,
                             f"Model dispersion {std_probability:.3f} exceeds max {max_dispersion:.3f}")
    if entropy_norm_100 < 0.90:
        # a materially non-uniform 100-tick window is itself a signal
        # something odd is happening (feed issue, degenerate data, etc.)
        return RegimeResult(UNSTABLE, False, f"Entropy compressed to {entropy_norm_100:.3f} (expect ~1.0 for R_100)")
    return RegimeResult(NORMAL, True, "within normal parameters")
