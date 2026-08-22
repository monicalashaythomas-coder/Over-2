"""
Probability ensemble - spec section 13/15.

Combines individual model outputs into one raw probability, with
weights driven by each model's recent Brier score (lower error =
higher weight) rather than fixed forever. Also computes model
agreement/dispersion so the signal engine can refuse to trade when
models disagree even if the mean probability looks attractive.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List

from models.base import ModelOutput

MIN_OBSERVATIONS_BEFORE_REWEIGHT = 30
ROLLING_WINDOW = 300


@dataclass
class EnsembleResult:
    raw_probability: float
    weights: Dict[str, float]
    contributing_models: List[str]
    mean_probability: float
    std_probability: float
    n_models_used: int
    n_models_abstained: int


class ModelPerformanceTracker:
    """Tracks rolling Brier score per model to drive adaptive weights.
    Weights only start deviating from equal once a model has enough
    resolved predictions - a single lucky/unlucky streak on a tiny
    sample cannot dominate (spec section 25)."""

    def __init__(self):
        self._errors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))

    def record(self, model_name: str, predicted_p: float, outcome_win: bool) -> None:
        if math.isnan(predicted_p):
            return
        brier = (predicted_p - (1.0 if outcome_win else 0.0)) ** 2
        self._errors[model_name].append(brier)

    def weight_for(self, model_name: str) -> float:
        errs = self._errors.get(model_name)
        if not errs or len(errs) < MIN_OBSERVATIONS_BEFORE_REWEIGHT:
            return 1.0  # equal weight until proven otherwise
        mean_brier = sum(errs) / len(errs)
        return 1.0 / (mean_brier + 0.05)  # +epsilon avoids divide-by-zero / runaway weight


class Ensemble:
    def __init__(self, tracker: ModelPerformanceTracker | None = None):
        self.tracker = tracker or ModelPerformanceTracker()

    def combine(self, outputs: List[ModelOutput]) -> EnsembleResult:
        usable = [o for o in outputs if not o.abstains()]
        abstained = len(outputs) - len(usable)

        if not usable:
            return EnsembleResult(
                raw_probability=float("nan"), weights={}, contributing_models=[],
                mean_probability=float("nan"), std_probability=float("nan"),
                n_models_used=0, n_models_abstained=abstained,
            )

        raw_weights = {o.model_name: self.tracker.weight_for(o.model_name) * o.confidence for o in usable}
        total = sum(raw_weights.values())
        if total <= 0:
            # all confidences were zero - fall back to unweighted mean
            weights = {o.model_name: 1.0 / len(usable) for o in usable}
        else:
            weights = {k: v / total for k, v in raw_weights.items()}

        weighted_p = sum(weights[o.model_name] * o.probability for o in usable)

        probs = [o.probability for o in usable]
        mean_p = sum(probs) / len(probs)
        variance = sum((p - mean_p) ** 2 for p in probs) / len(probs)
        std_p = variance ** 0.5

        return EnsembleResult(
            raw_probability=weighted_p,
            weights=weights,
            contributing_models=[o.model_name for o in usable],
            mean_probability=mean_p,
            std_probability=std_p,
            n_models_used=len(usable),
            n_models_abstained=abstained,
        )
