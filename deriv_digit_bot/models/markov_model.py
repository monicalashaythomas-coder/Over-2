"""Markov transition model - spec section 7. Cascades: prefers 3rd
order when it has enough support, then 2nd order, then 1st order,
abstains if none clear their minimum sample size."""
from __future__ import annotations

from features.feature_engine import FeatureSnapshot
from models.base import ModelOutput

MIN_SAMPLE_1ST = 30


class MarkovModel:
    def predict(self, snap: FeatureSnapshot) -> ModelOutput:
        if snap.p_win_given_last_three_n >= 80:
            p, n = snap.p_win_given_last_three, snap.p_win_given_last_three_n
            conf = min(1.0, n / 300)
            return ModelOutput("markov_order3", p, conf, n, uncertainty=1.0 / max(1, n) ** 0.5)
        if snap.p_win_given_last_two_n >= 30:
            p, n = snap.p_win_given_last_two, snap.p_win_given_last_two_n
            conf = min(1.0, n / 200)
            return ModelOutput("markov_order2", p, conf, n, uncertainty=1.0 / max(1, n) ** 0.5)
        if snap.p_win_given_last_digit_n >= MIN_SAMPLE_1ST:
            p, n = snap.p_win_given_last_digit, snap.p_win_given_last_digit_n
            conf = min(1.0, n / 300)
            return ModelOutput("markov_order1", p, conf, n, uncertainty=1.0 / max(1, n) ** 0.5)
        return ModelOutput("markov", float("nan"), 0.0, snap.p_win_given_last_digit_n, float("nan"))
