"""Raw digit-sequence pattern model - the "digit analyser" ensemble
member, distinct from the W/L run-based pattern_model.py. Prefers the
longer (3-digit) sequence when it has enough support, falls back to
the 2-digit sequence, abstains otherwise."""
from __future__ import annotations

from features.feature_engine import FeatureSnapshot
from models.base import ModelOutput


class DigitSequenceModel:
    def predict(self, snap: FeatureSnapshot) -> ModelOutput:
        if snap.p_win_seq3_n >= 50:
            p, n = snap.p_win_seq3, snap.p_win_seq3_n
            conf = min(1.0, n / 200)
            return ModelOutput("digit_sequence_3", p, conf, n, uncertainty=1.0 / max(1, n) ** 0.5)
        if snap.p_win_seq2_n >= 50:
            p, n = snap.p_win_seq2, snap.p_win_seq2_n
            conf = min(1.0, n / 200)
            return ModelOutput("digit_sequence_2", p, conf, n, uncertainty=1.0 / max(1, n) ** 0.5)
        return ModelOutput("digit_sequence", float("nan"), 0.0, snap.p_win_seq2_n, float("nan"))
