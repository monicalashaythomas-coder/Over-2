from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelOutput:
    model_name: str
    probability: float   # P(next digit > 2), NaN if the model abstains
    confidence: float     # 0..1, model's own confidence in this estimate
    sample_size: int
    uncertainty: float    # e.g. half-width of a CI, or posterior std

    def abstains(self) -> bool:
        import math
        return math.isnan(self.probability) or self.sample_size == 0
