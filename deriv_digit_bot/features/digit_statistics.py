"""
Distribution statistics per rolling window (spec section 5).

Implements:
  - per-digit frequency vs expected uniform frequency, z-score
  - P_empirical(next_digit > 2)
  - Wilson score confidence interval on that proportion (better
    behaved than a normal-approx CI at small n / extreme p)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from features.rolling_windows import RollingWindow


@dataclass
class DigitDeviation:
    digit: int
    count: int
    frequency: float
    expected_frequency: float
    deviation: float
    z_score: float


@dataclass
class DistributionSnapshot:
    n: int
    p_over2: float
    wilson_low: float
    wilson_high: float
    per_digit: List[DigitDeviation]


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def digit_z_score(count: int, n: int, p_null: float = 0.1) -> float:
    """z-score of an observed digit count against a null of uniform p=0.1."""
    if n == 0:
        return 0.0
    expected = n * p_null
    var = n * p_null * (1 - p_null)
    if var <= 0:
        return 0.0
    return (count - expected) / math.sqrt(var)


def compute_distribution(window: RollingWindow) -> DistributionSnapshot:
    n = window.n
    per_digit = []
    for d in range(10):
        c = window.counts[d]
        freq = c / n if n else 0.0
        expected = 0.1
        per_digit.append(DigitDeviation(
            digit=d,
            count=c,
            frequency=freq,
            expected_frequency=expected,
            deviation=freq - expected,
            z_score=digit_z_score(c, n),
        ))
    over2 = window.count_over2()
    low, high = wilson_interval(over2, n)
    p = over2 / n if n else float("nan")
    return DistributionSnapshot(n=n, p_over2=p, wilson_low=low, wilson_high=high, per_digit=per_digit)
