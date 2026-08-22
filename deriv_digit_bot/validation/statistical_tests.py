"""
Statistical significance tests - spec section 22/5, the piece that
was missing: formal hypothesis tests against the null of a uniform,
memoryless digit generator, rather than just eyeballing a raw
probability.

Used by models/significance_model.py to decide whether an observed
deviation is worth reporting at all - it should trip roughly at the
chosen alpha's false-positive rate on genuinely random data, which is
the expected and correct behavior, not a bug.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from scipy import stats


@dataclass
class ChiSquareResult:
    statistic: float
    p_value: float
    reject_uniform: bool


@dataclass
class ZTestResult:
    z_statistic: float
    p_value: float
    reject_baseline: bool


def chi_square_uniform_test(counts: List[int], alpha: float = 0.05) -> ChiSquareResult:
    """Goodness-of-fit test: are these 10 digit counts consistent with
    a uniform distribution? df=9."""
    n = sum(counts)
    if n < 20:
        return ChiSquareResult(float("nan"), float("nan"), False)
    expected = [n / 10.0] * 10
    stat, p = stats.chisquare(f_obs=counts, f_exp=expected)
    return ChiSquareResult(statistic=float(stat), p_value=float(p), reject_uniform=p < alpha)


def z_test_over2(count_over2: int, n: int, p0: float = 0.70, alpha: float = 0.05) -> ZTestResult:
    """Two-sided test of H0: P(over2) == p0 (the theoretical baseline),
    NOT H0: p == 0.5. Rejecting this only means "different from 0.70",
    which could be a data/feed problem as easily as an edge - it is a
    necessary, not sufficient, condition to trust a probability."""
    if n < 20:
        return ZTestResult(float("nan"), float("nan"), False)
    p_hat = count_over2 / n
    se = (p0 * (1 - p0) / n) ** 0.5
    if se == 0:
        return ZTestResult(float("nan"), float("nan"), False)
    z = (p_hat - p0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return ZTestResult(z_statistic=float(z), p_value=float(p_value), reject_baseline=p_value < alpha)
