"""
Feature engine - spec section 4/6: orchestrates rolling windows,
distribution stats, transitions, run-length, and entropy on every
tick, producing one FeatureSnapshot consumed by the model layer.

Critically: this is the SAME object used by main.py (live) and
validation/historical_simulator.py (backtest) - there is exactly one
feature-computation code path, which is how we avoid the
train/production skew the spec explicitly warns against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from features.rolling_windows import MultiScaleWindows
from features.digit_statistics import compute_distribution, DistributionSnapshot
from features.transition_features import TransitionModel
from features.run_features import RunLengthModel
from features.entropy_features import normalized_entropy
from features.pattern_features import DigitSequenceAnalyzer


@dataclass
class FeatureSnapshot:
    tick_index: int
    last_digit: Optional[int]
    distributions: Dict[int, DistributionSnapshot]  # window_size -> snapshot
    p_win_given_last_digit: float
    p_win_given_last_digit_n: int
    p_win_given_last_two: float
    p_win_given_last_two_n: int
    p_win_given_last_three: float
    p_win_given_last_three_n: int
    run_p_win: float
    run_sample_size: int
    run_type: Optional[str]
    run_length: int
    p_win_seq2: float
    p_win_seq2_n: int
    p_win_seq3: float
    p_win_seq3_n: int
    entropy_norm: Dict[int, float]  # window_size -> normalized entropy
    ready: bool  # has min_history_size been reached


class FeatureEngine:
    def __init__(self, window_sizes, min_history_size: int):
        self.windows = MultiScaleWindows(window_sizes)
        self.transitions = TransitionModel()
        self.runs = RunLengthModel()
        self.seq2 = DigitSequenceAnalyzer(k=2)
        self.seq3 = DigitSequenceAnalyzer(k=3)
        self.min_history_size = min_history_size
        self._last_digit: Optional[int] = None

    def update(self, digit: int) -> FeatureSnapshot:
        self.windows.push(digit)
        self.transitions.update(digit)
        self.runs.update(digit)
        self.seq2.update(digit)
        self.seq3.update(digit)

        distributions = {
            size: compute_distribution(self.windows.get(size))
            for size in self.windows.sizes
        }
        entropy_norm = {
            size: normalized_entropy(self.windows.get(size).counts)
            for size in self.windows.sizes
        }

        pw_last = self.transitions.p_win_given_last(self._last_digit)
        pw_last2 = self.transitions.p_win_given_last_two()
        pw_last3 = self.transitions.p_win_given_last_three()
        run_p, run_n = self.runs.p_win_next()
        seq2_est = self.seq2.p_win_next()
        seq3_est = self.seq3.p_win_next()

        snapshot = FeatureSnapshot(
            tick_index=self.windows.total_ticks,
            last_digit=digit,
            distributions=distributions,
            p_win_given_last_digit=pw_last.probability,
            p_win_given_last_digit_n=pw_last.sample_size,
            p_win_given_last_two=pw_last2.probability,
            p_win_given_last_two_n=pw_last2.sample_size,
            p_win_given_last_three=pw_last3.probability,
            p_win_given_last_three_n=pw_last3.sample_size,
            run_p_win=run_p,
            run_sample_size=run_n,
            run_type=self.runs.current.run_type,
            run_length=self.runs.current.run_length,
            p_win_seq2=seq2_est.probability,
            p_win_seq2_n=seq2_est.sample_size,
            p_win_seq3=seq3_est.probability,
            p_win_seq3_n=seq3_est.sample_size,
            entropy_norm=entropy_norm,
            ready=self.windows.total_ticks >= self.min_history_size,
        )
        self._last_digit = digit
        return snapshot
