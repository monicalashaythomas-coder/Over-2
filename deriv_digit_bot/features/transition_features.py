"""
Transition (Markov) features - spec section 7.

Tracks:
  - first-order digit->digit transition counts (10x10)
  - grouped W/L (W = digit>2, L = digit<=2) first-order transitions
  - grouped W/L second- and third-order transitions (conditioned on
    the last two/three outcomes), only trusted once sample size
    clears a minimum - the state space triples in size in going from
    order 2 (4 states) to order 3 (8 states), so order 3 needs
    considerably more data before it's trustworthy.

All estimates carry sample_size so downstream consumers can apply
shrinkage toward baseline for sparse states (spec requirement: "Do
not allow sparse transition states to dominate the final ensemble").
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

MIN_SAMPLES_2ND_ORDER = 30
MIN_SAMPLES_3RD_ORDER = 80


def outcome(digit: int) -> str:
    return "W" if digit > 2 else "L"


@dataclass
class TransitionEstimate:
    probability: float
    sample_size: int


class TransitionModel:
    def __init__(self):
        self.digit_transition = [[0] * 10 for _ in range(10)]  # [from][to]
        self.wl_first: Dict[str, Dict[str, int]] = defaultdict(lambda: {"W": 0, "L": 0})
        self.wl_second: Dict[str, Dict[str, int]] = defaultdict(lambda: {"W": 0, "L": 0})
        self.wl_third: Dict[str, Dict[str, int]] = defaultdict(lambda: {"W": 0, "L": 0})
        self._last_digit: Optional[int] = None
        self._recent_outcomes: Deque[str] = deque(maxlen=3)  # most recent last, i.e. [-1] = most recent

    def update(self, digit: int) -> None:
        cur_state = outcome(digit)

        if self._last_digit is not None:
            self.digit_transition[self._last_digit][digit] += 1
            self.wl_first[outcome(self._last_digit)][cur_state] += 1

        if len(self._recent_outcomes) >= 2:
            key2 = "".join(list(self._recent_outcomes)[-2:])  # two-ago, one-ago
            self.wl_second[key2][cur_state] += 1

        if len(self._recent_outcomes) >= 3:
            key3 = "".join(list(self._recent_outcomes)[-3:])  # three-ago..one-ago
            self.wl_third[key3][cur_state] += 1

        self._recent_outcomes.append(cur_state)
        self._last_digit = digit

    def p_win_given_last(self, current_digit: Optional[int]) -> TransitionEstimate:
        if current_digit is None:
            return TransitionEstimate(probability=float("nan"), sample_size=0)
        row = self.wl_first[outcome(current_digit)]
        n = row["W"] + row["L"]
        if n == 0:
            return TransitionEstimate(probability=float("nan"), sample_size=0)
        return TransitionEstimate(probability=row["W"] / n, sample_size=n)

    def p_win_given_last_two(self) -> TransitionEstimate:
        if len(self._recent_outcomes) < 2:
            return TransitionEstimate(probability=float("nan"), sample_size=0)
        key = "".join(list(self._recent_outcomes)[-2:])
        row = self.wl_second[key]
        n = row["W"] + row["L"]
        if n < MIN_SAMPLES_2ND_ORDER:
            return TransitionEstimate(probability=float("nan"), sample_size=n)
        return TransitionEstimate(probability=row["W"] / n, sample_size=n)

    def p_win_given_last_three(self) -> TransitionEstimate:
        if len(self._recent_outcomes) < 3:
            return TransitionEstimate(probability=float("nan"), sample_size=0)
        key = "".join(list(self._recent_outcomes)[-3:])
        row = self.wl_third[key]
        n = row["W"] + row["L"]
        if n < MIN_SAMPLES_3RD_ORDER:
            return TransitionEstimate(probability=float("nan"), sample_size=n)
        return TransitionEstimate(probability=row["W"] / n, sample_size=n)

