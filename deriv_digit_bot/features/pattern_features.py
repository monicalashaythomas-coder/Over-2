"""
Digit sequence pattern analyzer - spec section 8, the raw-digit
variant (as distinct from run_features.py's W/L run tracking).

Tracks what actually followed each observed sequence of the last k
raw digits, e.g. (7, 2) -> {digit 3: 4 times, digit 9: 2 times, ...}.
State space grows fast (10^k), so this needs real volume before any
given sequence has enough observations to trust - that gate is
enforced here via MIN_SUPPORT, not left to the caller.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

MIN_SUPPORT = 50


@dataclass
class SequenceEstimate:
    probability: float
    sample_size: int


class DigitSequenceAnalyzer:
    def __init__(self, k: int, min_support: int = MIN_SUPPORT):
        self.k = k
        self.min_support = min_support
        self.next_after_seq: Dict[Tuple[int, ...], Dict[str, int]] = defaultdict(lambda: {"W": 0, "L": 0})
        self._recent: Deque[int] = deque(maxlen=k)

    def update(self, digit: int) -> None:
        outcome = "W" if digit > 2 else "L"
        if len(self._recent) == self.k:
            key = tuple(self._recent)
            self.next_after_seq[key][outcome] += 1
        self._recent.append(digit)

    def p_win_next(self) -> SequenceEstimate:
        if len(self._recent) < self.k:
            return SequenceEstimate(float("nan"), 0)
        key = tuple(self._recent)
        row = self.next_after_seq.get(key)
        if row is None:
            return SequenceEstimate(float("nan"), 0)
        n = row["W"] + row["L"]
        if n < self.min_support:
            return SequenceEstimate(float("nan"), n)
        return SequenceEstimate(row["W"] / n, n)
