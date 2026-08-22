"""
Run-length model features - spec section 9.

Tracks the current streak (type + length) and, historically, what
happened next after streaks of each (type, length) - so the model can
be validated (does continuation/reversal actually predict anything)
rather than assumed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

MAX_TRACKED_RUN_LENGTH = 8  # lengths beyond this are bucketed together


@dataclass
class RunState:
    run_type: Optional[str]  # "W" or "L"
    run_length: int


class RunLengthModel:
    def __init__(self):
        self.current = RunState(run_type=None, run_length=0)
        # key: (run_type, capped_length) -> {"W": n, "L": n}
        self.next_after_run: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: {"W": 0, "L": 0})

    def _cap(self, length: int) -> int:
        return min(length, MAX_TRACKED_RUN_LENGTH)

    def update(self, digit: int) -> None:
        cur = "W" if digit > 2 else "L"
        if self.current.run_type is not None:
            key = (self.current.run_type, self._cap(self.current.run_length))
            self.next_after_run[key][cur] += 1

        if cur == self.current.run_type:
            self.current.run_length += 1
        else:
            self.current = RunState(run_type=cur, run_length=1)

    def p_win_next(self) -> Tuple[float, int]:
        """Probability of W next given the current run state, plus sample size."""
        if self.current.run_type is None:
            return float("nan"), 0
        key = (self.current.run_type, self._cap(self.current.run_length))
        row = self.next_after_run[key]
        n = row["W"] + row["L"]
        if n == 0:
            return float("nan"), 0
        return row["W"] / n, n
