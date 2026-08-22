"""
Multi-scale rolling digit windows.

Each window maintains:
  - a deque of the last N digits
  - a running count array (10,) updated incrementally (O(1) per tick,
    not recomputed from scratch)

This directly implements spec section 4: rolling windows of
10/25/50/100/250/500/1000 ticks, using collections.deque and
incremental statistics rather than recalculating on every tick.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List


class RollingWindow:
    __slots__ = ("size", "buf", "counts", "_n")

    def __init__(self, size: int):
        self.size = size
        self.buf: Deque[int] = deque(maxlen=size)
        self.counts: List[int] = [0] * 10
        self._n = 0

    def push(self, digit: int) -> None:
        if len(self.buf) == self.size:
            evicted = self.buf[0]  # will be popped on append (maxlen)
            self.counts[evicted] -= 1
        else:
            self._n += 1
        self.buf.append(digit)
        self.counts[digit] += 1

    @property
    def n(self) -> int:
        return len(self.buf)

    def is_full(self) -> bool:
        return len(self.buf) == self.size

    def count_over2(self) -> int:
        return sum(self.counts[3:10])

    def p_over2(self) -> float:
        n = self.n
        if n == 0:
            return float("nan")
        return self.count_over2() / n


class MultiScaleWindows:
    """Owns one RollingWindow per configured size and updates them together."""

    def __init__(self, sizes: List[int]):
        self.sizes = sorted(sizes)
        self.windows = {s: RollingWindow(s) for s in self.sizes}
        self.total_ticks = 0
        # full unbounded history is needed for run/transition/pattern
        # features that look back further than any single window in
        # some configurations; kept modest since callers only need
        # the tail for pattern lookups.
        self.history: Deque[int] = deque(maxlen=max(self.sizes) * 2)

    def push(self, digit: int) -> None:
        self.total_ticks += 1
        for w in self.windows.values():
            w.push(digit)
        self.history.append(digit)

    def get(self, size: int) -> RollingWindow:
        return self.windows[size]
