"""
Martingale state machine - spec section 18.

Critically: this class does NOT decide to place the next trade. It
only tracks level/stake and exposes the confidence threshold required
for the CURRENT level. The signal engine must independently produce a
fresh, passing signal after every loss before martingale.next_stake()
is ever used - there is no auto-continue path here at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MartingaleState:
    base_stake: float
    multiplier: float
    max_steps: int
    thresholds: List[float]  # required calibrated probability per level
    level: int = 0
    in_cooldown: bool = False
    cooldown_remaining: int = 0

    def current_stake(self) -> float:
        return round(self.base_stake * (self.multiplier ** self.level), 2)

    def required_threshold(self) -> float:
        idx = min(self.level, len(self.thresholds) - 1)
        return self.thresholds[idx]

    def register_win(self) -> None:
        self.level = 0
        self.in_cooldown = False
        self.cooldown_remaining = 0

    def register_loss(self, cooldown_ticks: int) -> None:
        if self.level >= self.max_steps:
            # exhausted the martingale ladder - reset and cool down
            self.level = 0
            self.in_cooldown = True
            self.cooldown_remaining = cooldown_ticks
        else:
            self.level += 1

    def tick_cooldown(self) -> None:
        if self.in_cooldown:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.in_cooldown = False

    def total_cycle_exposure(self) -> float:
        return round(sum(self.base_stake * (self.multiplier ** i) for i in range(self.max_steps + 1)), 2)
