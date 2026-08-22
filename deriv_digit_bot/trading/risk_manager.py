"""Hard risk controls - spec section 19. The bot must never bypass these."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskState:
    starting_balance: float
    balance: float
    daily_pnl: float = 0.0
    peak_balance: float = 0.0
    consecutive_losses: int = 0
    max_daily_loss: float = 50.0
    max_drawdown: float = 75.0
    max_consecutive_losses: int = 6
    halted: bool = False
    halt_reason: Optional[str] = None

    def __post_init__(self):
        self.peak_balance = self.starting_balance

    def record_trade_result(self, pnl: float) -> None:
        self.balance += pnl
        self.daily_pnl += pnl
        self.peak_balance = max(self.peak_balance, self.balance)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self._evaluate()

    def _evaluate(self) -> None:
        drawdown = self.peak_balance - self.balance
        if self.daily_pnl <= -abs(self.max_daily_loss):
            self.halted = True
            self.halt_reason = f"Max daily loss reached ({self.daily_pnl:.2f})"
        elif drawdown >= self.max_drawdown:
            self.halted = True
            self.halt_reason = f"Max drawdown reached ({drawdown:.2f})"
        elif self.consecutive_losses >= self.max_consecutive_losses:
            self.halted = True
            self.halt_reason = f"Max consecutive losses reached ({self.consecutive_losses})"

    def can_trade(self) -> bool:
        return not self.halted

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0

    def manual_kill(self, reason: str = "manual kill switch") -> None:
        self.halted = True
        self.halt_reason = reason
