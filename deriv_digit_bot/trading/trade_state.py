"""Trade state machine - spec section 27."""
from __future__ import annotations

from enum import Enum, auto


class TradeState(Enum):
    INITIALIZING = auto()
    COLLECTING_DATA = auto()
    WARMING_UP = auto()
    OBSERVING = auto()
    SIGNAL_EVALUATION = auto()
    TRADE_READY = auto()
    TRADE_OPEN = auto()
    TRADE_RESULT = auto()
    MARTINGALE_REASSESSMENT = auto()
    COOLDOWN = auto()
    RISK_STOP = auto()
    ERROR = auto()


class StateMachine:
    def __init__(self):
        self.state = TradeState.INITIALIZING

    def transition(self, new_state: TradeState) -> None:
        self.state = new_state

    def is_tradeable_state(self) -> bool:
        return self.state in (TradeState.OBSERVING, TradeState.SIGNAL_EVALUATION,
                               TradeState.TRADE_READY, TradeState.MARTINGALE_REASSESSMENT)
