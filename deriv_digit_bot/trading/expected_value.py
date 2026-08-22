"""Expected value engine - spec section 17. Uses the ACTUAL Deriv
payout/stake, never assumes fair odds."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EVResult:
    breakeven_probability: float
    expected_value: float


def compute_ev(probability: float, stake: float, payout: float) -> EVResult:
    """
    payout: total return if the contract wins (Deriv's `payout` field),
            i.e. net profit on win = payout - stake.
    """
    if payout <= 0:
        raise ValueError("payout must be > 0")
    net_profit_if_win = payout - stake
    breakeven_p = stake / payout
    ev = probability * net_profit_if_win - (1 - probability) * stake
    return EVResult(breakeven_probability=breakeven_p, expected_value=ev)
