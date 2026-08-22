"""
Extracts the last significant digit from a Deriv tick quote.

Deriv quotes are decimal strings with a fixed number of decimal places
per symbol (the "pip size"). The traded digit is the last digit of the
quote AT THAT PIP SIZE - not just str(quote)[-1], because floating
point repr can drop or add trailing zeros. We format explicitly to the
known decimal places before reading the last digit.
"""
from __future__ import annotations


def extract_last_digit(quote: float, decimals: int) -> int:
    """
    quote: the tick price, e.g. 1234.567
    decimals: number of decimal places for this symbol's pip size
              (Deriv reports this via active_symbols -> pip_size)
    """
    if decimals < 0:
        raise ValueError("decimals must be >= 0")
    formatted = f"{quote:.{decimals}f}"
    last_char = formatted[-1]
    if not last_char.isdigit():
        # decimals == 0 case still yields a digit; this branch guards
        # against unexpected formatting (e.g. negative zero edge cases)
        digits_only = [c for c in formatted if c.isdigit()]
        if not digits_only:
            raise ValueError(f"Could not extract digit from quote={quote!r} decimals={decimals}")
        last_char = digits_only[-1]
    return int(last_char)


def is_over_2(digit: int) -> bool:
    return digit > 2
