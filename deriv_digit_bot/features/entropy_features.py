"""
Entropy / randomness features - spec section 11.

Entropy is used as a market-state / confidence feature ONLY - it does
not itself generate a trade signal (per spec: "Entropy should NOT
itself trigger trades").
"""
from __future__ import annotations

import math
from typing import List

MAX_ENTROPY_10 = math.log2(10)


def shannon_entropy(counts: List[int]) -> float:
    n = sum(counts)
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / n
        h -= p * math.log2(p)
    return h


def normalized_entropy(counts: List[int]) -> float:
    """0..1, where 1 = maximally uniform (max entropy for 10 symbols)."""
    return shannon_entropy(counts) / MAX_ENTROPY_10
