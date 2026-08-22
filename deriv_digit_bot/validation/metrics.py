"""Backtest metrics - spec section 22 (trimmed to what's computable
without a full walk-forward harness)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TradeRecord:
    tick_index: int
    calibrated_probability: float
    stake: float
    payout: float
    won: bool
    pnl: float
    martingale_level: int


@dataclass
class BacktestMetrics:
    total_trades: int
    win_rate: float
    profit: float
    max_drawdown: float
    profit_factor: float
    longest_losing_streak: int
    martingale_activations: int
    max_martingale_level_used: int
    brier_score: float
    log_loss: float
    calibration_buckets: List[Tuple[str, int, float, float]]  # (range, n, predicted, actual)


def compute_metrics(trades: List[TradeRecord]) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics(0, float("nan"), 0.0, 0.0, float("nan"), 0, 0, 0, float("nan"), float("nan"), [])

    n = len(trades)
    wins = sum(1 for t in trades if t.won)
    win_rate = wins / n

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    running_losses = 0
    longest_losing_streak = 0
    gross_win = 0.0
    gross_loss = 0.0
    martingale_activations = sum(1 for t in trades if t.martingale_level > 0)
    max_level = max(t.martingale_level for t in trades)

    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        if t.pnl < 0:
            running_losses += 1
            longest_losing_streak = max(longest_losing_streak, running_losses)
            gross_loss += -t.pnl
        else:
            running_losses = 0
            gross_win += t.pnl

    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    briers = []
    logs = []
    for t in trades:
        p = t.calibrated_probability
        y = 1.0 if t.won else 0.0
        if math.isnan(p):
            continue
        briers.append((p - y) ** 2)
        p_clamped = min(max(p, 1e-6), 1 - 1e-6)
        logs.append(-(y * math.log(p_clamped) + (1 - y) * math.log(1 - p_clamped)))

    brier = sum(briers) / len(briers) if briers else float("nan")
    logloss = sum(logs) / len(logs) if logs else float("nan")

    edges = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bucket = [t for t in trades if lo <= t.calibrated_probability < hi or (i == len(edges) - 2 and t.calibrated_probability == hi)]
        if not in_bucket:
            continue
        pred_mean = sum(t.calibrated_probability for t in in_bucket) / len(in_bucket)
        actual = sum(1 for t in in_bucket if t.won) / len(in_bucket)
        buckets.append((f"{lo:.0%}-{hi:.0%}", len(in_bucket), pred_mean, actual))

    return BacktestMetrics(
        total_trades=n, win_rate=win_rate, profit=equity, max_drawdown=max_dd,
        profit_factor=profit_factor, longest_losing_streak=longest_losing_streak,
        martingale_activations=martingale_activations, max_martingale_level_used=max_level,
        brier_score=brier, log_loss=logloss, calibration_buckets=buckets,
    )
