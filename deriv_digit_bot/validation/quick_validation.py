"""
Quick validation - one chronological train/test split.

This exists because "forget walk-forward" cannot mean "forget
out-of-sample checking entirely" without the calibrated_probability
and metrics this bot reports being meaningless numbers. This is a
strict downgrade from spec section 21 (single split vs. rolling
re-fit) - it will NOT catch calibration drift over time, and a report
that looks good here is not a promise of live performance. It is the
minimum needed so "calibrated" means something at all today.

Split: first `train_fraction` of ticks are used ONLY to (a) let models
accumulate sample size and (b) fit the Calibrator's bucket mapping.
The remaining ticks are the test period: the calibrator is frozen,
and the full SignalEngine/Martingale/RiskManager pipeline is run
exactly as it would live.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from config import Config
from features.feature_engine import FeatureEngine
from models.ensemble import Ensemble, ModelPerformanceTracker
from models.calibration import Calibrator
from validation.historical_simulator import (
    build_models, collect_training_pairs, run_test_phase, digit_outcome_win, simulate_payout,
)
from validation.metrics import compute_metrics, BacktestMetrics, TradeRecord


@dataclass
class BaselineResult:
    name: str
    metrics: BacktestMetrics


@dataclass
class QuickValidationReport:
    n_train: int
    n_test: int
    strategy_metrics: BacktestMetrics
    strategy_no_trade_reasons: dict
    baselines: List[BaselineResult]
    calibration_buckets: list


def always_trade_baseline(digits: List[int], stake: float, payout_ratio: float) -> BacktestMetrics:
    """Baseline 1: always trade Over 2 on every tick, fixed stake, no martingale."""
    trades = []
    for t in range(len(digits) - 1):
        won = digit_outcome_win(digits[t + 1])
        payout = simulate_payout(stake, payout_ratio)
        pnl = (payout - stake) if won else -stake
        trades.append(TradeRecord(t, 0.70, stake, payout, won, pnl, 0))
    return compute_metrics(trades)


def run_quick_validation(config: Config, digits: List[int], payout_ratio: float = 1.32,
                          train_fraction: float = 0.6) -> QuickValidationReport:
    if len(digits) < config.min_history_size * 3:
        raise ValueError(
            f"Need at least {config.min_history_size * 3} digits for a meaningful train/test split "
            f"(got {len(digits)})."
        )

    split_idx = int(len(digits) * train_fraction)
    train_digits = digits[:split_idx]
    test_digits = digits[split_idx:]

    feature_engine = FeatureEngine(config.window_sizes, config.min_history_size)
    models = build_models()
    ensemble = Ensemble(ModelPerformanceTracker())
    calibrator = Calibrator(n_buckets=10)

    train_pairs = collect_training_pairs(config, feature_engine, models, ensemble, train_digits)
    calibrator.fit(train_pairs)

    result = run_test_phase(config, feature_engine, models, ensemble, calibrator, test_digits, payout_ratio)
    strategy_metrics = compute_metrics(result.trades)

    baselines = [
        BaselineResult("always_trade_over2", always_trade_baseline(test_digits, config.base_stake, payout_ratio)),
    ]

    return QuickValidationReport(
        n_train=len(train_digits), n_test=len(test_digits),
        strategy_metrics=strategy_metrics, strategy_no_trade_reasons=result.no_trade_reasons,
        baselines=baselines, calibration_buckets=[
            (f"{b.low:.0%}-{b.high:.0%}", b.n, b.actual_win_rate) for b in calibrator.report()
        ],
    )


def print_report(report: QuickValidationReport) -> None:
    print("=" * 70)
    print("QUICK VALIDATION REPORT (single train/test split - NOT walk-forward)")
    print("=" * 70)
    print(f"Train ticks: {report.n_train}   Test ticks: {report.n_test}")
    print()
    print("-- Calibration mapping fit on TRAIN, applied frozen on TEST --")
    for rng, n, actual in report.calibration_buckets:
        print(f"  bucket {rng:>9}  n={n:<6} actual_win_rate={actual:.3f}")
    print()
    print("-- Strategy (full model ensemble + gates + martingale) on TEST --")
    m = report.strategy_metrics
    print(f"  Trades taken:        {m.total_trades}")
    if m.total_trades:
        print(f"  Win rate:            {m.win_rate:.3f}")
        print(f"  Net P&L:             {m.profit:.2f}")
        print(f"  Max drawdown:        {m.max_drawdown:.2f}")
        print(f"  Profit factor:       {m.profit_factor:.3f}")
        print(f"  Longest loss streak: {m.longest_losing_streak}")
        print(f"  Martingale activations: {m.martingale_activations} (max level {m.max_martingale_level_used})")
        print(f"  Brier score:         {m.brier_score:.4f}  (lower is better; 0.21 = uninformative 70% guess)")
        print(f"  Log loss:            {m.log_loss:.4f}")
        print("  Calibration by bucket (predicted vs actual, on TEST trades only):")
        for rng, n, pred, actual in m.calibration_buckets:
            print(f"    {rng:>9}  n={n:<5} predicted={pred:.3f} actual={actual:.3f} diff={actual - pred:+.3f}")
    else:
        print("  (No trades passed all gates during the test period.)")
    print()
    print("-- Why the bot didn't trade (top reasons, test period) --")
    for reason, count in sorted(report.strategy_no_trade_reasons.items(), key=lambda x: -x[1])[:8]:
        print(f"  [{count:>6}] {reason}")
    print()
    print("-- Baseline comparisons (same test period, same synthetic payout) --")
    for b in report.baselines:
        bm = b.metrics
        print(f"  {b.name}: trades={bm.total_trades} win_rate={bm.win_rate:.3f} "
              f"P&L={bm.profit:.2f} max_dd={bm.max_drawdown:.2f}")
    print("=" * 70)
