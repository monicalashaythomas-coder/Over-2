"""
Historical simulator - spec section 20.

Sequential, tick-by-tick replay. At tick t, only information available
through digit[t] is used to decide whether to trade on digit[t+1] -
there is no lookahead. Uses the exact same FeatureEngine, model
objects, Ensemble, SignalEngine, MartingaleState and RiskState classes
as would be used live; no parallel "backtest-only" strategy logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import Config
from features.feature_engine import FeatureEngine, FeatureSnapshot
from models.ensemble import Ensemble
from models.calibration import Calibrator
from trading.martingale import MartingaleState
from trading.risk_manager import RiskState
from trading.signal_engine import SignalEngine, Signal
from validation.metrics import TradeRecord


def digit_outcome_win(digit: int) -> bool:
    return digit > 2


def simulate_payout(stake: float, payout_ratio: float) -> float:
    """
    Approximates Deriv's actual DIGITOVER/2 payout as stake * ratio.
    THIS IS A STAND-IN. Live/demo trading must use the real `proposal`
    response's payout field - Deriv's actual pricing depends on
    current market conditions and is not a fixed ratio. Default ratio
    below is a conservative approximation reflecting typical
    house-edge-adjusted pricing for a ~70%-probability digit contract.
    """
    return round(stake * payout_ratio, 4)


@dataclass
class SimulationResult:
    trades: List[TradeRecord]
    signals_evaluated: int
    no_trade_reasons: dict


def build_models():
    from models.empirical_model import EmpiricalModel
    from models.bayesian_model import BayesianModel
    from models.markov_model import MarkovModel
    from models.run_length_model import RunLengthModel
    from models.pattern_model import PatternModel
    from models.significance_model import SignificanceModel
    from models.digit_sequence_model import DigitSequenceModel
    return [
        EmpiricalModel(window_size=100),
        BayesianModel(baseline_p=0.70, prior_strength=10.0, window_size=250),
        MarkovModel(),
        RunLengthModel(),
        PatternModel(),
        SignificanceModel(window_size=250, alpha=0.05, baseline_p=0.70),
        DigitSequenceModel(),
    ]


def collect_training_pairs(config: Config, feature_engine: FeatureEngine, models, ensemble: Ensemble,
                            digits: List[int]) -> List[Tuple[float, bool]]:
    """Pure observation pass: advance shared feature_engine/model-tracker
    state and record (raw_probability, outcome) pairs for calibration
    fitting. No trades, no risk/martingale state touched."""
    pairs = []
    for t in range(len(digits) - 1):
        snap = feature_engine.update(digits[t])
        if not snap.ready:
            continue
        outputs = [m.predict(snap) for m in models]
        ens = ensemble.combine(outputs)
        outcome = digit_outcome_win(digits[t + 1])
        for o in outputs:
            ensemble.tracker.record(o.model_name, o.probability, outcome)
        if ens.n_models_used >= 2:
            pairs.append((ens.raw_probability, outcome))
    return pairs


def run_test_phase(config: Config, feature_engine: FeatureEngine, models, ensemble: Ensemble,
                    calibrator: Calibrator, digits: List[int], payout_ratio: float) -> SimulationResult:
    signal_engine = SignalEngine(config, models, ensemble, calibrator)
    martingale = MartingaleState(
        base_stake=config.base_stake, multiplier=config.martingale_multiplier,
        max_steps=config.max_martingale_steps, thresholds=config.min_calibrated_probability,
    )
    risk = RiskState(
        starting_balance=1000.0, balance=1000.0,
        max_daily_loss=config.max_daily_loss, max_drawdown=config.max_drawdown,
        max_consecutive_losses=config.max_consecutive_losses,
    )

    trades: List[TradeRecord] = []
    reasons: dict = {}
    signals_evaluated = 0

    for t in range(len(digits) - 1):
        snap = feature_engine.update(digits[t])
        martingale.tick_cooldown()

        stake_for_payout = martingale.current_stake()
        payout = simulate_payout(stake_for_payout, payout_ratio)

        signal: Signal = signal_engine.evaluate(snap, martingale, risk, payout)
        signals_evaluated += 1
        reasons[signal.reason] = reasons.get(signal.reason, 0) + 1

        if signal.action == "TRADE":
            outcome_win = digit_outcome_win(digits[t + 1])
            pnl = (payout - signal.stake) if outcome_win else (-signal.stake)
            trades.append(TradeRecord(
                tick_index=snap.tick_index, calibrated_probability=signal.calibrated_probability,
                stake=signal.stake, payout=payout, won=outcome_win, pnl=pnl,
                martingale_level=signal.martingale_level,
            ))
            risk.record_trade_result(pnl)
            for name, p in signal.model_probabilities.items():
                ensemble.tracker.record(name, p, outcome_win)
            if outcome_win:
                martingale.register_win()
            else:
                martingale.register_loss(config.cooldown_ticks)

    return SimulationResult(trades=trades, signals_evaluated=signals_evaluated, no_trade_reasons=reasons)
