"""
Signal engine - spec section 16.

This is the ONE place a TRADE decision is produced. It requires every
gate to pass; a single failing gate forces NO_TRADE and records why.
Used identically by the historical simulator and (once wired) the
live loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config
from features.feature_engine import FeatureSnapshot
from models.base import ModelOutput
from models.ensemble import Ensemble, EnsembleResult
from models.calibration import Calibrator
from trading.expected_value import compute_ev
from trading.martingale import MartingaleState
from trading.risk_manager import RiskState
from trading.regime import classify as classify_regime


@dataclass
class Signal:
    action: str  # "TRADE" or "NO_TRADE"
    raw_probability: float
    calibrated_probability: float
    calibration_trusted: bool
    model_agreement: float  # 1 - dispersion, clipped to [0,1]
    dispersion: float
    expected_value: float
    breakeven_probability: float
    regime: str
    martingale_level: int
    stake: float
    model_probabilities: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


class SignalEngine:
    def __init__(self, config: Config, models: List, ensemble: Ensemble, calibrator: Calibrator):
        self.config = config
        self.models = models
        self.ensemble = ensemble
        self.calibrator = calibrator

    def evaluate(self, snap: FeatureSnapshot, martingale: MartingaleState,
                 risk: RiskState, payout: Optional[float]) -> Signal:
        cfg = self.config

        if not snap.ready:
            return self._no_trade(martingale, reason=f"Warming up ({snap.tick_index}/{cfg.min_history_size} ticks)")

        if not risk.can_trade():
            return self._no_trade(martingale, reason=f"Risk manager halted: {risk.halt_reason}")

        if martingale.in_cooldown:
            return self._no_trade(martingale, reason=f"Martingale cooldown ({martingale.cooldown_remaining} ticks left)")

        outputs: List[ModelOutput] = [m.predict(snap) for m in self.models]
        model_probs = {o.model_name: o.probability for o in outputs}

        ens: EnsembleResult = self.ensemble.combine(outputs)
        if ens.n_models_used < 2:
            return self._no_trade(martingale, reason="Fewer than 2 models produced a usable estimate", model_probs=model_probs)

        entropy_100 = snap.entropy_norm.get(100, 1.0)
        regime = classify_regime(ens.std_probability, ens.n_models_used, entropy_100, cfg.max_model_dispersion)
        agreement = max(0.0, 1.0 - ens.std_probability / max(cfg.max_model_dispersion, 1e-9))
        agreement = min(1.0, agreement)

        if not regime.tradeable:
            return self._no_trade(martingale, reason=f"Regime={regime.regime}: {regime.reason}",
                                   raw_p=ens.raw_probability, agreement=agreement, dispersion=ens.std_probability,
                                   regime=regime.regime, model_probs=model_probs)

        calibrated_p, trusted = self.calibrator.calibrate(ens.raw_probability)

        required_p = martingale.required_threshold()
        if calibrated_p < required_p:
            return self._no_trade(
                martingale, reason=f"Calibrated probability {calibrated_p:.3f} below required {required_p:.3f} for level {martingale.level}",
                raw_p=ens.raw_probability, calibrated_p=calibrated_p, calibration_trusted=trusted,
                agreement=agreement, dispersion=ens.std_probability, regime=regime.regime, model_probs=model_probs,
            )

        edge = calibrated_p - cfg.baseline_probability
        if edge < cfg.min_edge_over_baseline:
            return self._no_trade(
                martingale, reason=f"Edge over baseline {edge:.3f} below required {cfg.min_edge_over_baseline:.3f}",
                raw_p=ens.raw_probability, calibrated_p=calibrated_p, calibration_trusted=trusted,
                agreement=agreement, dispersion=ens.std_probability, regime=regime.regime, model_probs=model_probs,
            )

        if agreement < cfg.min_model_agreement:
            return self._no_trade(
                martingale, reason=f"Model agreement {agreement:.3f} below required {cfg.min_model_agreement:.3f}",
                raw_p=ens.raw_probability, calibrated_p=calibrated_p, calibration_trusted=trusted,
                agreement=agreement, dispersion=ens.std_probability, regime=regime.regime, model_probs=model_probs,
            )

        if not trusted:
            return self._no_trade(
                martingale, reason="Calibration bucket for this probability has insufficient samples - not trusted",
                raw_p=ens.raw_probability, calibrated_p=calibrated_p, calibration_trusted=trusted,
                agreement=agreement, dispersion=ens.std_probability, regime=regime.regime, model_probs=model_probs,
            )

        stake = martingale.current_stake()
        if payout is None or payout <= stake:
            return self._no_trade(
                martingale, reason="No valid payout quote available",
                raw_p=ens.raw_probability, calibrated_p=calibrated_p, calibration_trusted=trusted,
                agreement=agreement, dispersion=ens.std_probability, regime=regime.regime, model_probs=model_probs,
            )

        ev_result = compute_ev(calibrated_p, stake, payout)
        if ev_result.expected_value < cfg.min_expected_value + cfg.ev_safety_margin:
            return self._no_trade(
                martingale, reason=f"EV {ev_result.expected_value:.4f} below required margin",
                raw_p=ens.raw_probability, calibrated_p=calibrated_p, calibration_trusted=trusted,
                agreement=agreement, dispersion=ens.std_probability, regime=regime.regime,
                breakeven=ev_result.breakeven_probability, model_probs=model_probs,
            )

        return Signal(
            action="TRADE",
            raw_probability=ens.raw_probability,
            calibrated_probability=calibrated_p,
            calibration_trusted=trusted,
            model_agreement=agreement,
            dispersion=ens.std_probability,
            expected_value=ev_result.expected_value,
            breakeven_probability=ev_result.breakeven_probability,
            regime=regime.regime,
            martingale_level=martingale.level,
            stake=stake,
            model_probabilities=model_probs,
            reason="All gates passed",
        )

    def _no_trade(self, martingale: MartingaleState, reason: str, raw_p: float = float("nan"),
                  calibrated_p: float = float("nan"), calibration_trusted: bool = False,
                  agreement: float = float("nan"), dispersion: float = float("nan"),
                  regime: str = "UNKNOWN", breakeven: float = float("nan"),
                  model_probs: Optional[Dict[str, float]] = None) -> Signal:
        return Signal(
            action="NO_TRADE",
            raw_probability=raw_p,
            calibrated_probability=calibrated_p,
            calibration_trusted=calibration_trusted,
            model_agreement=agreement,
            dispersion=dispersion,
            expected_value=float("nan"),
            breakeven_probability=breakeven,
            regime=regime,
            martingale_level=martingale.level,
            stake=martingale.current_stake(),
            model_probabilities=model_probs or {},
            reason=reason,
        )
