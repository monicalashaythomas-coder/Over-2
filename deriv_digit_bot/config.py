"""
Central configuration for the digit bot.

Everything that should be tunable lives here and is sourced from the
environment (with .env support via python-dotenv if present). No
credentials or magic numbers are hard-coded into strategy modules -
they all read from a Config instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


@dataclass
class Config:
    # --- Deriv connection ---
    deriv_app_id: str = field(default_factory=lambda: _s("DERIV_APP_ID", "1089"))
    deriv_token: str = field(default_factory=lambda: _s("DERIV_TOKEN", ""))
    deriv_account_id: str = field(default_factory=lambda: _s("DERIV_ACCOUNT_ID", ""))

    # --- Contract spec ---
    symbol: str = field(default_factory=lambda: _s("SYMBOL", "R_100"))
    contract_type: str = field(default_factory=lambda: _s("CONTRACT_TYPE", "DIGITOVER"))
    barrier: str = field(default_factory=lambda: _s("BARRIER", "2"))
    duration: int = field(default_factory=lambda: _i("DURATION", 1))
    duration_unit: str = field(default_factory=lambda: _s("DURATION_UNIT", "t"))

    # --- Staking / martingale ---
    base_stake: float = field(default_factory=lambda: _f("BASE_STAKE", 0.35))
    martingale_multiplier: float = field(default_factory=lambda: _f("MARTINGALE_MULTIPLIER", 3.1))
    max_martingale_steps: int = field(default_factory=lambda: _i("MAX_MARTINGALE_STEPS", 3))

    # --- Data / warm-up ---
    min_history_size: int = field(default_factory=lambda: _i("MIN_HISTORY_SIZE", 300))
    window_sizes: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 250, 500, 1000])

    # --- Confidence gates per martingale level (index 0..max_steps) ---
    min_calibrated_probability: List[float] = field(default_factory=lambda: [
        _f("MIN_CALIBRATED_PROBABILITY_L0", 0.76),
        _f("MIN_CALIBRATED_PROBABILITY_L1", 0.80),
        _f("MIN_CALIBRATED_PROBABILITY_L2", 0.84),
        _f("MIN_CALIBRATED_PROBABILITY_L3", 0.88),
    ])

    baseline_probability: float = field(default_factory=lambda: _f("BASELINE_PROBABILITY", 0.70))
    min_edge_over_baseline: float = field(default_factory=lambda: _f("MIN_EDGE_OVER_BASELINE", 0.03))

    min_model_agreement: float = field(default_factory=lambda: _f("MIN_MODEL_AGREEMENT", 0.75))
    max_model_dispersion: float = field(default_factory=lambda: _f("MAX_MODEL_DISPERSION", 0.08))

    min_expected_value: float = field(default_factory=lambda: _f("MIN_EXPECTED_VALUE", 0.0))
    ev_safety_margin: float = field(default_factory=lambda: _f("EV_SAFETY_MARGIN", 0.01))

    # --- Risk controls ---
    max_daily_loss: float = field(default_factory=lambda: _f("MAX_DAILY_LOSS", 50))
    max_drawdown: float = field(default_factory=lambda: _f("MAX_DRAWDOWN", 75))
    max_consecutive_losses: int = field(default_factory=lambda: _i("MAX_CONSECUTIVE_LOSSES", 6))
    cooldown_ticks: int = field(default_factory=lambda: _i("COOLDOWN_TICKS", 100))

    # --- Bayesian prior ---
    bayesian_prior_strength: float = field(default_factory=lambda: _f("BAYESIAN_PRIOR_STRENGTH", 10.0))

    # --- Mode ---
    mode: str = field(default_factory=lambda: _s("MODE", "HISTORICAL_SIMULATION"))
    confirm_live: bool = field(default_factory=lambda: _b("CONFIRM_LIVE", False))

    def validate(self) -> None:
        thr = self.min_calibrated_probability
        if any(thr[i] > thr[i + 1] for i in range(len(thr) - 1)):
            raise ValueError("min_calibrated_probability thresholds must be non-decreasing by martingale level")
        if self.mode in ("DERIV_DEMO", "LIVE") and not self.confirm_live:
            raise RuntimeError(
                f"MODE={self.mode} requires CONFIRM_LIVE=true to be set explicitly. "
                "Refusing to start in a mode that places real trades without explicit confirmation."
            )
        if self.mode == "LIVE" and not self.deriv_token:
            raise RuntimeError("MODE=LIVE requires DERIV_TOKEN to be set.")
