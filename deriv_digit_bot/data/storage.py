"""SQLite persistence - spec section 28 (trimmed to sqlite3, no
external dependency needed)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    symbol TEXT NOT NULL,
    quote REAL NOT NULL,
    digit INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    tick_index INTEGER NOT NULL,
    action TEXT NOT NULL,
    raw_probability REAL,
    calibrated_probability REAL,
    model_agreement REAL,
    dispersion REAL,
    expected_value REAL,
    regime TEXT,
    martingale_level INTEGER,
    stake REAL,
    reason TEXT,
    model_probabilities_json TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    tick_index INTEGER NOT NULL,
    stake REAL NOT NULL,
    payout REAL NOT NULL,
    won INTEGER NOT NULL,
    pnl REAL NOT NULL,
    martingale_level INTEGER NOT NULL,
    calibrated_probability REAL,
    balance_after REAL
);
"""


class Storage:
    def __init__(self, path: str = "digit_bot.sqlite3"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def insert_tick(self, ts: float, symbol: str, quote: float, digit: int) -> None:
        self.conn.execute(
            "INSERT INTO ticks (ts, symbol, quote, digit) VALUES (?, ?, ?, ?)",
            (ts, symbol, quote, digit),
        )
        self.conn.commit()

    def insert_signal(self, ts: float, tick_index: int, signal) -> None:
        self.conn.execute(
            """INSERT INTO signals
               (ts, tick_index, action, raw_probability, calibrated_probability,
                model_agreement, dispersion, expected_value, regime,
                martingale_level, stake, reason, model_probabilities_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, tick_index, signal.action, signal.raw_probability, signal.calibrated_probability,
             signal.model_agreement, signal.dispersion, signal.expected_value, signal.regime,
             signal.martingale_level, signal.stake, signal.reason,
             json.dumps(signal.model_probabilities)),
        )
        self.conn.commit()

    def insert_trade(self, ts: float, tick_index: int, stake: float, payout: float,
                      won: bool, pnl: float, martingale_level: int,
                      calibrated_probability: float, balance_after: float) -> None:
        self.conn.execute(
            """INSERT INTO trades
               (ts, tick_index, stake, payout, won, pnl, martingale_level,
                calibrated_probability, balance_after)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, tick_index, stake, payout, int(won), pnl, martingale_level,
             calibrated_probability, balance_after),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
