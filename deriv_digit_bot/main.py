"""
Entry point.

    python main.py --data path/to/ticks.csv        # HISTORICAL_SIMULATION (default, safe)
    MODE=DERIV_DEMO CONFIRM_LIVE=true python main.py    # paper-money demo account
    MODE=LIVE CONFIRM_LIVE=true python main.py          # real money - do not use lightly

CSV format expected for --data: either a `digit` column (0-9), or a
`quote` column plus a `decimals` value passed via --decimals (pip
size for the symbol) so the last digit is extracted the same way
extract_last_digit() does it live.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from typing import List

from config import Config
from features.digit_extractor import extract_last_digit


def load_digits_from_csv(path: str, decimals: int) -> List[int]:
    digits = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_digit_col = "digit" in fieldnames
        has_quote_col = "quote" in fieldnames
        if not has_digit_col and not has_quote_col:
            raise ValueError(f"CSV must have a 'digit' or 'quote' column. Found: {fieldnames}")
        for row in reader:
            if has_digit_col and row.get("digit", "") != "":
                digits.append(int(row["digit"]))
            elif has_quote_col:
                digits.append(extract_last_digit(float(row["quote"]), decimals))
    return digits


def run_historical_simulation(cfg: Config, data_path: str, decimals: int,
                               train_fraction: float, payout_ratio: float) -> None:
    from validation.quick_validation import run_quick_validation, print_report

    print(f"Loading digits from {data_path} ...")
    digits = load_digits_from_csv(data_path, decimals)
    print(f"Loaded {len(digits)} digits.")

    report = run_quick_validation(cfg, digits, payout_ratio=payout_ratio, train_fraction=train_fraction)
    print_report(report)

    out_path = "quick_validation_report.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_train": report.n_train,
            "n_test": report.n_test,
            "trades": report.strategy_metrics.total_trades,
            "win_rate": report.strategy_metrics.win_rate,
            "profit": report.strategy_metrics.profit,
            "max_drawdown": report.strategy_metrics.max_drawdown,
            "brier_score": report.strategy_metrics.brier_score,
            "no_trade_reasons": report.strategy_no_trade_reasons,
            "calibration_buckets": report.calibration_buckets,
        }, f, indent=2, default=str)
    print(f"\nJSON summary written to {out_path}")


async def run_live(cfg: Config) -> None:
    """
    Live/demo loop. NOTE: this path has not been exercised against a
    real Deriv connection as part of this build - it is wired
    end-to-end using the same corrected schema and the same
    feature/model/signal/risk objects as the simulator, but you should
    run it on DERIV_DEMO and watch it closely before ever considering
    LIVE. It intentionally will not place a single trade until it has
    collected cfg.min_history_size ticks AND fit an initial
    calibration off the first chunk of live data (same train/test
    logic as the simulator, just running online).

    Handles SIGTERM (what Railway sends on redeploy/restart) by closing
    the websocket and sqlite connection cleanly rather than being killed
    mid-write.
    """
    from deriv.client import DerivClient
    from features.feature_engine import FeatureEngine
    from features.digit_extractor import extract_last_digit
    from models.ensemble import Ensemble, ModelPerformanceTracker
    from models.calibration import Calibrator
    from trading.martingale import MartingaleState
    from trading.risk_manager import RiskState
    from trading.signal_engine import SignalEngine
    from validation.historical_simulator import build_models
    from data.storage import Storage

    print(f"Starting in MODE={cfg.mode}. This places {'REAL' if cfg.mode == 'LIVE' else 'PAPER'} trades.")
    want_virtual = (cfg.mode == "DERIV_DEMO")  # LIVE requires a real-money account, DERIV_DEMO requires virtual
    client = DerivClient(cfg.deriv_app_id, cfg.deriv_token, want_virtual=want_virtual,
                          forced_account_id=cfg.deriv_account_id)
    await client.connect()

    active_symbols = await client.get_active_symbols()
    decimals = 2
    for s in active_symbols.get("active_symbols", []):
        if s.get("underlying_symbol") == cfg.symbol or s.get("symbol") == cfg.symbol:
            pip = s.get("pip_size") or s.get("pip")
            if pip is not None:
                decimals = len(str(pip).split(".")[-1]) if "." in str(pip) else 0
            break

    # One-time diagnostic: what does Deriv actually require to break even
    # on this contract right now, vs. the textbook 70%? This is the number
    # that should drive any threshold decision - not the theoretical baseline.
    diag_proposal = await client.get_proposal(cfg.contract_type, cfg.symbol, cfg.base_stake,
                                               cfg.duration, cfg.duration_unit)
    if diag_proposal and "proposal" in diag_proposal:
        from trading.expected_value import compute_ev
        p = diag_proposal["proposal"]
        real_payout, real_stake = float(p["payout"]), float(p["ask_price"])
        be = compute_ev(0.70, real_stake, real_payout).breakeven_probability
        print(f"Live breakeven check: stake={real_stake:.2f} payout={real_payout:.2f} "
              f"=> ACTUAL breakeven probability = {be:.4f} "
              f"(this is what a threshold has to clear, not the textbook 0.70).")
    else:
        print("Could not fetch a diagnostic proposal to compute real breakeven probability - "
              "proceeding without it, EV gate will still check per-trade.")


    feature_engine = FeatureEngine(cfg.window_sizes, cfg.min_history_size)
    models = build_models()
    ensemble = Ensemble(ModelPerformanceTracker())
    calibrator = Calibrator()
    signal_engine = SignalEngine(cfg, models, ensemble, calibrator)
    martingale = MartingaleState(cfg.base_stake, cfg.martingale_multiplier, cfg.max_martingale_steps,
                                  cfg.min_calibrated_probability)
    risk = RiskState(starting_balance=1000.0, balance=1000.0, max_daily_loss=cfg.max_daily_loss,
                      max_drawdown=cfg.max_drawdown, max_consecutive_losses=cfg.max_consecutive_losses)
    storage = Storage()

    shutdown_event = asyncio.Event()

    def _handle_shutdown(*_args):
        print("Shutdown signal received - will stop after the current tick.")
        shutdown_event.set()

    import signal
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, _handle_shutdown)
        except (NotImplementedError, RuntimeError):
            pass  # signal handlers unavailable on this platform - fall back to KeyboardInterrupt

    calibration_pairs = []
    calibration_target_n = cfg.min_history_size * 3
    last_raw_prediction = None  # (raw_probability, model_outputs) awaiting next digit's outcome

    tick_queue: asyncio.Queue = asyncio.Queue()

    async def on_tick(msg):
        tick = msg.get("tick")
        if tick:
            await tick_queue.put(tick)

    client.subscribe("tick", on_tick)
    await client.subscribe_ticks(cfg.symbol)

    print("Subscribed to ticks. Waiting for data...")
    try:
        while not shutdown_event.is_set():
            get_tick = asyncio.ensure_future(tick_queue.get())
            wait_shutdown = asyncio.ensure_future(shutdown_event.wait())
            done, pending = await asyncio.wait({get_tick, wait_shutdown}, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            if wait_shutdown in done:
                break
            tick = get_tick.result()
            quote = float(tick["quote"])
            digit = extract_last_digit(quote, decimals)
            storage.insert_tick(time.time(), cfg.symbol, quote, digit)

            snap = feature_engine.update(digit)
            martingale.tick_cooldown()

            outputs = [m.predict(snap) for m in models]

            if last_raw_prediction is not None:
                raw_p, prev_outputs = last_raw_prediction
                outcome = digit > 2
                for o in prev_outputs:
                    ensemble.tracker.record(o.model_name, o.probability, outcome)
                if not calibrator.fitted:
                    ens_check = ensemble.combine(prev_outputs)
                    if ens_check.n_models_used >= 2:
                        calibration_pairs.append((ens_check.raw_probability, outcome))
                    if len(calibration_pairs) >= calibration_target_n:
                        calibrator.fit(calibration_pairs)
                        print(f"Calibration fitted on {len(calibration_pairs)} observations. "
                              f"Trading gate now active (was collecting-only before this).")

            last_raw_prediction = (None, outputs)

            if not calibrator.fitted:
                continue  # still in calibration collection phase - never trades yet

            proposal = await client.get_proposal(cfg.contract_type, cfg.symbol, martingale.current_stake(),
                                                  cfg.duration, cfg.duration_unit)
            payout = None
            proposal_id = None
            price = None
            if proposal and "proposal" in proposal:
                payout = float(proposal["proposal"]["payout"])
                proposal_id = proposal["proposal"]["id"]
                price = float(proposal["proposal"]["ask_price"])

            signal = signal_engine.evaluate(snap, martingale, risk, payout)
            storage.insert_signal(time.time(), snap.tick_index, signal)

            if signal.action == "TRADE" and proposal_id is not None:
                print(f"TRADE: stake={signal.stake:.2f} calibrated_p={signal.calibrated_probability:.3f} "
                      f"EV={signal.expected_value:.4f} reason={signal.reason}")
                # Both DERIV_DEMO (paper money) and LIVE (real money) place the
                # order. cfg.validate() already refuses to reach this code path
                # at all unless CONFIRM_LIVE=true was explicitly set - that's
                # the actual safety gate, not withholding the buy() call.
                buy_result = await client.buy(proposal_id, price)
                contract_id = buy_result.get("buy", {}).get("contract_id") if buy_result else None
                print(f"  buy() sent (mode={cfg.mode}, contract_id={contract_id})")

                if contract_id is not None:
                    outcome_msg = await client.get_contract_status(contract_id)
                    poc = outcome_msg.get("proposal_open_contract", {}) if outcome_msg else {}
                    # 1-tick contracts settle almost immediately; poll briefly if not yet final.
                    attempts = 0
                    while not poc.get("is_sold") and attempts < 10:
                        await asyncio.sleep(1)
                        outcome_msg = await client.get_contract_status(contract_id)
                        poc = outcome_msg.get("proposal_open_contract", {}) if outcome_msg else {}
                        attempts += 1

                    if poc.get("is_sold"):
                        payout_received = float(poc.get("payout", 0.0))
                        won = payout_received > 0
                        pnl = (payout_received - signal.stake) if won else -signal.stake
                        risk.record_trade_result(pnl)
                        if won:
                            martingale.register_win()
                        else:
                            martingale.register_loss(cfg.cooldown_ticks)
                        storage.insert_trade(time.time(), snap.tick_index, signal.stake, payout_received,
                                              won, pnl, signal.martingale_level,
                                              signal.calibrated_probability, risk.balance)
                        print(f"  RESULT: {'WIN' if won else 'LOSS'} pnl={pnl:+.2f} balance={risk.balance:.2f}")
                    else:
                        print("  WARNING: could not confirm contract settlement after polling - "
                              "martingale/risk state NOT updated for this trade. Check manually.")
    finally:
        print("Closing Deriv connection and storage...")
        await client.close()
        storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deriv digit probability bot")
    parser.add_argument("--data", type=str, default=None, help="CSV of historical ticks/digits")
    parser.add_argument("--decimals", type=int, default=2, help="Pip decimals if CSV has 'quote' not 'digit'")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--payout-ratio", type=float, default=1.32,
                         help="Synthetic payout = stake * ratio, used ONLY in historical simulation")
    args = parser.parse_args()

    cfg = Config()
    try:
        cfg.validate()
    except (RuntimeError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    if cfg.mode == "HISTORICAL_SIMULATION":
        if not args.data:
            print("MODE=HISTORICAL_SIMULATION requires --data <csv path>. "
                  "See README.md for the expected format, or run the synthetic "
                  "smoke test in validation/quick_validation.py directly.", file=sys.stderr)
            sys.exit(1)
        run_historical_simulation(cfg, args.data, args.decimals, args.train_fraction, args.payout_ratio)
    elif cfg.mode in ("DERIV_DEMO", "LIVE"):
        try:
            asyncio.run(run_live(cfg))
        except KeyboardInterrupt:
            print("Shutdown requested - exiting cleanly.")
    elif cfg.mode == "PAPER":
        print("PAPER mode without a live tick feed is equivalent to HISTORICAL_SIMULATION on your own "
              "recorded ticks - pass --data. If you want live-tick paper trading, use MODE=DERIV_DEMO, "
              "which runs the identical live loop and simply avoids sending a real buy order.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Unknown MODE={cfg.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
