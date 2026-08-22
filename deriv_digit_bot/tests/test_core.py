import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.digit_extractor import extract_last_digit, is_over_2
from features.rolling_windows import RollingWindow, MultiScaleWindows
from features.digit_statistics import wilson_interval, digit_z_score
from features.feature_engine import FeatureEngine
from models.calibration import Calibrator
from trading.expected_value import compute_ev
from trading.martingale import MartingaleState
from trading.risk_manager import RiskState
from config import Config


class TestDigitExtraction(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(extract_last_digit(1234.567, 3), 7)
        self.assertEqual(extract_last_digit(1234.560, 3), 0)
        self.assertEqual(extract_last_digit(100.0, 2), 0)

    def test_over2(self):
        self.assertTrue(is_over_2(3))
        self.assertFalse(is_over_2(2))
        self.assertFalse(is_over_2(0))


class TestRollingWindow(unittest.TestCase):
    def test_incremental_counts_match_bruteforce(self):
        w = RollingWindow(5)
        seq = [1, 2, 3, 4, 5, 6, 7, 8]
        for d in seq:
            w.push(d)
        expected_tail = seq[-5:]
        bruteforce_counts = [0] * 10
        for d in expected_tail:
            bruteforce_counts[d] += 1
        self.assertEqual(w.counts, bruteforce_counts)
        self.assertEqual(list(w.buf), expected_tail)

    def test_multiscale_independent(self):
        ms = MultiScaleWindows([2, 4])
        for d in [9, 9, 9, 0]:
            ms.push(d)
        self.assertEqual(ms.get(2).counts[9], 1)  # last 2 = [9,0]
        self.assertEqual(ms.get(2).counts[0], 1)
        self.assertEqual(ms.get(4).counts[9], 3)


class TestStatistics(unittest.TestCase):
    def test_wilson_interval_bounds(self):
        low, high = wilson_interval(70, 100)
        self.assertTrue(0 <= low < 0.70 < high <= 1)

    def test_wilson_interval_empty(self):
        low, high = wilson_interval(0, 0)
        self.assertEqual((low, high), (0.0, 1.0))

    def test_z_score_uniform_is_zero_ish(self):
        z = digit_z_score(10, 100)
        self.assertAlmostEqual(z, 0.0, places=4)


class TestFeatureEngineNoLeakage(unittest.TestCase):
    def test_snapshot_only_reflects_seen_digits(self):
        fe = FeatureEngine(window_sizes=[5, 10], min_history_size=5)
        digits = [3, 3, 3, 3, 3]
        snaps = [fe.update(d) for d in digits]
        # after 5 identical "win" digits, window(5) should show n=5, p_over2=1.0
        self.assertEqual(snaps[-1].distributions[5].n, 5)
        self.assertEqual(snaps[-1].distributions[5].p_over2, 1.0)
        self.assertTrue(snaps[-1].ready)
        self.assertFalse(snaps[0].ready)


class TestCalibration(unittest.TestCase):
    def test_fit_and_apply(self):
        cal = Calibrator(n_buckets=5)
        pairs = []
        # construct data where raw probability 0.9 actually only wins 60% of the time
        # (i.e. simulate an overconfident model) and check calibration corrects it
        for _ in range(200):
            pairs.append((0.90, False))
        for _ in range(300):
            pairs.append((0.90, True))
        cal.fit(pairs)
        calibrated, trusted = cal.calibrate(0.90)
        self.assertTrue(trusted)
        self.assertAlmostEqual(calibrated, 0.6, delta=0.05)

    def test_untrusted_when_insufficient_samples(self):
        cal = Calibrator(n_buckets=10)
        cal.fit([(0.9, True)] * 3)  # below MIN_BUCKET_SAMPLES total
        self.assertFalse(cal.fitted)
        p, trusted = cal.calibrate(0.9)
        self.assertFalse(trusted)
        self.assertEqual(p, 0.9)  # falls back to raw


class TestExpectedValue(unittest.TestCase):
    def test_breakeven(self):
        # payout=1.4x stake => breakeven prob = stake/payout
        result = compute_ev(probability=0.75, stake=1.0, payout=1.4)
        self.assertAlmostEqual(result.breakeven_probability, 1 / 1.4, places=6)
        expected_ev = 0.75 * 0.4 - 0.25 * 1.0
        self.assertAlmostEqual(result.expected_value, expected_ev, places=6)

    def test_rejects_bad_payout(self):
        with self.assertRaises(ValueError):
            compute_ev(0.8, 1.0, 0.0)


class TestMartingaleNoAutoContinue(unittest.TestCase):
    def test_thresholds_nondecreasing_enforced_by_config(self):
        cfg = Config()
        cfg.validate()  # should not raise for defaults

    def test_stake_progression(self):
        m = MartingaleState(base_stake=0.35, multiplier=3.1, max_steps=3,
                             thresholds=[0.76, 0.80, 0.84, 0.88])
        self.assertAlmostEqual(m.current_stake(), 0.35)
        m.register_loss(cooldown_ticks=10)
        self.assertEqual(m.level, 1)
        self.assertAlmostEqual(m.current_stake(), round(0.35 * 3.1, 2))
        m.register_win()
        self.assertEqual(m.level, 0)
        self.assertAlmostEqual(m.current_stake(), 0.35)

    def test_cooldown_after_max_steps_exhausted(self):
        m = MartingaleState(base_stake=0.35, multiplier=3.1, max_steps=1,
                             thresholds=[0.76, 0.80])
        m.register_loss(cooldown_ticks=5)  # level 0 -> 1
        self.assertEqual(m.level, 1)
        self.assertFalse(m.in_cooldown)
        m.register_loss(cooldown_ticks=5)  # exceeds max_steps -> reset + cooldown
        self.assertEqual(m.level, 0)
        self.assertTrue(m.in_cooldown)
        self.assertEqual(m.cooldown_remaining, 5)

    def test_martingale_never_decides_to_trade_itself(self):
        # MartingaleState only exposes thresholds/stake; it has no method
        # that returns a trade decision. This test asserts that contract.
        m = MartingaleState(base_stake=1.0, multiplier=2.0, max_steps=2, thresholds=[0.7, 0.8, 0.9])
        forbidden_methods = ("should_trade", "decide", "auto_trade", "place_trade")
        for name in forbidden_methods:
            self.assertFalse(hasattr(m, name))


class TestRiskManager(unittest.TestCase):
    def test_halts_on_daily_loss(self):
        r = RiskState(starting_balance=100, balance=100, max_daily_loss=10,
                       max_drawdown=1000, max_consecutive_losses=1000)
        r.record_trade_result(-5)
        self.assertTrue(r.can_trade())
        r.record_trade_result(-6)
        self.assertFalse(r.can_trade())
        self.assertIn("daily loss", r.halt_reason.lower())

    def test_halts_on_consecutive_losses(self):
        r = RiskState(starting_balance=100, balance=100, max_daily_loss=1000,
                       max_drawdown=1000, max_consecutive_losses=3)
        for _ in range(3):
            r.record_trade_result(-1)
        self.assertFalse(r.can_trade())

    def test_manual_kill(self):
        r = RiskState(starting_balance=100, balance=100)
        self.assertTrue(r.can_trade())
        r.manual_kill("stop now")
        self.assertFalse(r.can_trade())


class TestConfigSafety(unittest.TestCase):
    def test_live_mode_requires_confirm(self):
        cfg = Config()
        cfg.mode = "LIVE"
        cfg.confirm_live = False
        cfg.deriv_token = "x"
        with self.assertRaises(RuntimeError):
            cfg.validate()

    def test_live_mode_requires_token(self):
        cfg = Config()
        cfg.mode = "LIVE"
        cfg.confirm_live = True
        cfg.deriv_token = ""
        with self.assertRaises(RuntimeError):
            cfg.validate()

    def test_historical_simulation_never_requires_confirm(self):
        cfg = Config()
        cfg.mode = "HISTORICAL_SIMULATION"
        cfg.confirm_live = False
        cfg.validate()  # should not raise


class TestStatisticalSignificance(unittest.TestCase):
    def test_chi_square_uniform_data_mostly_not_significant(self):
        from validation.statistical_tests import chi_square_uniform_test
        import random
        random.seed(1)
        false_positives = 0
        trials = 200
        for _ in range(trials):
            counts = [0] * 10
            for _ in range(300):
                counts[random.randint(0, 9)] += 1
            result = chi_square_uniform_test(counts, alpha=0.05)
            if result.reject_uniform:
                false_positives += 1
        # Expect roughly 5% false positive rate on truly uniform data;
        # allow generous slack since this is a stochastic test.
        self.assertLess(false_positives / trials, 0.15)

    def test_chi_square_detects_real_skew(self):
        from validation.statistical_tests import chi_square_uniform_test
        counts = [50, 50, 50, 5, 5, 5, 5, 5, 5, 5]  # heavily skewed toward 0-2
        result = chi_square_uniform_test(counts, alpha=0.05)
        self.assertTrue(result.reject_uniform)

    def test_z_test_over2_baseline(self):
        from validation.statistical_tests import z_test_over2
        # exactly at baseline -> should not reject
        result = z_test_over2(count_over2=700, n=1000, p0=0.70)
        self.assertFalse(result.reject_baseline)
        # way off baseline -> should reject
        result2 = z_test_over2(count_over2=900, n=1000, p0=0.70)
        self.assertTrue(result2.reject_baseline)

    def test_significance_model_abstains_on_uniform_data(self):
        from models.significance_model import SignificanceModel
        from features.feature_engine import FeatureEngine
        import random
        random.seed(2)
        fe = FeatureEngine(window_sizes=[250], min_history_size=250)
        model = SignificanceModel(window_size=250)
        abstentions = 0
        n_checks = 100
        for _ in range(300 + n_checks):
            snap = fe.update(random.randint(0, 9))
        # after warmup, sample a run of checks - most should abstain
        for _ in range(n_checks):
            snap = fe.update(random.randint(0, 9))
            out = model.predict(snap)
            if out.abstains():
                abstentions += 1
        self.assertGreater(abstentions / n_checks, 0.5)


class TestMarkovCascade(unittest.TestCase):
    def test_falls_back_through_orders(self):
        from models.markov_model import MarkovModel
        from features.feature_engine import FeatureSnapshot
        model = MarkovModel()
        # no data at any order -> abstain
        snap = FeatureSnapshot(
            tick_index=0, last_digit=None, distributions={},
            p_win_given_last_digit=float("nan"), p_win_given_last_digit_n=0,
            p_win_given_last_two=float("nan"), p_win_given_last_two_n=0,
            p_win_given_last_three=float("nan"), p_win_given_last_three_n=0,
            run_p_win=float("nan"), run_sample_size=0, run_type=None, run_length=0,
            p_win_seq2=float("nan"), p_win_seq2_n=0, p_win_seq3=float("nan"), p_win_seq3_n=0,
            entropy_norm={}, ready=True,
        )
        out = model.predict(snap)
        self.assertTrue(out.abstains())

        # only order-1 has enough samples
        snap.p_win_given_last_digit = 0.72
        snap.p_win_given_last_digit_n = 50
        out = model.predict(snap)
        self.assertEqual(out.model_name, "markov_order1")

        # order-2 now has enough samples too - should prefer order 2
        snap.p_win_given_last_two = 0.74
        snap.p_win_given_last_two_n = 40
        out = model.predict(snap)
        self.assertEqual(out.model_name, "markov_order2")

        # order-3 now has enough samples - should prefer order 3
        snap.p_win_given_last_three = 0.76
        snap.p_win_given_last_three_n = 90
        out = model.predict(snap)
        self.assertEqual(out.model_name, "markov_order3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
