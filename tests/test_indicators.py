"""Indicator tests, including known-value checks against hand computation."""

import math
import unittest

from options_desk import indicators as ind


class TestMovingAverages(unittest.TestCase):
    def test_sma_known_values(self):
        result = ind.sma([1, 2, 3, 4, 5], 3)
        self.assertEqual(result[:2], [None, None])
        self.assertAlmostEqual(result[2], 2.0)
        self.assertAlmostEqual(result[3], 3.0)
        self.assertAlmostEqual(result[4], 4.0)

    def test_sma_rolling_sum_matches_naive(self):
        """The incremental window must not drift from a direct recomputation."""
        values = [math.sin(i / 3) * 100 + 500 for i in range(200)]
        fast = ind.sma(values, 20)
        for i in range(19, len(values)):
            naive = sum(values[i - 19 : i + 1]) / 20
            self.assertAlmostEqual(fast[i], naive, places=9)

    def test_sma_insufficient_data(self):
        self.assertEqual(ind.sma([1, 2], 5), [None, None])

    def test_ema_seeds_with_sma(self):
        values = [1, 2, 3, 4, 5, 6]
        result = ind.ema(values, 3)
        self.assertAlmostEqual(result[2], 2.0)  # SMA of 1,2,3
        self.assertAlmostEqual(result[3], 0.5 * 4 + 0.5 * 2.0)

    def test_ema_tracks_constant_series(self):
        result = ind.ema([7.0] * 30, 10)
        self.assertAlmostEqual(result[-1], 7.0)

    def test_period_validation(self):
        with self.assertRaises(ValueError):
            ind.sma([1, 2, 3], 0)


class TestRSI(unittest.TestCase):
    def test_all_gains_pins_at_100(self):
        result = ind.rsi(list(range(1, 40)), 14)
        self.assertAlmostEqual(result[-1], 100.0)

    def test_all_losses_pins_at_zero(self):
        result = ind.rsi(list(range(40, 1, -1)), 14)
        self.assertAlmostEqual(result[-1], 0.0)

    def test_flat_series_is_neutral(self):
        """No movement means no gain and no loss -- neither extreme is right."""
        result = ind.rsi([50.0] * 40, 14)
        self.assertAlmostEqual(result[-1], 50.0)

    def test_alignment_and_range(self):
        closes = [100 + math.sin(i / 4) * 10 for i in range(80)]
        result = ind.rsi(closes, 14)
        self.assertEqual(len(result), len(closes))
        self.assertTrue(all(v is None for v in result[:14]))
        self.assertTrue(all(0 <= v <= 100 for v in result[14:]))


class TestMACD(unittest.TestCase):
    def test_histogram_is_line_minus_signal(self):
        closes = [100 + math.sin(i / 5) * 8 for i in range(120)]
        data = ind.macd(closes)
        for m, s, h in zip(data["macd"], data["signal"], data["histogram"]):
            if m is not None and s is not None:
                self.assertAlmostEqual(h, m - s, places=9)

    def test_signal_aligned_to_series(self):
        closes = [float(i) for i in range(120)]
        data = ind.macd(closes)
        self.assertEqual(len(data["signal"]), len(closes))
        self.assertEqual(len(data["macd"]), len(closes))

    def test_rejects_inverted_periods(self):
        with self.assertRaises(ValueError):
            ind.macd([1.0] * 50, fast=26, slow=12)


class TestVolatility(unittest.TestCase):
    def test_true_range_uses_gap(self):
        """A gap down makes the prior close the relevant extreme, not today's high."""
        highs, lows, closes = [10, 8], [9, 7], [10, 7]
        tr = ind.true_range(highs, lows, closes)
        self.assertAlmostEqual(tr[1], 3.0)  # |10 - 7|, wider than today's 1.0 range

    def test_atr_positive_and_aligned(self):
        n = 60
        highs = [100 + i % 5 + 1 for i in range(n)]
        lows = [100 + i % 5 - 1 for i in range(n)]
        closes = [100 + i % 5 for i in range(n)]
        result = ind.atr(highs, lows, closes, 14)
        self.assertIsNotNone(result[-1])
        self.assertGreater(result[-1], 0)

    def test_realized_vol_zero_for_flat_series(self):
        result = ind.realized_volatility([100.0] * 60, 20)
        self.assertAlmostEqual(result[-1], 0.0)

    def test_realized_vol_scales_with_annualization(self):
        closes = [100 * (1.01 if i % 2 else 0.99) ** 1 for i in range(60)]
        daily = ind.realized_volatility(closes, 20, annualize=1)[-1]
        annual = ind.realized_volatility(closes, 20, annualize=252)[-1]
        self.assertAlmostEqual(annual, daily * math.sqrt(252), places=6)

    def test_realized_vol_rejects_non_positive_price(self):
        with self.assertRaises(ValueError):
            ind.realized_volatility([100.0, 0.0, 100.0] * 20, 20)

    def test_bollinger_bands_ordered(self):
        closes = [100 + math.sin(i / 3) * 5 for i in range(80)]
        bands = ind.bollinger(closes, 20)
        for lo, mid, hi in zip(bands["lower"], bands["middle"], bands["upper"]):
            if lo is not None:
                self.assertLessEqual(lo, mid)
                self.assertLessEqual(mid, hi)

    def test_bollinger_flat_series_has_zero_width(self):
        bands = ind.bollinger([50.0] * 40, 20)
        self.assertAlmostEqual(bands["bandwidth"][-1], 0.0)
        self.assertAlmostEqual(bands["percent_b"][-1], 0.5)

    def test_parkinson_positive(self):
        highs = [101 + i % 3 for i in range(60)]
        lows = [99 + i % 3 for i in range(60)]
        result = ind.parkinson_volatility(highs, lows, 20)
        self.assertGreater(result[-1], 0)


class TestTrend(unittest.TestCase):
    def test_adx_detects_strong_trend(self):
        """A clean one-way march should read as a strong trend."""
        n = 120
        closes = [100 + i for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        data = ind.adx(highs, lows, closes, 14)
        self.assertGreater(data["adx"][-1], 40)
        self.assertGreater(data["plus_di"][-1], data["minus_di"][-1])

    def test_adx_low_in_chop(self):
        n = 200
        closes = [100 + (2 if i % 2 else -2) for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        data = ind.adx(highs, lows, closes, 14)
        self.assertLess(data["adx"][-1], 25)

    def test_adx_aligned(self):
        n = 100
        closes = [100 + math.sin(i / 7) * 10 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        data = ind.adx(highs, lows, closes, 14)
        for key in ("adx", "plus_di", "minus_di"):
            self.assertEqual(len(data[key]), n, key)

    def test_regression_on_perfect_line(self):
        slope, intercept, r2 = ind.linear_regression([2 * i + 5 for i in range(50)])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 5.0)
        self.assertAlmostEqual(r2, 1.0)

    def test_regression_r2_low_for_noise(self):
        values = [100 + (7 * i) % 13 for i in range(60)]
        _, _, r2 = ind.linear_regression(values)
        self.assertLess(r2, 0.5)

    def test_regression_flat_series(self):
        slope, _, r2 = ind.linear_regression([5.0] * 20)
        self.assertAlmostEqual(slope, 0.0)
        self.assertAlmostEqual(r2, 1.0)

    def test_regression_needs_two_points(self):
        with self.assertRaises(ValueError):
            ind.linear_regression([1.0])

    def test_donchian_brackets_price(self):
        highs = [100 + i % 7 for i in range(60)]
        lows = [95 + i % 7 for i in range(60)]
        chan = ind.donchian(highs, lows, 20)
        self.assertGreaterEqual(chan["upper"][-1], chan["lower"][-1])


class TestVolume(unittest.TestCase):
    def test_obv_accumulates_on_up_closes(self):
        obv = ind.on_balance_volume([10, 11, 12], [100, 200, 300])
        self.assertAlmostEqual(obv[-1], 500.0)

    def test_obv_subtracts_on_down_closes(self):
        obv = ind.on_balance_volume([10, 9, 8], [100, 200, 300])
        self.assertAlmostEqual(obv[-1], -500.0)

    def test_obv_ignores_unchanged_closes(self):
        obv = ind.on_balance_volume([10, 10, 10], [100, 200, 300])
        self.assertAlmostEqual(obv[-1], 0.0)

    def test_volume_ratio(self):
        volumes = [100.0] * 20 + [200.0]
        result = ind.volume_ratio(volumes, 20)
        self.assertAlmostEqual(result[-1], 200.0 / 105.0, places=6)


class TestStatistics(unittest.TestCase):
    def test_percentile_rank_extremes(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertLess(ind.percentile_rank(values, 0.5), 10)
        self.assertGreater(ind.percentile_rank(values, 9.0), 90)

    def test_percentile_rank_midpoint(self):
        self.assertAlmostEqual(ind.percentile_rank([1.0, 2.0, 3.0], 2.0), 50.0)

    def test_percentile_rank_ignores_none(self):
        self.assertAlmostEqual(ind.percentile_rank([None, 1.0, 3.0], 3.0), 75.0)

    def test_percentile_rank_all_none(self):
        self.assertIsNone(ind.percentile_rank([None, None]))

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ind.true_range([1, 2], [1], [1, 2])


if __name__ == "__main__":
    unittest.main()
