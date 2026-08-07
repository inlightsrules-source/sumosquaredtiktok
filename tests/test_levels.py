"""Tests for pivot detection, clustering, and the analysis pipeline."""

import math
import unittest
from datetime import date, timedelta

from options_desk import Series, analyze, from_rows, render
from options_desk.levels import (
    cluster_levels,
    fibonacci_retracements,
    find_pivots,
    pivot_points,
    round_number_levels,
)
from options_desk.ohlcv import Bar


def make_series(closes, symbol="TEST", volumes=None, spread=1.0):
    """Wrap a close path in plausible OHLC bars."""
    start = date(2024, 1, 1)
    bars = []
    for i, close in enumerate(closes):
        vol = volumes[i] if volumes else 1_000_000.0
        bars.append(
            Bar(
                day=start + timedelta(days=i),
                open=close,
                high=close + spread,
                low=close - spread,
                close=close,
                volume=vol,
            )
        )
    return Series(symbol, bars)


def oscillating(n=300, level=100.0, amplitude=10.0, wavelength=30.0):
    """A repeating wave -- it revisits the same highs and lows by construction."""
    return [level + amplitude * math.sin(2 * math.pi * i / wavelength) for i in range(n)]


class TestSeries(unittest.TestCase):
    def test_rejects_duplicate_dates(self):
        bar = Bar(date(2024, 1, 1), 10, 11, 9, 10)
        with self.assertRaises(ValueError):
            Series("X", [bar, bar])

    def test_sorts_unordered_bars(self):
        a = Bar(date(2024, 1, 2), 10, 11, 9, 10)
        b = Bar(date(2024, 1, 1), 10, 11, 9, 10)
        self.assertEqual(Series("X", [a, b]).bars[0].day, date(2024, 1, 1))

    def test_rejects_high_below_low(self):
        with self.assertRaises(ValueError):
            Bar(date(2024, 1, 1), 10, 8, 9, 10)

    def test_rejects_close_outside_range(self):
        with self.assertRaises(ValueError):
            Bar(date(2024, 1, 1), 10, 11, 9, 12)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            Series("X", [])

    def test_from_rows_parses_iso_dates(self):
        series = from_rows("X", [("2024-01-01", 10, 11, 9, 10, 100)])
        self.assertEqual(series.last.day, date(2024, 1, 1))
        self.assertEqual(series.last.volume, 100)

    def test_missing_volume_flagged(self):
        series = Series("X", [Bar(date(2024, 1, 1), 10, 11, 9, 10)])
        self.assertFalse(series.has_volume)

    def test_slice_returns_series(self):
        series = make_series(list(range(100, 150)))
        self.assertIsInstance(series[:10], Series)
        self.assertEqual(len(series[:10]), 10)


class TestPivots(unittest.TestCase):
    def test_finds_obvious_peak(self):
        closes = [10, 11, 12, 13, 20, 13, 12, 11, 10]
        pivots = find_pivots(make_series(closes), window=2)
        highs = [p for p in pivots if p.kind == "high"]
        self.assertTrue(any(p.index == 4 for p in highs))

    def test_finds_obvious_trough(self):
        closes = [20, 19, 18, 17, 10, 17, 18, 19, 20]
        pivots = find_pivots(make_series(closes), window=2)
        lows = [p for p in pivots if p.kind == "low"]
        self.assertTrue(any(p.index == 4 for p in lows))

    def test_ignores_unconfirmed_edges(self):
        """Bars too near either end have no future to confirm them."""
        series = make_series(oscillating(120))
        pivots = find_pivots(series, window=5)
        self.assertTrue(all(5 <= p.index < len(series) - 5 for p in pivots))

    def test_larger_window_is_more_selective(self):
        series = make_series(oscillating(300))
        self.assertGreaterEqual(
            len(find_pivots(series, window=3)), len(find_pivots(series, window=10))
        )

    def test_window_validation(self):
        with self.assertRaises(ValueError):
            find_pivots(make_series([10.0] * 20), window=0)


class TestClustering(unittest.TestCase):
    def test_repeated_wave_produces_levels(self):
        series = make_series(oscillating(300, level=100, amplitude=10))
        levels = cluster_levels(series, find_pivots(series, window=5))
        self.assertTrue(levels, "an oscillating series must produce levels")
        # Turning points sit near 90 and 110 by construction.
        prices = [lv.price for lv in levels]
        self.assertTrue(any(abs(p - 110) < 5 for p in prices), prices)
        self.assertTrue(any(abs(p - 90) < 5 for p in prices), prices)

    def test_strength_normalized_to_100(self):
        series = make_series(oscillating(300))
        levels = cluster_levels(series, find_pivots(series, window=5))
        self.assertAlmostEqual(max(lv.strength for lv in levels), 100.0)
        self.assertTrue(all(0 <= lv.strength <= 100 for lv in levels))

    def test_min_touches_filters_singletons(self):
        series = make_series(oscillating(300))
        pivots = find_pivots(series, window=5)
        strict = cluster_levels(series, pivots, min_touches=4)
        loose = cluster_levels(series, pivots, min_touches=1)
        self.assertLessEqual(len(strict), len(loose))
        self.assertTrue(all(lv.touches >= 4 for lv in strict))

    def test_empty_pivots_yields_no_levels(self):
        self.assertEqual(cluster_levels(make_series([100.0] * 60), []), [])

    def test_recency_weighting_favours_recent_touches(self):
        """Two identical levels: the more recent one must score higher."""
        # Touch 95 early, then 105 late, with equal touch counts.
        closes = ([100, 95, 100] * 20) + ([100, 105, 100] * 20)
        series = make_series(closes, spread=0.5)
        levels = cluster_levels(series, find_pivots(series, window=1), half_life_bars=30)
        recent = [lv for lv in levels if lv.price > 102]
        old = [lv for lv in levels if lv.price < 98]
        if recent and old:
            self.assertGreater(max(l.strength for l in recent), max(l.strength for l in old))

    def test_classify_by_role(self):
        series = make_series(oscillating(300))
        analysis = analyze(series)
        for level in analysis.levels:
            if level.role == "resistance":
                self.assertGreater(level.price, analysis.price)
            elif level.role == "support":
                self.assertLess(level.price, analysis.price)

    def test_tolerance_validation(self):
        series = make_series(oscillating(120))
        with self.assertRaises(ValueError):
            cluster_levels(series, find_pivots(series), tolerance_atr=0)


class TestDerivedLevels(unittest.TestCase):
    def test_pivot_points_ordered(self):
        series = make_series([100.0] * 30)
        pts = pivot_points(series)
        self.assertLess(pts["s1"], pts["pivot"])
        self.assertLess(pts["pivot"], pts["r1"])
        self.assertLess(pts["r1"], pts["r2"])

    def test_fibonacci_method_accepted(self):
        series = make_series([100.0] * 30)
        pts = pivot_points(series, method="fibonacci")
        self.assertIn("r3", pts)

    def test_unknown_pivot_method(self):
        with self.assertRaises(ValueError):
            pivot_points(make_series([100.0] * 30), method="camarilla")

    def test_fibonacci_levels_inside_swing(self):
        closes = list(range(100, 200))
        fib = fibonacci_retracements(make_series(closes))
        self.assertEqual(fib["direction"], "up")
        for key, value in fib.items():
            if key.startswith("fib_"):
                self.assertGreaterEqual(value, fib["swing_low"] - 1)
                self.assertLessEqual(value, fib["swing_high"] + 1)

    def test_fibonacci_detects_downswing(self):
        fib = fibonacci_retracements(make_series(list(range(200, 100, -1))))
        self.assertEqual(fib["direction"], "down")

    def test_round_numbers_bracket_price(self):
        levels = round_number_levels(187.4)
        self.assertTrue(any(l < 187.4 for l in levels))
        self.assertTrue(any(l > 187.4 for l in levels))
        self.assertTrue(all(l > 0 for l in levels))

    def test_round_numbers_reject_bad_price(self):
        with self.assertRaises(ValueError):
            round_number_levels(0)


class TestAnalysis(unittest.TestCase):
    def test_uptrend_detected(self):
        series = make_series([100 + i * 0.5 for i in range(200)])
        analysis = analyze(series)
        self.assertEqual(analysis.trend.direction, "up")
        self.assertGreater(analysis.trend.quality, 0.9)

    def test_downtrend_detected(self):
        series = make_series([200 - i * 0.5 for i in range(200)])
        self.assertEqual(analyze(series).trend.direction, "down")

    def test_sideways_detected(self):
        series = make_series(oscillating(300, amplitude=5))
        self.assertEqual(analyze(series).trend.direction, "sideways")

    def test_requires_minimum_history(self):
        with self.assertRaises(ValueError):
            analyze(make_series([100.0] * 30))

    def test_every_reading_has_an_interpretation(self):
        analysis = analyze(make_series(oscillating(300)))
        groups = (
            analysis.trend.readings
            + analysis.momentum
            + analysis.volatility
            + analysis.participation
        )
        self.assertTrue(groups)
        for reading in groups:
            self.assertTrue(reading.interpretation.strip(), reading.name)
            self.assertTrue(reading.label.strip(), reading.name)

    def test_handles_missing_volume(self):
        start = date(2024, 1, 1)
        bars = [
            Bar(start + timedelta(days=i), 100 + i, 101 + i, 99 + i, 100 + i)
            for i in range(200)
        ]
        analysis = analyze(Series("NOVOL", bars))
        self.assertTrue(any(r.label == "unavailable" for r in analysis.participation))

    def test_regime_is_populated(self):
        analysis = analyze(make_series(oscillating(300)))
        self.assertTrue(analysis.regime)
        self.assertTrue(analysis.regime_note)

    def test_render_produces_report(self):
        text = render(analyze(make_series(oscillating(300))))
        for heading in ("TREND", "MOMENTUM", "VOLATILITY", "SUPPORT & RESISTANCE"):
            self.assertIn(heading, text)

    def test_render_survives_trending_series(self):
        text = render(analyze(make_series([100 + i * 0.4 for i in range(250)])))
        self.assertIn("REGIME", text)


class TestReadoutCoherence(unittest.TestCase):
    """Guards against the readout contradicting or garbling itself."""

    def _drifting_series(self):
        """Upward drift with repeated stalls -- direction up, ADX weak."""
        closes = []
        level = 100.0
        for i in range(300):
            level += 0.25
            closes.append(level + 6 * math.sin(2 * math.pi * i / 18))
        return make_series(closes)

    def test_no_glued_strength_and_quality_labels(self):
        """'Absent well-defined uptrend' is not English -- it must not appear."""
        for series in (self._drifting_series(), make_series(oscillating(300))):
            summary = analyze(series).trend.summary
            for bad in ("Absent well-defined", "Weak well-defined", "Absent choppy"):
                self.assertNotIn(bad, summary, summary)

    def test_regime_does_not_contradict_trend_direction(self):
        """A regime of 'range-bound' alongside a directional trend is incoherent."""
        analysis = analyze(self._drifting_series())
        if analysis.trend.direction != "sideways":
            self.assertNotIn("range-bound", analysis.regime)
            self.assertNotIn("quiet range", analysis.regime)

    def test_drifting_regime_named_when_strength_absent(self):
        analysis = analyze(self._drifting_series())
        if analysis.trend.direction != "sideways" and analysis.trend.strength in (
            "absent",
            "weak",
        ):
            self.assertIn("drifting", analysis.regime)

    def test_summary_non_empty_for_every_combination(self):
        for series in (
            make_series([100 + i * 0.5 for i in range(200)]),
            make_series([200 - i * 0.5 for i in range(200)]),
            make_series(oscillating(300)),
            self._drifting_series(),
        ):
            trend = analyze(series).trend
            self.assertTrue(trend.summary.strip())
            self.assertTrue(trend.summary.endswith("."), trend.summary)


class TestStrikeReference(unittest.TestCase):
    def test_prefers_strong_level_over_merely_nearest(self):
        """A weak level just below should lose to a strong one slightly further."""
        from options_desk.levels import Level, Pivot, most_significant

        def level(price, strength, role):
            lv = Level(price=price, low=price - 1, high=price + 1, strength=strength)
            lv.role = role
            lv.pivots = [Pivot(0, date(2024, 1, 1), price, "low")]
            return lv

        weak_near = level(98.0, 15.0, "support")
        strong_far = level(92.0, 100.0, "support")
        chosen = most_significant([weak_near, strong_far], 100.0, "support")
        self.assertIs(chosen, strong_far)

    def test_distance_discount_rejects_far_monster(self):
        from options_desk.levels import Level, Pivot, most_significant

        def level(price, strength):
            lv = Level(price=price, low=price - 1, high=price + 1, strength=strength)
            lv.role = "support"
            lv.pivots = [Pivot(0, date(2024, 1, 1), price, "low")]
            return lv

        near = level(99.0, 80.0)
        far = level(55.0, 100.0)
        self.assertIs(most_significant([near, far], 100.0, "support"), near)

    def test_falls_back_to_nearest_when_all_out_of_range(self):
        from options_desk.levels import Level, Pivot, most_significant

        lv = Level(price=50.0, low=49, high=51, strength=90.0)
        lv.role = "support"
        lv.pivots = [Pivot(0, date(2024, 1, 1), 50.0, "low")]
        self.assertIs(most_significant([lv], 100.0, "support", 10.0), lv)

    def test_returns_none_when_no_levels_of_role(self):
        from options_desk.levels import most_significant

        self.assertIsNone(most_significant([], 100.0, "support"))

    def test_rejects_bad_distance(self):
        from options_desk.levels import most_significant

        with self.assertRaises(ValueError):
            most_significant([], 100.0, "support", max_distance_pct=0)

    def test_analysis_populates_key_levels(self):
        analysis = analyze(make_series(oscillating(300)))
        for level in (analysis.key_support, analysis.key_resistance):
            if level is not None:
                self.assertIn(level, analysis.levels)


if __name__ == "__main__":
    unittest.main()
