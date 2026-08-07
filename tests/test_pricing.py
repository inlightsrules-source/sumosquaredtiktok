"""Pricing, Greeks, and probability tests.

Verified against identities that hold independently of the implementation --
put-call parity, delta bounds, gamma symmetry, and round-tripping implied
volatility -- rather than against numbers this code produced.
"""

import math
import unittest

from options_desk.pricing import (
    CALL,
    PUT,
    Option,
    delta_to_strike,
    expected_move,
    greeks,
    implied_volatility,
    norm_cdf,
    price,
    probabilities,
    probability_of_touch,
)


def opt(**kw):
    defaults = dict(spot=100.0, strike=100.0, dte=30.0, volatility=0.25, rate=0.04)
    defaults.update(kw)
    return Option(**defaults)


class TestValidation(unittest.TestCase):
    def test_rejects_non_positive_spot(self):
        with self.assertRaises(ValueError):
            opt(spot=0)

    def test_rejects_non_positive_strike(self):
        with self.assertRaises(ValueError):
            opt(strike=-5)

    def test_rejects_negative_dte(self):
        with self.assertRaises(ValueError):
            opt(dte=-1)

    def test_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            opt(kind="straddle")

    def test_rejects_negative_volatility(self):
        with self.assertRaises(ValueError):
            opt(volatility=-0.1)


class TestNormal(unittest.TestCase):
    def test_cdf_at_zero(self):
        self.assertAlmostEqual(norm_cdf(0.0), 0.5, places=12)

    def test_cdf_symmetry(self):
        for x in (0.3, 1.0, 2.5):
            self.assertAlmostEqual(norm_cdf(x) + norm_cdf(-x), 1.0, places=12)

    def test_cdf_known_value(self):
        # 1.96 is the canonical 97.5% point.
        self.assertAlmostEqual(norm_cdf(1.959963985), 0.975, places=6)


class TestPricing(unittest.TestCase):
    def test_put_call_parity(self):
        """C - P = S*e^(-qT) - K*e^(-rT). Holds regardless of the model."""
        for strike in (80.0, 100.0, 125.0):
            for q in (0.0, 0.02):
                call = opt(strike=strike, kind=CALL, dividend_yield=q)
                put = opt(strike=strike, kind=PUT, dividend_yield=q)
                t = call.t
                expected = 100.0 * math.exp(-q * t) - strike * math.exp(-0.04 * t)
                self.assertAlmostEqual(price(call) - price(put), expected, places=9)

    def test_price_exceeds_intrinsic(self):
        deep = opt(strike=70.0, kind=CALL)
        self.assertGreater(price(deep), deep.intrinsic)

    def test_expired_option_worth_intrinsic(self):
        self.assertAlmostEqual(price(opt(strike=90.0, dte=0, kind=CALL)), 10.0)
        self.assertAlmostEqual(price(opt(strike=90.0, dte=0, kind=PUT)), 0.0)

    def test_zero_vol_worth_intrinsic(self):
        self.assertAlmostEqual(price(opt(strike=90.0, volatility=0.0, kind=CALL)), 10.0)

    def test_price_rises_with_volatility(self):
        prices = [price(opt(volatility=v)) for v in (0.1, 0.2, 0.4, 0.8)]
        self.assertEqual(prices, sorted(prices))

    def test_price_rises_with_time(self):
        prices = [price(opt(dte=d)) for d in (7, 30, 90, 365)]
        self.assertEqual(prices, sorted(prices))

    def test_call_price_within_no_arbitrage_bounds(self):
        call = opt(strike=95.0, kind=CALL)
        self.assertGreaterEqual(price(call), call.intrinsic)
        self.assertLessEqual(price(call), call.spot)


class TestGreeks(unittest.TestCase):
    def test_call_delta_between_zero_and_one(self):
        for strike in (60.0, 100.0, 160.0):
            d = greeks(opt(strike=strike, kind=CALL)).delta
            self.assertGreater(d, 0.0)
            self.assertLess(d, 1.0)

    def test_put_delta_between_minus_one_and_zero(self):
        for strike in (60.0, 100.0, 160.0):
            d = greeks(opt(strike=strike, kind=PUT)).delta
            self.assertLess(d, 0.0)
            self.assertGreater(d, -1.0)

    def test_delta_parity(self):
        """Call delta minus put delta equals e^(-qT)."""
        for q in (0.0, 0.03):
            c = greeks(opt(kind=CALL, dividend_yield=q)).delta
            p = greeks(opt(kind=PUT, dividend_yield=q)).delta
            self.assertAlmostEqual(c - p, math.exp(-q * opt().t), places=9)

    def test_gamma_and_vega_identical_across_kinds(self):
        c, p = greeks(opt(kind=CALL)), greeks(opt(kind=PUT))
        self.assertAlmostEqual(c.gamma, p.gamma, places=12)
        self.assertAlmostEqual(c.vega, p.vega, places=12)

    def test_gamma_positive_and_peaks_near_the_money(self):
        near = greeks(opt(strike=100.0)).gamma
        far = greeks(opt(strike=140.0)).gamma
        self.assertGreater(near, 0)
        self.assertGreater(near, far)

    def test_vega_positive(self):
        self.assertGreater(greeks(opt()).vega, 0)

    def test_theta_negative_for_long_atm_option(self):
        self.assertLess(greeks(opt(kind=CALL)).theta, 0)
        self.assertLess(greeks(opt(kind=PUT)).theta, 0)

    def test_delta_matches_numerical_derivative(self):
        """Analytic delta must agree with a bumped finite difference."""
        base = opt(kind=CALL)
        h = 0.01
        numeric = (
            price(base.replace(spot=base.spot + h))
            - price(base.replace(spot=base.spot - h))
        ) / (2 * h)
        self.assertAlmostEqual(greeks(base).delta, numeric, places=6)

    def test_vega_matches_numerical_derivative(self):
        base = opt(kind=PUT)
        h = 1e-5
        numeric = (
            price(base.with_volatility(base.volatility + h))
            - price(base.with_volatility(base.volatility - h))
        ) / (2 * h)
        self.assertAlmostEqual(greeks(base).vega, numeric, places=4)

    def test_gamma_matches_numerical_second_derivative(self):
        base = opt(kind=CALL)
        h = 0.05
        numeric = (
            price(base.replace(spot=base.spot + h))
            - 2 * price(base)
            + price(base.replace(spot=base.spot - h))
        ) / (h * h)
        self.assertAlmostEqual(greeks(base).gamma, numeric, places=5)

    def test_scaling_per_contract(self):
        raw = greeks(opt())
        scaled = raw.per_contract(100.0)
        self.assertAlmostEqual(scaled.delta, raw.delta * 100)
        self.assertAlmostEqual(scaled.vega, raw.vega * 100)

    def test_quoted_conversions(self):
        g = greeks(opt())
        self.assertAlmostEqual(g.vega_per_point, g.vega / 100.0)
        self.assertAlmostEqual(g.theta_per_day, g.theta / 365.0)

    def test_expired_option_has_step_delta(self):
        itm = greeks(opt(strike=90.0, dte=0, kind=CALL))
        otm = greeks(opt(strike=110.0, dte=0, kind=CALL))
        self.assertAlmostEqual(itm.delta, 1.0)
        self.assertAlmostEqual(otm.delta, 0.0)
        self.assertAlmostEqual(itm.gamma, 0.0)


class TestImpliedVolatility(unittest.TestCase):
    def test_round_trip(self):
        """Price at a known vol, imply it back, recover the same number."""
        for strike in (90.0, 100.0, 110.0):
            for kind in (CALL, PUT):
                for true_vol in (0.15, 0.35, 0.7):
                    o = opt(strike=strike, kind=kind, volatility=true_vol, dte=45)
                    market = price(o)
                    recovered = implied_volatility(o.with_volatility(0.5), market)
                    self.assertAlmostEqual(
                        recovered, true_vol, places=6, msg=f"{kind} {strike} {true_vol}"
                    )

    def test_round_trip_out_of_the_money(self):
        """Where vega is small, Newton stalls and bisection must take over."""
        o = opt(strike=130.0, kind=CALL, volatility=0.4, dte=45)
        market = price(o)
        self.assertAlmostEqual(
            implied_volatility(o.with_volatility(0.2), market), 0.4, places=5
        )

    def test_round_trip_deep_in_the_money(self):
        o = opt(strike=70.0, kind=CALL, volatility=0.35, dte=90)
        market = price(o)
        self.assertAlmostEqual(
            implied_volatility(o.with_volatility(0.5), market), 0.35, places=5
        )

    def test_refuses_when_volatility_is_not_identifiable(self):
        """A strike worth zero at any vol carries no IV -- say so, don't guess."""
        o = opt(strike=400.0, kind=CALL, volatility=0.3, dte=3)
        with self.assertRaises(ValueError) as ctx:
            implied_volatility(o.with_volatility(0.25), price(o))
        self.assertIn("not identifiable", str(ctx.exception))

    def test_rejects_price_below_intrinsic(self):
        with self.assertRaises(ValueError):
            implied_volatility(opt(strike=80.0, kind=CALL), 1.0)

    def test_rejects_price_above_ceiling(self):
        with self.assertRaises(ValueError):
            implied_volatility(opt(kind=CALL), 500.0)

    def test_rejects_negative_price(self):
        with self.assertRaises(ValueError):
            implied_volatility(opt(), -1.0)

    def test_rejects_expired_option(self):
        with self.assertRaises(ValueError):
            implied_volatility(opt(dte=0), 5.0)


class TestProbabilities(unittest.TestCase):
    def test_itm_and_otm_sum_to_one(self):
        p = probabilities(opt(strike=95.0, kind=PUT))
        self.assertAlmostEqual(p.prob_itm + p.prob_otm, 1.0, places=12)

    def test_far_otm_has_low_itm_chance(self):
        p = probabilities(opt(strike=200.0, kind=CALL, dte=7))
        self.assertLess(p.prob_itm_pct, 1.0)

    def test_deep_itm_has_high_chance(self):
        p = probabilities(opt(strike=50.0, kind=CALL, dte=7))
        self.assertGreater(p.prob_itm_pct, 99.0)

    def test_touch_at_least_itm(self):
        """Touching is a weaker condition than finishing there -- always."""
        for kind in (CALL, PUT):
            for strike in (60.0, 85.0, 95.0, 105.0, 115.0, 140.0):
                p = probabilities(opt(strike=strike, kind=kind))
                self.assertGreaterEqual(
                    p.prob_touch + 1e-9, p.prob_itm, f"{kind} {strike}"
                )

    def test_touch_is_certain_when_already_in_the_money(self):
        """Spot past the strike means the strike has already been touched."""
        self.assertAlmostEqual(
            probabilities(opt(strike=105.0, kind=PUT)).prob_touch, 1.0
        )
        self.assertAlmostEqual(
            probabilities(opt(strike=95.0, kind=CALL)).prob_touch, 1.0
        )

    def test_touch_roughly_doubles_itm(self):
        """The classic rule of thumb, which should hold with small drift."""
        o = opt(strike=90.0, kind=PUT, rate=0.0, dte=30)
        p = probabilities(o)
        self.assertAlmostEqual(p.prob_touch / p.prob_itm, 2.0, delta=0.15)

    def test_touch_is_certain_at_the_barrier(self):
        self.assertAlmostEqual(probability_of_touch(opt(), 100.0), 1.0)

    def test_touch_bounded(self):
        for barrier in (1.0, 50.0, 99.0, 101.0, 500.0, 10_000.0):
            value = probability_of_touch(opt(), barrier)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_touch_rejects_bad_barrier(self):
        with self.assertRaises(ValueError):
            probability_of_touch(opt(), 0.0)

    def test_prob_profit_beats_prob_otm_for_short(self):
        """Breakeven sits beyond the strike, so profit is likelier than OTM."""
        p = probabilities(opt(strike=95.0, kind=PUT), premium=1.5)
        self.assertIsNotNone(p.prob_profit)
        self.assertGreater(p.prob_profit, p.prob_otm)

    def test_prob_profit_none_without_premium(self):
        self.assertIsNone(probabilities(opt()).prob_profit)

    def test_breakeven_direction(self):
        put = probabilities(opt(strike=95.0, kind=PUT), premium=2.0)
        call = probabilities(opt(strike=105.0, kind=CALL), premium=2.0)
        self.assertAlmostEqual(put.breakeven, 93.0)
        self.assertAlmostEqual(call.breakeven, 107.0)

    def test_rejects_negative_premium(self):
        with self.assertRaises(ValueError):
            probabilities(opt(), premium=-1.0)

    def test_expected_move_scales_with_sqrt_time(self):
        near = expected_move(opt(dte=30))
        far = expected_move(opt(dte=120))
        self.assertAlmostEqual(far / near, 2.0, places=6)

    def test_expected_move_zero_at_expiry(self):
        self.assertAlmostEqual(expected_move(opt(dte=0)), 0.0)

    def test_expected_move_rejects_bad_sigma_count(self):
        with self.assertRaises(ValueError):
            expected_move(opt(), standard_deviations=0)


class TestDeltaToStrike(unittest.TestCase):
    def test_recovers_target_delta(self):
        for target in (0.16, 0.30, 0.50):
            for kind in (CALL, PUT):
                o = opt(kind=kind)
                strike = delta_to_strike(o, target)
                actual = abs(greeks(o.with_strike(strike)).delta)
                self.assertAlmostEqual(actual, target, places=5)

    def test_lower_delta_is_further_out_of_the_money(self):
        o = opt(kind=PUT)
        self.assertLess(delta_to_strike(o, 0.10), delta_to_strike(o, 0.40))

    def test_call_delta_ordering(self):
        o = opt(kind=CALL)
        self.assertGreater(delta_to_strike(o, 0.10), delta_to_strike(o, 0.40))

    def test_rejects_out_of_range_delta(self):
        for bad in (0.0, 1.0, 1.5):
            with self.assertRaises(ValueError):
                delta_to_strike(opt(), bad)


if __name__ == "__main__":
    unittest.main()
