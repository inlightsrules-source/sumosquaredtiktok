"""Tests for IV rank/percentile, return on collateral, and the trade view."""

import math
import unittest
from datetime import date, timedelta

from options_desk import (
    CALL,
    PUT,
    CollateralAssumptions,
    Option,
    Series,
    analyze,
    annualize_compound,
    annualize_simple,
    build_context,
    cash_secured_put,
    covered_call,
    credit_spread,
    evaluate_trade,
    iv_percentile,
    iv_rank,
    naked_option,
    render_trade,
    term_structure_slope,
    with_collateral_yield,
)
from options_desk.ohlcv import Bar


def make_series(closes, symbol="TEST"):
    start = date(2024, 1, 1)
    return Series(
        symbol,
        [
            Bar(start + timedelta(days=i), c, c + 1, c - 1, c, 1_000_000.0)
            for i, c in enumerate(closes)
        ],
    )


class TestIVRank(unittest.TestCase):
    def test_rank_at_extremes(self):
        history = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertAlmostEqual(iv_rank(history, 10.0), 0.0)
        self.assertAlmostEqual(iv_rank(history, 50.0), 100.0)
        self.assertAlmostEqual(iv_rank(history, 30.0), 50.0)

    def test_rank_uses_last_value_by_default(self):
        self.assertAlmostEqual(iv_rank([10.0, 20.0, 30.0]), 100.0)

    def test_rank_of_flat_history_is_neutral(self):
        """No range means no position within it -- 50 says that honestly."""
        self.assertAlmostEqual(iv_rank([20.0] * 10, 20.0), 50.0)

    def test_rank_empty_history(self):
        self.assertIsNone(iv_rank([]))

    def test_percentile_counts_distribution_not_range(self):
        history = [10.0] * 90 + [100.0] * 10
        # 20 is above 90 of 100 observations, but low within the 10-100 range.
        self.assertAlmostEqual(iv_percentile(history, 20.0), 90.0)
        self.assertAlmostEqual(iv_rank(history, 20.0), 100.0 * 10 / 90)

    def test_rank_and_percentile_diverge_on_outliers(self):
        """The reason both are reported: one spike distorts rank, not percentile."""
        history = [20.0] * 200 + [120.0]
        rank = iv_rank(history, 25.0)
        pct = iv_percentile(history, 25.0)
        self.assertLess(rank, 10.0)
        self.assertGreater(pct, 90.0)


class TestVolContext(unittest.TestCase):
    def test_rejects_negative_iv(self):
        with self.assertRaises(ValueError):
            build_context(-5.0)

    def test_reports_gap_between_rank_and_percentile(self):
        ctx = build_context(25.0, iv_history=[20.0] * 200 + [120.0])
        self.assertIsNotNone(ctx.rank_percentile_gap)
        self.assertGreater(ctx.rank_percentile_gap, 20)
        self.assertIn("disagree", ctx.interpretation)

    def test_iv_below_realized_argues_against_selling(self):
        """A quiet options market against a volatile stock is not an edge."""
        closes = [100 * (1.04 if i % 2 else 0.96) for i in range(80)]
        ctx = build_context(10.0, iv_history=[10.0] * 50, series=make_series(closes))
        self.assertLess(ctx.premium_ratio, 1.0)
        self.assertEqual(ctx.edge, "buy premium")

    def test_iv_far_above_realized_supports_selling(self):
        closes = [100 + 0.01 * i for i in range(80)]
        ctx = build_context(45.0, iv_history=[20.0] * 100 + [45.0], series=make_series(closes))
        self.assertGreater(ctx.premium_ratio, 1.4)
        self.assertEqual(ctx.edge, "sell premium")

    def test_missing_history_is_stated_not_hidden(self):
        ctx = build_context(30.0)
        self.assertIsNone(ctx.iv_rank)
        self.assertIn("no history", ctx.interpretation.lower())

    def test_missing_series_is_stated(self):
        ctx = build_context(30.0, iv_history=[20.0, 30.0, 40.0])
        self.assertIsNone(ctx.realized)
        self.assertIn("realized", ctx.interpretation.lower())


class TestTermStructure(unittest.TestCase):
    def test_contango(self):
        slope, note = term_structure_slope(20.0, 26.0, 30, 90)
        self.assertGreater(slope, 0)
        self.assertIn("Contango", note)

    def test_backwardation(self):
        slope, note = term_structure_slope(45.0, 25.0, 7, 60)
        self.assertLess(slope, 0)
        self.assertIn("Backwardation", note)

    def test_rejects_inverted_expiries(self):
        with self.assertRaises(ValueError):
            term_structure_slope(20.0, 25.0, 90, 30)


class TestAnnualization(unittest.TestCase):
    def test_simple_scales_linearly(self):
        self.assertAlmostEqual(annualize_simple(0.02, 30), 0.02 * 365 / 30)

    def test_compound_exceeds_simple_for_positive_returns(self):
        self.assertGreater(annualize_compound(0.02, 30), annualize_simple(0.02, 30))

    def test_compound_known_value(self):
        self.assertAlmostEqual(annualize_compound(0.02, 365), 0.02, places=12)

    def test_rejects_zero_dte(self):
        with self.assertRaises(ValueError):
            annualize_simple(0.02, 0)


class TestCashSecuredPut(unittest.TestCase):
    def test_collateral_is_full_strike(self):
        p = cash_secured_put(strike=100.0, premium=2.0, dte=30)
        self.assertAlmostEqual(p.collateral, 10_000.0)
        self.assertAlmostEqual(p.premium, 200.0)
        self.assertAlmostEqual(p.net_collateral, 9_800.0)

    def test_static_and_annualized_returns(self):
        p = cash_secured_put(strike=100.0, premium=2.0, dte=30)
        self.assertAlmostEqual(p.static_return_pct, 2.0)
        self.assertAlmostEqual(p.annualized_simple_pct, 2.0 * 365 / 30, places=6)

    def test_net_return_exceeds_static(self):
        """Net backs out the credit, a smaller base, so the ratio is larger."""
        p = cash_secured_put(strike=100.0, premium=2.0, dte=30)
        self.assertGreater(p.net_return, p.static_return)

    def test_breakeven_below_strike(self):
        p = cash_secured_put(strike=100.0, premium=2.0, dte=30)
        self.assertAlmostEqual(p.breakeven, 98.0)

    def test_scales_with_contracts(self):
        one = cash_secured_put(strike=100.0, premium=2.0, dte=30, contracts=1)
        five = cash_secured_put(strike=100.0, premium=2.0, dte=30, contracts=5)
        self.assertAlmostEqual(five.collateral, one.collateral * 5)
        self.assertAlmostEqual(five.static_return, one.static_return)

    def test_rejects_bad_inputs(self):
        for kwargs in (
            dict(strike=0, premium=2.0, dte=30),
            dict(strike=100.0, premium=0, dte=30),
            dict(strike=100.0, premium=2.0, dte=0),
            dict(strike=100.0, premium=2.0, dte=30, contracts=0),
        ):
            with self.assertRaises(ValueError):
                cash_secured_put(**kwargs)


class TestCoveredCall(unittest.TestCase):
    def test_collateral_is_stock_basis(self):
        c = covered_call(stock_basis=100.0, strike=105.0, premium=2.0, dte=30)
        self.assertAlmostEqual(c.collateral, 10_000.0)

    def test_called_away_return_reported_in_notes(self):
        c = covered_call(stock_basis=100.0, strike=105.0, premium=2.0, dte=30)
        self.assertTrue(any("called away" in n for n in c.notes))

    def test_flags_strike_below_basis(self):
        c = covered_call(stock_basis=100.0, strike=90.0, premium=2.0, dte=30)
        self.assertTrue(any("below your basis" in n for n in c.notes))

    def test_max_profit_includes_appreciation(self):
        c = covered_call(stock_basis=100.0, strike=105.0, premium=2.0, dte=30)
        self.assertAlmostEqual(c.max_profit, 700.0)  # (2 + 105 - 100) * 100

    def test_rejects_bad_basis(self):
        with self.assertRaises(ValueError):
            covered_call(stock_basis=0, strike=105.0, premium=2.0, dte=30)


class TestCreditSpread(unittest.TestCase):
    def test_collateral_is_width_less_credit(self):
        s = credit_spread(short_strike=100.0, long_strike=95.0, credit=1.5, dte=30)
        self.assertAlmostEqual(s.collateral, 350.0)  # (5 - 1.5) * 100
        self.assertAlmostEqual(s.max_loss, 350.0)

    def test_return_on_collateral_exceeds_cash_secured(self):
        """Defined risk means less capital, hence a higher rate on it."""
        spread = credit_spread(short_strike=100.0, long_strike=95.0, credit=1.5, dte=30)
        csp = cash_secured_put(strike=100.0, premium=1.5, dte=30)
        self.assertGreater(spread.static_return, csp.static_return)

    def test_identifies_put_versus_call_spread(self):
        put = credit_spread(short_strike=100.0, long_strike=95.0, credit=1.0, dte=30)
        call = credit_spread(short_strike=100.0, long_strike=105.0, credit=1.0, dte=30)
        self.assertIn("Put", put.strategy)
        self.assertIn("Call", call.strategy)

    def test_breakeven_direction(self):
        put = credit_spread(short_strike=100.0, long_strike=95.0, credit=1.0, dte=30)
        call = credit_spread(short_strike=100.0, long_strike=105.0, credit=1.0, dte=30)
        self.assertAlmostEqual(put.breakeven, 99.0)
        self.assertAlmostEqual(call.breakeven, 101.0)

    def test_rejects_credit_exceeding_width(self):
        with self.assertRaises(ValueError):
            credit_spread(short_strike=100.0, long_strike=95.0, credit=5.0, dte=30)

    def test_rejects_identical_strikes(self):
        with self.assertRaises(ValueError):
            credit_spread(short_strike=100.0, long_strike=100.0, credit=1.0, dte=30)

    def test_return_on_risk(self):
        s = credit_spread(short_strike=100.0, long_strike=95.0, credit=1.5, dte=30)
        self.assertAlmostEqual(s.return_on_risk, 150.0 / 350.0)


class TestNakedOption(unittest.TestCase):
    def test_requirement_below_cash_secured(self):
        naked = naked_option(spot=100.0, strike=95.0, premium=2.0, dte=30)
        secured = cash_secured_put(strike=95.0, premium=2.0, dte=30)
        self.assertLess(naked.collateral, secured.collateral)

    def test_naked_call_loss_is_unbounded(self):
        c = naked_option(spot=100.0, strike=105.0, premium=2.0, dte=30, is_put=False)
        self.assertIsNone(c.max_loss)
        self.assertIsNone(c.return_on_risk)
        self.assertTrue(any("unbounded" in n for n in c.notes))

    def test_naked_put_loss_is_bounded_by_strike(self):
        p = naked_option(spot=100.0, strike=95.0, premium=2.0, dte=30)
        self.assertAlmostEqual(p.max_loss, 95.0 * 100 - 200.0)

    def test_further_otm_needs_less_margin(self):
        near = naked_option(spot=100.0, strike=98.0, premium=2.0, dte=30)
        far = naked_option(spot=100.0, strike=80.0, premium=2.0, dte=30)
        self.assertLess(far.collateral, near.collateral)

    def test_notes_warn_requirement_is_an_estimate(self):
        p = naked_option(spot=100.0, strike=95.0, premium=2.0, dte=30)
        self.assertTrue(any("estimate" in n.lower() for n in p.notes))


class TestCollateralYieldStacking(unittest.TestCase):
    def test_total_exceeds_premium_alone(self):
        profile = cash_secured_put(strike=100.0, premium=2.0, dte=30)
        stacked = with_collateral_yield(profile, fund_net_yield=0.04)
        self.assertGreater(
            stacked["total_yield_annualized"], stacked["premium_yield_annualized"]
        )

    def test_share_computed_correctly(self):
        profile = cash_secured_put(strike=100.0, premium=2.0, dte=30)
        stacked = with_collateral_yield(profile, fund_net_yield=0.04)
        expected = 0.04 / (profile.annualized_simple + 0.04)
        self.assertAlmostEqual(stacked["collateral_share_of_total"], expected)

    def test_uses_full_collateral_model_when_given_assumptions(self):
        profile = cash_secured_put(strike=100.0, premium=2.0, dte=30, contracts=10)
        stacked = with_collateral_yield(
            profile, assumptions=CollateralAssumptions(collateral=1.0)
        )
        self.assertIn("money market", stacked["collateral_source"])
        self.assertTrue(stacked["collateral_verdict"])

    def test_low_dte_makes_collateral_share_small(self):
        """A rich weekly premium dwarfs the collateral yield -- and says so."""
        profile = cash_secured_put(strike=100.0, premium=2.0, dte=7)
        stacked = with_collateral_yield(profile, fund_net_yield=0.04)
        self.assertLess(stacked["collateral_share_of_total"], 0.15)
        self.assertIn("premium is doing the work", stacked["summary"])

    def test_requires_one_of_the_two_inputs(self):
        profile = cash_secured_put(strike=100.0, premium=2.0, dte=30)
        with self.assertRaises(ValueError):
            with_collateral_yield(profile)

    def test_rejects_negative_yield(self):
        profile = cash_secured_put(strike=100.0, premium=2.0, dte=30)
        with self.assertRaises(ValueError):
            with_collateral_yield(profile, fund_net_yield=-0.01)


class TestTradeView(unittest.TestCase):
    def _option(self):
        return Option(spot=100.0, strike=92.0, dte=30.0, volatility=0.28, kind=PUT)

    def test_assembles_all_sections(self):
        view = evaluate_trade(
            self._option(),
            premium=1.2,
            vol_context=build_context(28.0, iv_history=[20.0] * 100 + [28.0]),
            profile=cash_secured_put(strike=92.0, premium=1.2, dte=30),
        )
        self.assertIsNotNone(view.greeks)
        self.assertIsNotNone(view.odds)
        self.assertTrue(view.verdict)

    def test_greeks_are_per_contract(self):
        view = evaluate_trade(self._option(), premium=1.2)
        # A short-dated OTM put should carry delta well inside one contract.
        self.assertLess(abs(view.greeks.delta), 100.0)
        self.assertGreater(abs(view.greeks.delta), 0.0)

    def test_verdict_mentions_probabilities(self):
        view = evaluate_trade(self._option(), premium=1.2)
        self.assertIn("%", view.verdict)

    def test_level_note_when_analysis_supplied(self):
        closes = [100 + 8 * math.sin(2 * math.pi * i / 30) for i in range(300)]
        analysis = analyze(make_series(closes))
        view = evaluate_trade(
            Option(spot=analysis.price, strike=analysis.price * 0.9, dte=30,
                   volatility=0.3, kind=PUT),
            premium=1.0,
            analysis=analysis,
        )
        self.assertTrue(view.level_note)

    def test_render_includes_key_sections(self):
        view = evaluate_trade(
            self._option(),
            premium=1.2,
            vol_context=build_context(28.0, iv_history=[20.0] * 100),
            profile=cash_secured_put(strike=92.0, premium=1.2, dte=30),
        )
        text = render_trade(view)
        for heading in ("GREEKS", "PROBABILITY", "RETURN ON COLLATERAL", "VERDICT"):
            self.assertIn(heading, text)

    def test_rejects_non_positive_premium(self):
        with self.assertRaises(ValueError):
            evaluate_trade(self._option(), premium=0)


if __name__ == "__main__":
    unittest.main()


class TestOrdinalFormatting(unittest.TestCase):
    def test_suffixes(self):
        from options_desk.vol import _ordinal

        cases = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 11: "11th",
                 12: "12th", 13: "13th", 21: "21st", 42: "42nd",
                 53: "53rd", 72: "72nd", 100: "100th", 111: "111th"}
        for value, expected in cases.items():
            self.assertEqual(_ordinal(value), expected)
