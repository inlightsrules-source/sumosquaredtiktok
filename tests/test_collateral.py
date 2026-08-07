"""Tests for the money market collateral decision model."""

import unittest

from options_desk import CollateralAssumptions, compare_scenarios, evaluate


def base(**overrides):
    defaults = dict(
        collateral=1_000_000.0,
        broker_credit_rate=0.005,
        fund_gross_yield=0.042,
        fund_expense_ratio=0.0015,
        haircut=0.01,
        buying_power_return=0.0,
        margin_debit_rate=0.085,
        prob_forced_borrow=0.10,
        forced_borrow_fraction=0.25,
        forced_borrow_days=3,
    )
    defaults.update(overrides)
    return CollateralAssumptions(**defaults)


class TestValidation(unittest.TestCase):
    def test_rejects_non_positive_collateral(self):
        with self.assertRaises(ValueError):
            base(collateral=0)

    def test_rejects_out_of_range_haircut(self):
        with self.assertRaises(ValueError):
            base(haircut=1.0)

    def test_rejects_out_of_range_probability(self):
        with self.assertRaises(ValueError):
            base(prob_forced_borrow=1.5)

    def test_rejects_negative_borrow_days(self):
        with self.assertRaises(ValueError):
            base(forced_borrow_days=-1)

    def test_rejects_negative_expense_ratio(self):
        with self.assertRaises(ValueError):
            base(fund_expense_ratio=-0.01)


class TestEconomics(unittest.TestCase):
    def test_gross_pickup_is_net_of_fees(self):
        result = evaluate(base())
        # (4.2% - 0.15%) - 0.5% = 3.55% on 1,000,000
        self.assertAlmostEqual(result.gross_pickup, 35_500.0, places=2)

    def test_positive_case_recommends_sweeping(self):
        result = evaluate(base())
        self.assertGreater(result.net_benefit, 0)
        self.assertIn("sweep", result.verdict.lower())

    def test_no_spread_means_no_sweep(self):
        """If the broker already pays the short rate there is nothing to capture."""
        result = evaluate(base(broker_credit_rate=0.045))
        self.assertLessEqual(result.gross_pickup, 0)
        self.assertIn("do not sweep", result.verdict.lower())

    def test_haircut_costs_nothing_when_buying_power_idle(self):
        self.assertAlmostEqual(evaluate(base(buying_power_return=0.0)).haircut_cost, 0.0)

    def test_haircut_bites_when_buying_power_is_productive(self):
        idle = evaluate(base(buying_power_return=0.0))
        productive = evaluate(base(buying_power_return=0.20))
        self.assertGreater(productive.haircut_cost, idle.haircut_cost)
        self.assertLess(productive.net_benefit, idle.net_benefit)

    def test_forced_borrow_reduces_benefit_monotonically(self):
        benefits = [
            evaluate(base(prob_forced_borrow=p)).net_benefit
            for p in (0.0, 0.25, 0.5, 1.0)
        ]
        self.assertEqual(benefits, sorted(benefits, reverse=True))

    def test_borrow_cost_uses_spread_not_full_rate(self):
        """Collateral keeps earning while pledged, so only the spread is lost."""
        result = evaluate(
            base(prob_forced_borrow=1.0, forced_borrow_fraction=1.0, forced_borrow_days=360)
        )
        net_yield = 0.042 - 0.0015
        expected = 1_000_000.0 * (0.085 - net_yield)
        self.assertAlmostEqual(result.expected_borrow_cost, expected, places=2)

    def test_heavy_borrow_risk_can_flip_the_answer(self):
        result = evaluate(
            base(prob_forced_borrow=1.0, forced_borrow_fraction=1.0, forced_borrow_days=300)
        )
        self.assertLess(result.net_benefit, 0)
        self.assertIn("do not sweep", result.verdict.lower())

    def test_state_tax_exemption_adds_value(self):
        plain = evaluate(base())
        exempt = evaluate(base(state_tax_rate=0.10, treasury_exempt_fraction=1.0))
        self.assertGreater(exempt.state_tax_saving, 0)
        self.assertGreater(exempt.net_benefit, plain.net_benefit)

    def test_net_bps_matches_net_benefit(self):
        result = evaluate(base())
        self.assertAlmostEqual(result.net_bps, 10_000 * result.net_benefit / 1_000_000, places=6)

    def test_breakeven_probability_zeroes_the_trade(self):
        """Where a break-even exists, plugging it back in must net zero."""
        # Borrow costs large enough that the break-even lands inside [0, 1].
        assumptions = base(forced_borrow_fraction=1.0, forced_borrow_days=340)
        result = evaluate(assumptions)
        self.assertIsNotNone(result.breakeven_prob)
        self.assertFalse(result.survives_certain_borrow)
        at_breakeven = evaluate(
            base(
                forced_borrow_fraction=1.0,
                forced_borrow_days=340,
                prob_forced_borrow=result.breakeven_prob,
            )
        )
        self.assertAlmostEqual(at_breakeven.net_benefit, 0.0, places=2)

    def test_breakeven_is_a_real_probability(self):
        result = evaluate(base(forced_borrow_fraction=1.0, forced_borrow_days=340))
        self.assertGreaterEqual(result.breakeven_prob, 0.0)
        self.assertLessEqual(result.breakeven_prob, 1.0)

    def test_no_breakeven_when_trade_survives_certain_borrow(self):
        """A 140% 'break-even' is not a probability -- report none instead."""
        result = evaluate(base(prob_forced_borrow=0.0, forced_borrow_days=1))
        self.assertIsNone(result.breakeven_prob)
        self.assertTrue(result.survives_certain_borrow)
        self.assertGreater(evaluate(base(prob_forced_borrow=1.0)).net_benefit, 0)

    def test_no_borrow_scenario_has_no_breakeven(self):
        result = evaluate(base(forced_borrow_days=0))
        self.assertIsNone(result.breakeven_prob)
        self.assertTrue(result.survives_certain_borrow)


class TestOutput(unittest.TestCase):
    def test_notes_warn_about_prime_funds(self):
        notes = " ".join(evaluate(base()).notes).lower()
        self.assertIn("prime", notes)

    def test_notes_flag_free_haircut_assumption(self):
        notes = " ".join(evaluate(base(buying_power_return=0.0)).notes).lower()
        self.assertIn("buying power", notes)

    def test_str_renders_all_components(self):
        text = str(evaluate(base()))
        for label in ("Gross yield pickup", "Haircut cost", "Net benefit", "Verdict"):
            self.assertIn(label, text)

    def test_compare_scenarios_covers_each_probability(self):
        probs = (0.0, 0.2, 0.6)
        results = compare_scenarios(base(), probs)
        self.assertEqual(tuple(p for p, _ in results), probs)
        self.assertEqual(len(results), 3)

    def test_compare_scenarios_leaves_base_untouched(self):
        assumptions = base()
        compare_scenarios(assumptions, (0.0, 0.9))
        self.assertAlmostEqual(assumptions.prob_forced_borrow, 0.10)


if __name__ == "__main__":
    unittest.main()
