"""Tests for the short-premium screen."""

import unittest

from options_desk.pricing import CALL, PUT, Option, price
from options_desk.screen import (
    DEFAULT_WEIGHTS,
    Candidate,
    Filters,
    best,
    render_screen,
    score_candidates,
)
from options_desk.vol import build_context


def contract(strike=90.0, dte=30.0, iv=0.30, spot=100.0, kind=PUT, **quote):
    """A candidate priced at ``iv``, quoted around that price by default."""
    option = Option(spot=spot, strike=strike, dte=dte, volatility=iv, kind=kind)
    theoretical = price(option)
    defaults = dict(
        bid=round(theoretical * 0.98, 2),
        ask=round(theoretical * 1.02, 2),
        open_interest=1500,
        volume=300,
        underlying="TEST",
    )
    defaults.update(quote)
    return Candidate(option=option, **defaults)


class TestCandidate(unittest.TestCase):
    def test_spread_metrics(self):
        c = contract(bid=1.00, ask=1.10)
        self.assertAlmostEqual(c.mid, 1.05)
        self.assertAlmostEqual(c.spread, 0.10, places=9)
        self.assertAlmostEqual(c.spread_pct, 100 * 0.10 / 1.05, places=6)

    def test_rejects_crossed_market(self):
        with self.assertRaises(ValueError):
            contract(bid=2.0, ask=1.0)

    def test_rejects_negative_bid(self):
        with self.assertRaises(ValueError):
            contract(bid=-1.0, ask=1.0)

    def test_zero_mid_reads_as_fully_wide(self):
        self.assertAlmostEqual(contract(bid=0.0, ask=0.0).spread_pct, 100.0)


class TestEdge(unittest.TestCase):
    def test_positive_edge_when_iv_exceeds_realized(self):
        """Sold at 40 vol, stock realizing 20 -- the premium is being paid."""
        ranked = score_candidates([contract(iv=0.40)], realized_vol=20.0)
        self.assertGreater(ranked[0].edge_dollars, 0)

    def test_negative_edge_when_iv_below_realized(self):
        ranked = score_candidates([contract(iv=0.20)], realized_vol=40.0)
        self.assertLess(ranked[0].edge_dollars, 0)
        self.assertIsNotNone(ranked[0].disqualified)

    def test_edge_near_zero_when_iv_matches_realized(self):
        """Priced at the realized vol, only the bid/mid haircut remains."""
        c = contract(iv=0.30)
        theoretical = price(c.option)
        exact = Candidate(
            option=c.option,
            bid=theoretical,
            ask=theoretical,
            open_interest=1500,
            volume=300,
        )
        ranked = score_candidates(
            [exact], realized_vol=30.0, filters=Filters(require_positive_edge=False)
        )
        self.assertAlmostEqual(ranked[0].edge_dollars, 0.0, places=6)

    def test_uses_bid_by_default_not_mid(self):
        """Mid fills are not guaranteed; pricing at mid overstates every edge."""
        c = contract(iv=0.40)
        at_bid = score_candidates([c], realized_vol=20.0, use_bid=True)[0]
        at_mid = score_candidates([c], realized_vol=20.0, use_bid=False)[0]
        self.assertLess(at_bid.edge_dollars, at_mid.edge_dollars)

    def test_rejects_negative_realized_vol(self):
        with self.assertRaises(ValueError):
            score_candidates([contract()], realized_vol=-5.0)


class TestFilters(unittest.TestCase):
    def test_wide_spread_excluded(self):
        c = contract(iv=0.40, bid=1.00, ask=2.00)
        self.assertIn("spread", score_candidates([c], 20.0)[0].disqualified)

    def test_thin_open_interest_excluded(self):
        c = contract(iv=0.40, open_interest=5)
        self.assertIn("open interest", score_candidates([c], 20.0)[0].disqualified)

    def test_high_delta_excluded(self):
        c = contract(strike=99.0, iv=0.40)
        self.assertIn("delta", score_candidates([c], 20.0)[0].disqualified)

    def test_dte_bounds_enforced(self):
        near = contract(dte=2.0, iv=0.40)
        far = contract(dte=200.0, iv=0.40)
        self.assertIsNotNone(score_candidates([near], 20.0)[0].disqualified)
        self.assertIsNotNone(score_candidates([far], 20.0)[0].disqualified)

    def test_negative_edge_excluded_by_default(self):
        # A real credit, but priced well under what the stock is delivering.
        c = contract(strike=95.0, iv=0.25)
        self.assertIn("realized", score_candidates([c], 45.0)[0].disqualified)

    def test_negative_edge_allowed_when_filter_relaxed(self):
        c = contract(strike=95.0, iv=0.25)
        ranked = score_candidates(
            [c], 45.0, filters=Filters(require_positive_edge=False)
        )
        self.assertIsNone(ranked[0].disqualified)

    def test_tiny_credit_excluded(self):
        c = contract(iv=0.40, bid=0.01, ask=0.02)
        self.assertIsNotNone(score_candidates([c], 20.0)[0].disqualified)


class TestRanking(unittest.TestCase):
    def test_higher_edge_ranks_above_lower(self):
        rich = contract(strike=90.0, iv=0.45)
        thin = contract(strike=90.0, iv=0.32)
        ranked = score_candidates([thin, rich], realized_vol=25.0)
        self.assertGreater(ranked[0].score, ranked[1].score)
        self.assertAlmostEqual(ranked[0].candidate.option.volatility, 0.45)

    def test_tighter_spread_wins_all_else_equal(self):
        base = contract(iv=0.40)
        theoretical = price(base.option)
        tight = Candidate(
            option=base.option,
            bid=round(theoretical * 0.995, 2),
            ask=round(theoretical * 1.005, 2),
            open_interest=1500,
            volume=300,
        )
        wide = Candidate(
            option=base.option,
            bid=round(theoretical * 0.97, 2),
            ask=round(theoretical * 1.03, 2),
            open_interest=1500,
            volume=300,
        )
        ranked = score_candidates([wide, tight], realized_vol=22.0)
        self.assertGreater(ranked[0].components["liquidity"], ranked[1].components["liquidity"])

    def test_excluded_sort_below_eligible(self):
        good = contract(iv=0.40)
        bad = contract(iv=0.40, open_interest=1)
        ranked = score_candidates([bad, good], realized_vol=20.0)
        self.assertTrue(ranked[0].is_eligible)
        self.assertFalse(ranked[-1].is_eligible)

    def test_best_returns_top_eligible(self):
        rich = contract(strike=88.0, iv=0.45)
        thin = contract(strike=92.0, iv=0.33)
        self.assertAlmostEqual(
            best([thin, rich], realized_vol=25.0).candidate.option.volatility, 0.45
        )

    def test_best_returns_none_when_nothing_qualifies(self):
        """No trade is a real answer and must be representable."""
        self.assertIsNone(best([contract(strike=95.0, iv=0.25)], realized_vol=50.0))

    def test_empty_input(self):
        self.assertEqual(score_candidates([], 25.0), [])
        self.assertIsNone(best([], 25.0))

    def test_score_bounded(self):
        ranked = score_candidates(
            [contract(iv=v) for v in (0.35, 0.5, 0.8)], realized_vol=20.0
        )
        for s in ranked:
            self.assertGreaterEqual(s.score, 0.0)
            self.assertLessEqual(s.score, 100.0)


class TestWeights(unittest.TestCase):
    def test_defaults_sum_to_one(self):
        self.assertAlmostEqual(sum(DEFAULT_WEIGHTS.values()), 1.0, places=9)

    def test_custom_weights_normalized(self):
        ranked = score_candidates(
            [contract(iv=0.40)],
            20.0,
            weights={k: 2.0 for k in DEFAULT_WEIGHTS},
        )
        self.assertLessEqual(ranked[0].score, 100.0)

    def test_rejects_missing_weight_key(self):
        with self.assertRaises(ValueError):
            score_candidates([contract()], 20.0, weights={"edge": 1.0})

    def test_rejects_unknown_weight_key(self):
        bad = dict(DEFAULT_WEIGHTS)
        bad["vibes"] = 0.5
        with self.assertRaises(ValueError):
            score_candidates([contract()], 20.0, weights=bad)

    def test_rejects_negative_weight(self):
        bad = dict(DEFAULT_WEIGHTS)
        bad["edge"] = -1.0
        with self.assertRaises(ValueError):
            score_candidates([contract()], 20.0, weights=bad)


class TestVolContextInfluence(unittest.TestCase):
    def test_buy_premium_context_lowers_score(self):
        c = contract(iv=0.40)
        neutral = score_candidates([c], 20.0)[0]
        cheap = build_context(10.0, iv_history=[40.0] * 100)
        with_context = score_candidates([c], 20.0, vol_context=cheap)[0]
        self.assertLess(with_context.components["vol_context"], neutral.components["vol_context"])

    def test_absent_context_is_neutral(self):
        self.assertAlmostEqual(
            score_candidates([contract(iv=0.40)], 20.0)[0].components["vol_context"], 0.5
        )


class TestRendering(unittest.TestCase):
    def test_render_names_top_candidate(self):
        text = render_screen(score_candidates([contract(iv=0.42)], 20.0))
        self.assertIn("TOP CANDIDATE", text)
        self.assertIn("Component scores", text)

    def test_render_states_when_nothing_qualifies(self):
        text = render_screen(score_candidates([contract(strike=95.0, iv=0.25)], 50.0))
        self.assertIn("No candidate passed", text)
        self.assertIn("EXCLUDED", text)

    def test_render_carries_the_caveat(self):
        text = render_screen(score_candidates([contract(iv=0.42)], 20.0))
        self.assertIn("not a recommendation", text)

    def test_render_handles_empty(self):
        self.assertIn("No candidate passed", render_screen([]))


if __name__ == "__main__":
    unittest.main()
