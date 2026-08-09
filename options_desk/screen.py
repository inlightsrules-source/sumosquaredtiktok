"""Rank candidate short-premium contracts and return the best of them.

The central idea, and the reason this is not just a premium sort:

    You collect the premium priced at *implied* volatility.
    You bear the risk that plays out at *realized* volatility.
    The edge is the difference, in dollars, on that specific contract.

Concretely, a short option sold for credit C carries an expected cost equal
to its Black-Scholes value repriced at realized volatility. So

    edge = C - BS_price(sigma_realized)

is the variance risk premium captured on this contract. It is positive only
when the option is priced above what the stock has actually been delivering,
which is the entire thesis of selling premium. Ranking on raw yield instead
puts the richest-looking contract on top precisely when the market has
correctly identified a dangerous name.

Two limits on that, stated up front because they bound how much the ranking
can be trusted:

**Realized volatility is backward-looking.** Using it as the truth assumes
volatility persists. It usually does -- vol clusters -- but it breaks exactly
at the events that matter, and an earnings date inside the expiry makes the
whole calculation wrong in the dangerous direction.

**A ranking is not a recommendation.** This orders candidates by a stated
scoring function whose weights are judgment calls. Two reasonable desks would
weight these differently and get different winners.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .interpret import Analysis
from .pricing import Greeks, Option, Probabilities, greeks, price, probabilities
from .returns import ReturnProfile, cash_secured_put
from .vol import VolContext

# Default weights. They sum to 1.0 and are deliberately exposed: the ranking
# is only as defensible as these numbers, and they are opinion, not fact.
DEFAULT_WEIGHTS = {
    "edge": 0.35,        # variance risk premium captured, annualized
    "probability": 0.20, # chance of keeping the credit
    "liquidity": 0.20,   # can you actually get filled, and get out
    "technical": 0.15,   # is the strike protected by a real level
    "vol_context": 0.10, # is IV elevated for this name historically
}


@dataclass
class Candidate:
    """One quotable contract, as it comes off a chain."""

    option: Option
    bid: float
    ask: float
    open_interest: int = 0
    volume: int = 0
    underlying: str = ""

    def __post_init__(self) -> None:
        if self.bid < 0:
            raise ValueError(f"bid cannot be negative, got {self.bid}")
        if self.ask < self.bid:
            raise ValueError(f"ask {self.ask} is below bid {self.bid}")
        if self.open_interest < 0 or self.volume < 0:
            raise ValueError("open interest and volume cannot be negative")

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as a share of mid. The cost of being wrong about the fill."""
        return 100.0 if self.mid <= 0 else 100.0 * self.spread / self.mid


@dataclass
class Scored:
    """A candidate with its score and the reasoning behind it."""

    candidate: Candidate
    score: float                       # 0-100 composite
    components: dict[str, float]       # each sub-score, 0-1
    edge_dollars: float                # variance premium captured, per contract
    edge_annualized: float             # that edge as a rate on collateral
    odds: Probabilities
    greeks: Greeks
    profile: ReturnProfile | None
    reasons: list[str] = field(default_factory=list)
    disqualified: str | None = None    # why it was excluded, if it was

    @property
    def is_eligible(self) -> bool:
        return self.disqualified is None

    def __str__(self) -> str:
        head = (
            f"{self.candidate.underlying or 'N/A'} "
            f"{self.candidate.option.kind} {self.candidate.option.strike:g} "
            f"{self.candidate.option.dte:.0f}d"
        )
        if self.disqualified:
            return f"{head}  EXCLUDED -- {self.disqualified}"
        return (
            f"{head}  score {self.score:.1f}  "
            f"edge {self.edge_dollars:+,.2f} ({100 * self.edge_annualized:+.1f}% ann) "
            f"POP {self.odds.prob_profit_pct or 0:.0f}%"
        )


@dataclass
class Filters:
    """Hard exclusions applied before scoring.

    These are disqualifiers rather than penalties: a contract failing any of
    them is not a worse trade, it is one that should not be on the list at
    all. Scoring a wide-spread illiquid contract highly and trusting the
    number is how a screen produces trades that cannot be exited.
    """

    max_spread_pct: float = 10.0
    min_open_interest: int = 100
    max_abs_delta: float = 0.35
    min_credit: float = 0.05
    require_positive_edge: bool = True
    min_dte: float = 7.0
    max_dte: float = 60.0


def score_candidates(
    candidates: list[Candidate],
    realized_vol: float,
    vol_context: VolContext | None = None,
    analysis: Analysis | None = None,
    filters: Filters | None = None,
    weights: dict[str, float] | None = None,
    use_bid: bool = True,
) -> list[Scored]:
    """Score every candidate and return them ranked, best first.

    ``realized_vol`` is the annualized realized volatility as a percent
    (28.0 == 28%), matching the rest of this package. It is the yardstick the
    premium is measured against, so it should come from the same underlying
    and a window comparable to the expiry.

    ``use_bid`` prices the credit at the bid rather than the mid. That is the
    conservative and usually correct choice: mid-price fills are not
    guaranteed, and a screen built on mid systematically overstates every
    edge it finds, worst on exactly the illiquid contracts it should avoid.

    Excluded candidates are returned too, carrying their reason, so the screen
    can be audited rather than silently dropping things.
    """
    if realized_vol < 0:
        raise ValueError(f"realized_vol cannot be negative, got {realized_vol}")
    filters = filters or Filters()
    weights = _validate_weights(weights or DEFAULT_WEIGHTS)

    scored = [
        _score_one(c, realized_vol, vol_context, analysis, filters, weights, use_bid)
        for c in candidates
    ]
    # Eligible first, then by score. Excluded contracts sort to the bottom
    # rather than vanishing, so a screen returning nothing shows you why.
    return sorted(
        scored, key=lambda s: (s.is_eligible, s.score), reverse=True
    )


def best(
    candidates: list[Candidate],
    realized_vol: float,
    **kwargs,
) -> Scored | None:
    """The top-ranked eligible candidate, or ``None`` if none qualify.

    ``None`` is a real answer and a common one. A screen that always returns
    something will hand you a trade on the day when nothing is worth doing.
    """
    ranked = score_candidates(candidates, realized_vol, **kwargs)
    return next((s for s in ranked if s.is_eligible), None)


def _score_one(
    candidate: Candidate,
    realized_vol: float,
    vol_context: VolContext | None,
    analysis: Analysis | None,
    filters: Filters,
    weights: dict[str, float],
    use_bid: bool,
) -> Scored:
    option = candidate.option
    credit = candidate.bid if use_bid else candidate.mid

    # Fair value under realized vol: what this contract should cost if the
    # stock keeps moving the way it has been.
    fair_value = price(option.with_volatility(realized_vol / 100.0))
    edge_dollars = (credit - fair_value) * 100.0

    g = greeks(option).per_contract()
    odds = probabilities(option, credit if credit > 0 else None)
    profile = _build_profile(candidate, credit)
    collateral = profile.collateral if profile else option.strike * 100.0
    edge_annualized = (
        0.0
        if collateral <= 0 or option.dte <= 0
        else (edge_dollars / collateral) * 365.0 / option.dte
    )

    result = Scored(
        candidate=candidate,
        score=0.0,
        components={},
        edge_dollars=edge_dollars,
        edge_annualized=edge_annualized,
        odds=odds,
        greeks=g,
        profile=profile,
    )

    disqualifier = _disqualify(candidate, credit, g, edge_dollars, filters)
    if disqualifier:
        result.disqualified = disqualifier
        return result

    components = {
        "edge": _clamp(edge_annualized / 0.20),
        "probability": _clamp(((odds.prob_profit or odds.prob_otm) - 0.5) / 0.45),
        "liquidity": _liquidity_score(candidate),
        "technical": _technical_score(option, analysis),
        "vol_context": _vol_score(vol_context),
    }
    result.components = components
    result.score = 100.0 * sum(components[k] * weights[k] for k in weights)
    result.reasons = _reasons(candidate, result, realized_vol, analysis, credit)
    return result


def _disqualify(
    candidate: Candidate,
    credit: float,
    g: Greeks,
    edge_dollars: float,
    filters: Filters,
) -> str | None:
    option = candidate.option
    if credit < filters.min_credit:
        return f"credit {credit:.2f} below the {filters.min_credit:.2f} minimum"
    if candidate.spread_pct > filters.max_spread_pct:
        return (
            f"bid/ask spread is {candidate.spread_pct:.0f}% of mid, above the "
            f"{filters.max_spread_pct:.0f}% limit -- the fill cost would eat the edge"
        )
    if candidate.open_interest < filters.min_open_interest:
        return (
            f"open interest {candidate.open_interest} below "
            f"{filters.min_open_interest}; exiting early may not be possible"
        )
    if abs(g.delta / 100.0) > filters.max_abs_delta:
        return (
            f"delta {abs(g.delta / 100.0):.2f} exceeds the "
            f"{filters.max_abs_delta:.2f} limit -- too close to the money"
        )
    if option.dte < filters.min_dte:
        return f"{option.dte:.0f} DTE is inside the {filters.min_dte:.0f} day floor"
    if option.dte > filters.max_dte:
        return f"{option.dte:.0f} DTE is beyond the {filters.max_dte:.0f} day ceiling"
    if filters.require_positive_edge and edge_dollars <= 0:
        return (
            f"implied volatility is at or below realized, so the credit "
            f"({edge_dollars:+.2f} vs fair value) does not pay for the risk"
        )
    return None


def _build_profile(candidate: Candidate, credit: float) -> ReturnProfile | None:
    """Return profile for the position, where one applies.

    Only cash-secured puts are modelled here. A short call's collateral
    depends on whether it is covered by stock or held on margin, which is a
    property of the account rather than of the contract, so the screen
    declines to guess.
    """
    option = candidate.option
    if credit <= 0 or option.is_call:
        return None
    try:
        return cash_secured_put(strike=option.strike, premium=credit, dte=option.dte)
    except ValueError:
        return None


def _liquidity_score(candidate: Candidate) -> float:
    """Tight spreads and real open interest. Weighted toward the spread.

    Spread is the cost you pay on entry and again on exit, and it is certain.
    Open interest only tells you whether anyone will be there when you want
    out, which matters but is a softer constraint.
    """
    spread_score = _clamp(1.0 - candidate.spread_pct / 10.0)
    oi_score = _clamp(candidate.open_interest / 2000.0)
    volume_score = _clamp(candidate.volume / 500.0)
    return 0.6 * spread_score + 0.25 * oi_score + 0.15 * volume_score


def _technical_score(option: Option, analysis: Analysis | None) -> float:
    """Reward strikes sitting beyond a defended level.

    Neutral (0.5) without an analysis: absence of information should not
    read as a good or a bad sign.
    """
    if analysis is None:
        return 0.5
    is_upside = option.strike > option.spot
    level = analysis.key_resistance if is_upside else analysis.key_support
    if level is None:
        return 0.5
    beyond = (option.strike > level.price) if is_upside else (option.strike < level.price)
    if not beyond:
        return 0.3  # price can reach the strike without breaking anything
    # Past the level, and the stronger the level the better the protection.
    return _clamp(0.6 + 0.4 * level.strength / 100.0)


def _vol_score(vol_context: VolContext | None) -> float:
    if vol_context is None:
        return 0.5
    if vol_context.edge == "buy premium":
        return 0.1
    if vol_context.iv_percentile is not None:
        return _clamp(vol_context.iv_percentile / 100.0)
    return 0.5


def _reasons(
    candidate: Candidate,
    scored: Scored,
    realized_vol: float,
    analysis: Analysis | None,
    credit: float,
) -> list[str]:
    option = candidate.option
    out = [
        f"Collecting {credit:.2f} against a fair value of "
        f"{credit - scored.edge_dollars / 100.0:.2f} at {realized_vol:.1f}% realized "
        f"vol -- an edge of {scored.edge_dollars:+,.2f} per contract, "
        f"{100 * scored.edge_annualized:+.1f}% annualized on collateral.",
        f"{scored.odds.prob_otm_pct:.0f}% chance of expiring worthless; "
        f"{scored.odds.prob_touch_pct:.0f}% chance of touching the strike first.",
        f"Spread is {candidate.spread_pct:.1f}% of mid on "
        f"{candidate.open_interest:,} open interest.",
    ]
    if scored.odds.prob_touch_pct > 2.2 * scored.odds.prob_itm_pct:
        out.append(
            "Touch probability runs well above twice the ITM chance, so managing "
            "this on a breach rather than holding to expiry changes the odds "
            "materially for the worse."
        )
    if analysis is not None:
        level = (
            analysis.key_resistance
            if option.strike > option.spot
            else analysis.key_support
        )
        if level is not None:
            out.append(
                f"Nearest defended level is {level.price:,.2f} "
                f"(strength {level.strength:.0f}/100)."
            )
    theta, vega = scored.greeks.theta_per_day, scored.greeks.vega_per_point
    if abs(vega) > abs(theta) * 3:
        out.append(
            f"Vega ({vega:+,.2f} per point) dominates theta ({theta:+,.2f} per "
            "day) -- a volatility expansion hurts faster than decay pays."
        )
    return out


def render_screen(ranked: list[Scored], limit: int = 10) -> str:
    """Terminal readout for a screen result."""
    width = 78
    lines = ["=" * width, "  SHORT PREMIUM SCREEN", "=" * width, ""]
    eligible = [s for s in ranked if s.is_eligible]

    if not eligible:
        lines += [
            "  No candidate passed the filters.",
            "",
            _wrap(
                "That is a real answer, not a failure. A screen that always "
                "returns something will hand you a trade on the day nothing is "
                "worth doing. Exclusions below."
            ),
            "",
        ]
    else:
        top = eligible[0]
        lines += ["  TOP CANDIDATE", "-" * width, f"  {top}", ""]
        for reason in top.reasons:
            lines.append(_wrap(reason, bullet="-"))
        lines += ["", "  Component scores (0-1):"]
        for name, value in sorted(top.components.items(), key=lambda kv: -kv[1]):
            bar = "#" * max(0, round(value * 20))
            lines.append(f"    {name:<12} {value:>5.2f}  {bar}")
        if top.profile is not None:
            lines += ["", "  RETURN ON COLLATERAL", "-" * width, str(top.profile)]

        if len(eligible) > 1:
            lines += ["", "  RUNNERS-UP", "-" * width]
            for s in eligible[1:limit]:
                lines.append(f"  {s}")

    excluded = [s for s in ranked if not s.is_eligible]
    if excluded:
        lines += ["", "  EXCLUDED", "-" * width]
        for s in excluded[:limit]:
            lines.append(f"  {s}")

    lines += [
        "",
        "=" * width,
        _wrap(
            "This is a ranking under a stated scoring function, not a "
            "recommendation. The weights are judgment calls and the edge "
            "calculation assumes realized volatility persists -- which is false "
            "around earnings and other scheduled events."
        ),
        "=" * width,
    ]
    return "\n".join(lines)


def _wrap(text: str, indent: int = 4, bullet: str = "") -> str:
    import textwrap

    pad = " " * indent
    first = f"{' ' * (indent - 2)}{bullet} " if bullet else pad
    return textwrap.fill(text, width=78, initial_indent=first, subsequent_indent=pad)


def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
    missing = set(DEFAULT_WEIGHTS) - set(weights)
    if missing:
        raise ValueError(f"weights missing keys: {sorted(missing)}")
    unknown = set(weights) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown weight keys: {sorted(unknown)}")
    if any(v < 0 for v in weights.values()):
        raise ValueError("weights cannot be negative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to something positive")
    return {k: v / total for k, v in weights.items()}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
