"""Black-Scholes pricing, Greeks, implied volatility, and probabilities.

Everything here is risk-neutral. That matters more than it sounds: the
"probability" numbers below are the market's risk-neutral probabilities, not
real-world odds. They are the right inputs for pricing and hedging, and they
are systematically *not* the frequencies you would observe -- the risk premium
embedded in index puts means the market's implied chance of a large drop
exceeds how often such drops actually happen. Use them to compare strikes and
to price risk, not as a forecast.

Time is measured in calendar days over 365. Some desks use trading days over
252 for volatility and calendar days for decay; if yours does, pass the
convention you want through ``days_per_year`` rather than mixing them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CALL = "call"
PUT = "put"
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


@dataclass(frozen=True)
class Option:
    """A single option contract and its market inputs.

    Rates and volatility are decimals (0.045 == 4.5%). ``dte`` is calendar
    days to expiry. ``dividend_yield`` is the continuous yield on the
    underlying -- for an index use the index yield, for a single name use the
    annualized dividend rate, and for a non-payer leave it at zero.
    """

    spot: float
    strike: float
    dte: float
    volatility: float
    rate: float = 0.04
    dividend_yield: float = 0.0
    kind: str = PUT
    days_per_year: float = 365.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.dte < 0:
            raise ValueError(f"dte cannot be negative, got {self.dte}")
        if self.volatility < 0:
            raise ValueError(f"volatility cannot be negative, got {self.volatility}")
        if self.days_per_year <= 0:
            raise ValueError(f"days_per_year must be positive, got {self.days_per_year}")
        if self.kind not in (CALL, PUT):
            raise ValueError(f"kind must be {CALL!r} or {PUT!r}, got {self.kind!r}")

    @property
    def t(self) -> float:
        """Time to expiry in years."""
        return self.dte / self.days_per_year

    @property
    def is_call(self) -> bool:
        return self.kind == CALL

    @property
    def moneyness(self) -> float:
        """Percent the strike sits away from spot. Positive means above spot."""
        return 100.0 * (self.strike - self.spot) / self.spot

    @property
    def intrinsic(self) -> float:
        if self.is_call:
            return max(0.0, self.spot - self.strike)
        return max(0.0, self.strike - self.spot)

    def replace(self, **changes) -> "Option":
        """Copy with fields overridden."""
        fields = {
            "spot": self.spot,
            "strike": self.strike,
            "dte": self.dte,
            "volatility": self.volatility,
            "rate": self.rate,
            "dividend_yield": self.dividend_yield,
            "kind": self.kind,
            "days_per_year": self.days_per_year,
        }
        fields.update(changes)
        return Option(**fields)

    def with_volatility(self, volatility: float) -> "Option":
        return self.replace(volatility=volatility)

    def with_strike(self, strike: float) -> "Option":
        return self.replace(strike=strike)


@dataclass(frozen=True)
class Greeks:
    """Greeks in raw (per-unit) terms, with desk-scaled views alongside.

    The raw values are per one unit of the underlying and per one full unit
    of volatility or time. Nobody trades in those units, so the scaled
    properties below convert to what a desk actually quotes.
    """

    delta: float
    gamma: float
    vega: float   # per 1.00 (100 points) of volatility
    theta: float  # per year
    rho: float    # per 1.00 of rate
    vanna: float  # d(delta)/d(vol)
    charm: float  # d(delta)/d(time), per year

    @property
    def vega_per_point(self) -> float:
        """Dollar change per 1 volatility point (1%) -- how vega is quoted."""
        return self.vega / 100.0

    @property
    def theta_per_day(self) -> float:
        """Dollar decay per calendar day -- how theta is quoted."""
        return self.theta / 365.0

    @property
    def rho_per_point(self) -> float:
        """Dollar change per 1% move in rates."""
        return self.rho / 100.0

    def per_contract(self, multiplier: float = 100.0) -> "Greeks":
        """Scale to one contract. Equity options carry 100 shares."""
        return Greeks(
            delta=self.delta * multiplier,
            gamma=self.gamma * multiplier,
            vega=self.vega * multiplier,
            theta=self.theta * multiplier,
            rho=self.rho * multiplier,
            vanna=self.vanna * multiplier,
            charm=self.charm * multiplier,
        )


def d1_d2(option: Option) -> tuple[float, float]:
    """The two Black-Scholes terms every formula here shares."""
    t = option.t
    sigma = option.volatility
    if t <= 0 or sigma <= 0:
        raise ValueError(
            "d1/d2 are undefined at zero time or zero volatility; the option "
            "is worth its intrinsic value and has no Greeks to speak of"
        )
    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (
        math.log(option.spot / option.strike)
        + (option.rate - option.dividend_yield + 0.5 * sigma * sigma) * t
    ) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(option: Option) -> float:
    """Black-Scholes-Merton value with a continuous dividend yield."""
    if option.t <= 0 or option.volatility <= 0:
        # Degenerate but legitimate: an expiring or zero-vol option is worth
        # its discounted intrinsic value, not an error.
        return option.intrinsic
    d1, d2 = d1_d2(option)
    discount = math.exp(-option.rate * option.t)
    carry = math.exp(-option.dividend_yield * option.t)
    if option.is_call:
        return option.spot * carry * norm_cdf(d1) - option.strike * discount * norm_cdf(d2)
    return option.strike * discount * norm_cdf(-d2) - option.spot * carry * norm_cdf(-d1)


def greeks(option: Option) -> Greeks:
    """All Greeks for one unit of the underlying.

    Multiply by the contract multiplier (100 for equity options) via
    ``Greeks.per_contract()`` to get position-level numbers.
    """
    if option.t <= 0 or option.volatility <= 0:
        # At expiry delta is a step function and everything else collapses.
        itm = option.intrinsic > 0
        delta = (1.0 if option.is_call else -1.0) if itm else 0.0
        return Greeks(delta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    t = option.t
    sigma = option.volatility
    sqrt_t = math.sqrt(t)
    d1, d2 = d1_d2(option)
    discount = math.exp(-option.rate * t)
    carry = math.exp(-option.dividend_yield * t)
    pdf_d1 = norm_pdf(d1)

    gamma = carry * pdf_d1 / (option.spot * sigma * sqrt_t)
    vega = option.spot * carry * pdf_d1 * sqrt_t
    decay = -option.spot * carry * pdf_d1 * sigma / (2.0 * sqrt_t)
    vanna = -carry * pdf_d1 * d2 / sigma

    if option.is_call:
        delta = carry * norm_cdf(d1)
        theta = (
            decay
            - option.rate * option.strike * discount * norm_cdf(d2)
            + option.dividend_yield * option.spot * carry * norm_cdf(d1)
        )
        rho = option.strike * t * discount * norm_cdf(d2)
        charm = carry * (
            option.dividend_yield * norm_cdf(d1)
            - pdf_d1
            * (2 * (option.rate - option.dividend_yield) * t - d2 * sigma * sqrt_t)
            / (2 * t * sigma * sqrt_t)
        )
    else:
        delta = -carry * norm_cdf(-d1)
        theta = (
            decay
            + option.rate * option.strike * discount * norm_cdf(-d2)
            - option.dividend_yield * option.spot * carry * norm_cdf(-d1)
        )
        rho = -option.strike * t * discount * norm_cdf(-d2)
        charm = carry * (
            -option.dividend_yield * norm_cdf(-d1)
            - pdf_d1
            * (2 * (option.rate - option.dividend_yield) * t - d2 * sigma * sqrt_t)
            / (2 * t * sigma * sqrt_t)
        )

    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        charm=charm,
    )


def implied_volatility(
    option: Option,
    market_price: float,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Back out the volatility that reproduces ``market_price``.

    Newton-Raphson on vega, falling back to bisection when Newton wanders --
    which it does for deep in- or out-of-the-money strikes where vega is
    nearly zero and the derivative carries no information.

    The ``volatility`` field on the passed option is used only as a starting
    guess; pass anything positive.
    """
    if market_price < 0:
        raise ValueError(f"market_price cannot be negative, got {market_price}")
    if option.t <= 0:
        raise ValueError("cannot imply volatility from an expired option")

    # No volatility can produce a price outside these bounds, so failing here
    # means bad data -- a stale quote, a wrong rate, or a mismatched strike.
    lower_bound = _no_arbitrage_floor(option)
    upper_bound = _no_arbitrage_ceiling(option)
    if market_price < lower_bound - 1e-9:
        raise ValueError(
            f"price {market_price:.4f} is below intrinsic {lower_bound:.4f}; "
            "no volatility can produce it"
        )
    if market_price > upper_bound + 1e-9:
        raise ValueError(
            f"price {market_price:.4f} exceeds the no-arbitrage ceiling "
            f"{upper_bound:.4f}; no volatility can produce it"
        )

    result = _newton_volatility(option, market_price, tolerance, max_iterations)
    if result is None:
        result = _bisect_volatility(option, market_price, tolerance, max_iterations)

    # Identifiability is checked once, on whichever method produced the answer.
    # Both methods can converge on price while the volatility remains pinned
    # down only loosely -- price tolerance is not volatility tolerance.
    _require_identifiable(option, result)
    return result


def _newton_volatility(
    option: Option, market_price: float, tolerance: float, max_iterations: int
) -> float | None:
    """Newton-Raphson on vega. Returns ``None`` when it fails to converge.

    Convergence requires the *step* to be small, not just the price error.
    A near-zero vega makes large volatility changes almost free in price
    terms, so a price-only test declares victory while the answer is still
    wrong by tens of volatility points.
    """
    guess = option.volatility if option.volatility > 0 else 0.3
    for _ in range(max_iterations):
        trial = option.with_volatility(guess)
        diff = price(trial) - market_price
        vega = greeks(trial).vega
        if vega < 1e-10:
            return None  # derivative carries no information
        step = diff / vega
        guess -= step
        if not (1e-9 < guess < 10.0):
            return None  # wandered outside the plausible range
        if abs(step) < 1e-12 and abs(diff) < tolerance:
            return guess
    return None


def _require_identifiable(option: Option, volatility: float) -> None:
    """Refuse to report an IV the price cannot actually pin down.

    Where vega is negligible -- far from the money, or nearly expired -- a
    wide band of volatilities all reprice to the same number within floating
    point. Returning a confident value there is worse than refusing, because
    it flows into IV rank and strike comparisons as though it meant something.
    """
    if greeks(option.with_volatility(volatility)).vega < 1e-6:
        raise ValueError(
            f"implied volatility is not identifiable for this strike: vega is "
            f"effectively zero near {volatility:.4f}, so the price does not pin "
            f"down a volatility. The strike is too far from the money, or too "
            f"close to expiry, for an IV to carry information."
        )


def _bisect_volatility(
    option: Option, market_price: float, tolerance: float, max_iterations: int
) -> float:
    """Bracketed search -- slower than Newton but it always converges.

    Converges on the *volatility* interval, not on price. Stopping at the
    first vol whose price matches within tolerance looks like it saves work,
    but for a strike with negligible vega a wide band of volatilities all
    reprice within any sane tolerance, and the search would return an
    arbitrary point in that band while appearing to have succeeded.
    """
    low, high = 1e-9, 10.0
    if price(option.with_volatility(high)) < market_price:
        raise ValueError(
            f"price {market_price:.4f} implies volatility above 1000%, which "
            "almost always means the inputs are wrong rather than the market"
        )
    for _ in range(max_iterations * 4):
        if (high - low) < 1e-12:
            break
        mid = 0.5 * (low + high)
        if price(option.with_volatility(mid)) < market_price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _no_arbitrage_floor(option: Option) -> float:
    """Lowest price consistent with no arbitrage (the forward intrinsic)."""
    discount = math.exp(-option.rate * option.t)
    carry = math.exp(-option.dividend_yield * option.t)
    if option.is_call:
        return max(0.0, option.spot * carry - option.strike * discount)
    return max(0.0, option.strike * discount - option.spot * carry)


def _no_arbitrage_ceiling(option: Option) -> float:
    """Highest price consistent with no arbitrage."""
    if option.is_call:
        return option.spot * math.exp(-option.dividend_yield * option.t)
    return option.strike * math.exp(-option.rate * option.t)


# ---------------------------------------------------------------------------
# probabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probabilities:
    """Risk-neutral chances attached to one strike.

    ``prob_itm`` is the chance of finishing in the money. ``prob_touch`` is
    the chance of trading through the strike at any point before expiry --
    roughly double ``prob_itm``, and the number that matters when you would
    be forced to act on a breach rather than ride it to expiry.

    ``prob_profit`` uses the breakeven, not the strike. For a short option
    that is the more honest number: a put sold for 2.00 against a 100 strike
    does not lose money until 98.
    """

    prob_itm: float
    prob_otm: float
    prob_touch: float
    prob_profit: float | None
    expected_move: float
    breakeven: float | None

    @property
    def prob_itm_pct(self) -> float:
        return 100.0 * self.prob_itm

    @property
    def prob_otm_pct(self) -> float:
        return 100.0 * self.prob_otm

    @property
    def prob_touch_pct(self) -> float:
        return 100.0 * self.prob_touch

    @property
    def prob_profit_pct(self) -> float | None:
        return None if self.prob_profit is None else 100.0 * self.prob_profit


def probabilities(option: Option, premium: float | None = None) -> Probabilities:
    """Chances of finishing ITM, touching the strike, and turning a profit.

    ``premium`` is the credit received (for a short) or debit paid (for a
    long). Supplying it enables ``prob_profit``; without it that field is
    ``None`` rather than silently assuming zero cost.
    """
    if premium is not None and premium < 0:
        raise ValueError(f"premium cannot be negative, got {premium}")

    t = option.t
    if t <= 0 or option.volatility <= 0:
        itm = 1.0 if option.intrinsic > 0 else 0.0
        return Probabilities(
            prob_itm=itm,
            prob_otm=1.0 - itm,
            prob_touch=itm,
            prob_profit=None,
            expected_move=0.0,
            breakeven=None,
        )

    _, d2 = d1_d2(option)
    prob_itm = norm_cdf(d2) if option.is_call else norm_cdf(-d2)
    # An already-in-the-money strike has been touched by definition. Running
    # the barrier formula here would answer a different question -- the odds of
    # price travelling back out to the strike -- and could return a number
    # below prob_itm, which is nonsense for a touch probability.
    prob_touch = (
        1.0
        if option.intrinsic > 0
        else probability_of_touch(option, option.strike)
    )
    expected_move = option.spot * option.volatility * math.sqrt(t)

    breakeven = None
    prob_profit = None
    if premium is not None:
        breakeven = (
            option.strike + premium if option.is_call else option.strike - premium
        )
        if breakeven > 0:
            _, be_d2 = d1_d2(option.with_strike(breakeven))
            # Chance of finishing beyond the breakeven -- i.e. the seller keeps
            # something. The complement of finishing past it.
            prob_profit = norm_cdf(-be_d2) if option.is_call else norm_cdf(be_d2)

    return Probabilities(
        prob_itm=prob_itm,
        prob_otm=1.0 - prob_itm,
        prob_touch=prob_touch,
        prob_profit=prob_profit,
        expected_move=expected_move,
        breakeven=breakeven,
    )


def probability_of_touch(option: Option, barrier: float) -> float:
    """Chance price trades through ``barrier`` at any point before expiry.

    The exact first-passage probability for geometric Brownian motion, not
    the "double the delta" shortcut. That shortcut is only right when drift
    is negligible; over longer horizons or with a meaningful rate-minus-
    dividend carry it drifts off, always in the direction of understating
    the upside barrier.
    """
    if barrier <= 0:
        raise ValueError(f"barrier must be positive, got {barrier}")
    t = option.t
    sigma = option.volatility
    if t <= 0 or sigma <= 0:
        return 0.0
    if barrier == option.spot:
        return 1.0  # already at the barrier

    nu = option.rate - option.dividend_yield - 0.5 * sigma * sigma
    b = math.log(barrier / option.spot)
    vol_sqrt_t = sigma * math.sqrt(t)
    # The reflection term can overflow for extreme barriers; at that point the
    # probability is saturated anyway, so clamping loses nothing real.
    exponent = max(-700.0, min(700.0, 2.0 * nu * b / (sigma * sigma)))
    reflection = math.exp(exponent)

    if b > 0:
        first = norm_cdf((-b + nu * t) / vol_sqrt_t)
        second = reflection * norm_cdf((-b - nu * t) / vol_sqrt_t)
    else:
        first = norm_cdf((b - nu * t) / vol_sqrt_t)
        second = reflection * norm_cdf((b + nu * t) / vol_sqrt_t)
    return max(0.0, min(1.0, first + second))


def expected_move(option: Option, standard_deviations: float = 1.0) -> float:
    """One-sigma implied move by expiry, in price units.

    The market's own estimate of the range. Compare it against the distance
    to support and resistance: a short strike inside the expected move is
    being sold into the range the market already expects price to cover.
    """
    if standard_deviations <= 0:
        raise ValueError(
            f"standard_deviations must be positive, got {standard_deviations}"
        )
    if option.t <= 0:
        return 0.0
    return standard_deviations * option.spot * option.volatility * math.sqrt(option.t)


def delta_to_strike(option: Option, target_delta: float) -> float:
    """Find the strike whose delta matches ``target_delta``.

    How strikes are actually chosen on a desk -- "sell the 16 delta put" is a
    statement about risk, and it stays comparable across names and expiries
    in a way that "sell the 95 strike" never does. Pass the magnitude; the
    sign is taken from the option kind.
    """
    magnitude = abs(target_delta)
    if not 0.0 < magnitude < 1.0:
        raise ValueError(f"target_delta must be between 0 and 1, got {target_delta}")
    if option.t <= 0 or option.volatility <= 0:
        raise ValueError("delta-to-strike needs positive time and volatility")

    low, high = option.spot * 0.01, option.spot * 5.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        current = abs(greeks(option.with_strike(mid)).delta)
        if abs(current - magnitude) < 1e-8 or (high - low) < 1e-9:
            return mid
        # Call delta falls as the strike rises; put delta magnitude rises.
        if (current > magnitude) == option.is_call:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
