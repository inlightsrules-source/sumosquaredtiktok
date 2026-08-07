"""Implied volatility context: IV rank, IV percentile, and IV vs realized.

A raw IV number is nearly useless on its own. "IV is 28" means one thing for
a utility and something else entirely for a biotech; the only way to read it
is against the name's own history. That is what rank and percentile do.

**IV Rank vs IV Percentile -- they are not the same, and the gap matters.**

    IV Rank       = (IV - min) / (max - min)     over the lookback
    IV Percentile = share of days IV closed below the current level

Rank is a position within the observed *range*. Percentile is a position
within the observed *distribution*. One earnings-day spike sets the high for
a whole year, and every reading afterwards gets measured against that
outlier, so rank can read 20 while the name has actually spent most of the
year cheaper than it is right now. Percentile would read 70 in that case.

Percentile is the more robust of the two and should usually drive decisions.
Rank is reported because it is what most retail platforms display, and
knowing both tells you when the range is being distorted by a single event.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

from .indicators import percentile_rank, realized_volatility
from .ohlcv import Series

TRADING_DAYS_PER_YEAR = 252


@dataclass
class VolContext:
    """Where implied volatility sits relative to its history and to realized."""

    implied: float             # current IV, as a percent (28.0 == 28%)
    iv_rank: float | None      # 0-100 position within the observed range
    iv_percentile: float | None  # 0-100 share of days below current
    realized: float | None     # trailing realized vol, same units
    premium: float | None      # IV minus realized, in vol points
    premium_ratio: float | None  # IV / realized
    observations: int
    interpretation: str = ""
    edge: str = ""  # "sell premium", "buy premium", or "no edge"

    @property
    def rank_percentile_gap(self) -> float | None:
        """How far rank and percentile disagree.

        A wide gap means the lookback range is being set by outliers, and
        rank is the less trustworthy of the two.
        """
        if self.iv_rank is None or self.iv_percentile is None:
            return None
        return self.iv_percentile - self.iv_rank


def iv_rank(history: Sequence[float], current: float | None = None) -> float | None:
    """Position within the observed high-low range, 0-100.

    Returns ``None`` for an empty history, and 50.0 when the range is
    degenerate (every observation identical) -- there is no meaningful
    position within a range of zero width, and 50 states "no information"
    more honestly than 0 or 100 would.
    """
    values = [v for v in history if v is not None]
    if not values:
        return None
    level = values[-1] if current is None else current
    low, high = min(values), max(values)
    if high == low:
        return 50.0
    return 100.0 * (level - low) / (high - low)


def iv_percentile(history: Sequence[float], current: float | None = None) -> float | None:
    """Share of the history that sits below ``current``, 0-100.

    Ties count half, so an unchanged IV against a flat history reads 50
    rather than 0 or 100.
    """
    return percentile_rank(list(history), current)


def build_context(
    implied: float,
    iv_history: Sequence[float] | None = None,
    series: Series | None = None,
    realized_window: int = 20,
) -> VolContext:
    """Assemble the full volatility picture.

    ``implied`` and ``iv_history`` are percents (28.0 == 28%), matching how
    realized volatility is reported elsewhere in this package. ``series``
    supplies the price history used to compute realized volatility -- the
    number implied vol has to be measured against.
    """
    if implied < 0:
        raise ValueError(f"implied volatility cannot be negative, got {implied}")

    history = [v for v in (iv_history or []) if v is not None]
    rank = iv_rank(history, implied) if history else None
    pct = iv_percentile(history, implied) if history else None

    realized = None
    if series is not None:
        values = realized_volatility(series.closes, realized_window)
        realized = next((v for v in reversed(values) if v is not None), None)

    premium = None if realized is None else implied - realized
    ratio = None if not realized else implied / realized

    context = VolContext(
        implied=implied,
        iv_rank=rank,
        iv_percentile=pct,
        realized=realized,
        premium=premium,
        premium_ratio=ratio,
        observations=len(history),
    )
    context.edge = _classify_edge(context)
    context.interpretation = _interpret(context)
    return context


def _classify_edge(ctx: VolContext) -> str:
    """Decide whether premium is worth selling, buying, or neither.

    The IV-to-realized spread carries most of the weight. Selling options is
    selling insurance: it pays when the premium collected exceeds the
    movement actually delivered. A high IV percentile with IV *below*
    realized is not an edge, it is a name that has become genuinely more
    dangerous, and the options are still not paying enough for it.
    """
    if ctx.premium_ratio is not None:
        if ctx.premium_ratio < 1.0:
            return "buy premium"
        if ctx.premium_ratio > 1.25 and (ctx.iv_percentile or 50) > 50:
            return "sell premium"
        if ctx.premium_ratio > 1.4:
            return "sell premium"
        if ctx.premium_ratio < 1.1:
            return "no edge"
    if ctx.iv_percentile is not None:
        if ctx.iv_percentile > 75:
            return "sell premium"
        if ctx.iv_percentile < 25:
            return "buy premium"
    return "no edge"


def _ordinal(value: float) -> str:
    """Format a percentile with the right English suffix (1st, 2nd, 72nd)."""
    n = int(round(value))
    # 11th through 13th are the exceptions that break the last-digit rule.
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _interpret(ctx: VolContext) -> str:
    parts: list[str] = []

    if ctx.iv_percentile is None:
        parts.append(
            f"IV is {ctx.implied:.1f}% with no history supplied, so there is no "
            "way to say whether that is expensive for this name. Feed in an IV "
            "history before drawing conclusions from the level."
        )
    else:
        parts.append(
            f"IV {ctx.implied:.1f}% sits at the {_ordinal(ctx.iv_percentile)} "
            f"percentile of its last {ctx.observations} observations"
            + (f" (IV rank {ctx.iv_rank:.0f})." if ctx.iv_rank is not None else ".")
        )
        gap = ctx.rank_percentile_gap
        if gap is not None and abs(gap) > 20:
            higher, lower = ("percentile", "rank") if gap > 0 else ("rank", "percentile")
            parts.append(
                f"Rank and percentile disagree by {abs(gap):.0f} points, with "
                f"{higher} the higher of the two. That gap means the lookback "
                f"range is set by a few outlier days, so {lower} is distorted -- "
                f"trust the percentile here."
            )

    if ctx.realized is None:
        parts.append(
            "No price series was supplied, so IV could not be compared against "
            "realized volatility -- the comparison that actually decides whether "
            "premium is rich."
        )
    else:
        parts.append(
            f"Realized volatility over the same window is {ctx.realized:.1f}%, "
            f"putting IV at a {ctx.premium:+.1f} point "
            f"{'premium' if ctx.premium >= 0 else 'discount'} "
            f"({ctx.premium_ratio:.2f}x)."
        )
        if ctx.premium_ratio < 1.0:
            parts.append(
                "Options are pricing less movement than the stock has actually "
                "been delivering. Selling premium here is taking on gap risk at "
                "a discount to what it has recently cost -- the wrong side."
            )
        elif ctx.premium_ratio > 1.4:
            parts.append(
                "Options are pricing substantially more movement than the stock "
                "has delivered. That spread is the premium seller's edge, though "
                "check for a pending catalyst first: a wide spread ahead of "
                "earnings is the market pricing a known event, not mispricing it."
            )
        elif ctx.premium_ratio > 1.15:
            parts.append(
                "A modest premium over realized -- the normal state of affairs, "
                "since sellers are compensated for carrying gap risk. Real but "
                "not exceptional."
            )
        else:
            parts.append(
                "IV and realized are close enough that selling premium is barely "
                "compensated for the tail risk it assumes."
            )

    return " ".join(parts)


def implied_vol_history_from_prices(
    prices: Sequence[float], window: int = 20
) -> list[float]:
    """Realized-vol proxy for an IV history, when no IV history is available.

    A fallback, not a substitute. Realized volatility is backward-looking and
    systematically sits below implied, so rank and percentile computed from it
    will read too high. Use a real IV history whenever you can get one.
    """
    values = realized_volatility(list(prices), window)
    return [v for v in values if v is not None]


def term_structure_slope(
    near_iv: float, far_iv: float, near_dte: float, far_dte: float
) -> tuple[float, str]:
    """Slope between two expiries, and what it implies.

    Contango (far above near) is the resting state. Backwardation -- near
    above far -- means the market expects something soon, and short-dated
    premium is rich for a reason rather than by mistake.
    """
    if near_dte <= 0 or far_dte <= 0:
        raise ValueError("both expiries must have positive time remaining")
    if far_dte <= near_dte:
        raise ValueError(
            f"far expiry ({far_dte}d) must be later than near ({near_dte}d)"
        )
    slope = far_iv - near_iv
    if slope < -2.0:
        note = (
            "Backwardation: near-dated IV exceeds far-dated. The market is "
            "pricing an event inside the near expiry. Short-dated premium looks "
            "rich because it is carrying known event risk -- selling it is "
            "selling the event, not selling an edge."
        )
    elif slope > 2.0:
        note = (
            "Contango, the normal shape. Far-dated options carry more "
            "volatility, which favours calendar structures that sell the near "
            "expiry and own the far one."
        )
    else:
        note = "The term structure is roughly flat; no timing signal from shape."
    return slope, note


def annualize(daily_vol: float, days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Scale a daily volatility to annual by the square root of time."""
    if daily_vol < 0:
        raise ValueError(f"daily_vol cannot be negative, got {daily_vol}")
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    return daily_vol * (days**0.5)


def average_iv(*values: float) -> float:
    """Mean of several IV readings -- e.g. straddle legs."""
    supplied = [v for v in values if v is not None]
    if not supplied:
        raise ValueError("need at least one IV value")
    return fmean(supplied)
