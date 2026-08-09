"""Support and resistance detection.

The approach, in three steps:

1. Find swing pivots -- bars whose high (or low) is the extreme of a window
   centred on them. These are the points where price actually turned.
2. Cluster pivots that sit within an ATR-scaled tolerance of each other. A
   level is not a line, it is a zone, and its width should scale with the
   name's volatility rather than being a fixed percentage.
3. Score each zone by how much evidence supports it.

Scoring weighs four things, because a level's authority comes from more
than how often it was touched:

- **touches** -- how many separate pivots formed the zone
- **recency** -- a level defended last month binds harder than one from
  two years ago, so touch weight decays exponentially
- **volume** -- pivots that formed on heavy volume mark real inventory
  change hands; light-volume pivots are often just noise
- **flips** -- a zone that acted as both support and resistance at
  different times is a genuine battle line and gets a bonus

The output is deliberately a ranked list of zones with reasons attached,
not a single "the resistance is X" number, because that is not how price
behaves and pretending otherwise makes for confident bad trades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from .indicators import atr
from .ohlcv import Series


@dataclass(frozen=True)
class Pivot:
    """A local turning point in price."""

    index: int
    day: date
    price: float
    kind: str  # "high" or "low"
    volume: float | None = None


@dataclass
class Level:
    """A clustered support/resistance zone."""

    price: float  # volume-and-recency weighted centre of the zone
    low: float  # zone lower bound
    high: float  # zone upper bound
    pivots: list[Pivot] = field(default_factory=list)
    strength: float = 0.0  # 0-100, relative within this analysis
    role: str = "level"  # "support", "resistance", or "pivot" (flipped)

    @property
    def touches(self) -> int:
        return len(self.pivots)

    @property
    def last_touch(self) -> date:
        return max(p.day for p in self.pivots)

    @property
    def width_pct(self) -> float:
        return 0.0 if self.price == 0 else 100.0 * (self.high - self.low) / self.price

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def distance_pct(self, price: float) -> float:
        """Signed distance from ``price`` to the zone centre, in percent."""
        return 0.0 if price == 0 else 100.0 * (self.price - price) / price


def find_pivots(series: Series, window: int = 5) -> list[Pivot]:
    """Locate swing highs and lows using a centred window.

    ``window`` is the number of bars required on *each* side, so a pivot at
    window=5 survived 5 bars either way. Larger windows find fewer, more
    structurally important turns. Bars within ``window`` of either end are
    skipped -- their status is not yet decided by future price.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    highs, lows, volumes, days = series.highs, series.lows, series.volumes, series.dates
    has_vol = series.has_volume
    pivots: list[Pivot] = []
    for i in range(window, len(series) - window):
        left = slice(i - window, i)
        right = slice(i + 1, i + window + 1)
        vol = volumes[i] if has_vol else None
        # Strict on one side, inclusive on the other, so a flat double-top
        # registers once rather than zero times.
        if highs[i] >= max(highs[left]) and highs[i] > max(highs[right]):
            pivots.append(Pivot(i, days[i], highs[i], "high", vol))
        if lows[i] <= min(lows[left]) and lows[i] < min(lows[right]):
            pivots.append(Pivot(i, days[i], lows[i], "low", vol))
    return pivots


def cluster_levels(
    series: Series,
    pivots: list[Pivot],
    tolerance_atr: float = 0.75,
    half_life_bars: float = 120.0,
    min_touches: int = 2,
) -> list[Level]:
    """Group pivots into zones and score them.

    ``tolerance_atr`` sets zone width as a multiple of current ATR -- the
    volatility-aware alternative to a fixed percentage band, which is too
    wide for a utility and far too narrow for a high-beta name.

    ``half_life_bars`` controls recency decay: a touch this many bars back
    counts half as much as one today.
    """
    if not pivots:
        return []
    if min_touches < 1:
        raise ValueError(f"min_touches must be at least 1, got {min_touches}")

    band = _tolerance(series, tolerance_atr)
    last_index = len(series) - 1
    avg_volume = _average_volume(series)

    ordered = sorted(pivots, key=lambda p: p.price)
    groups: list[list[Pivot]] = [[ordered[0]]]
    for pivot in ordered[1:]:
        # Compare against the running group mean so a long drift of pivots
        # cannot chain into one absurdly wide zone.
        centre = sum(p.price for p in groups[-1]) / len(groups[-1])
        if pivot.price - centre <= band:
            groups[-1].append(pivot)
        else:
            groups.append([pivot])

    levels: list[Level] = []
    for group in groups:
        if len(group) < min_touches:
            continue
        weights = []
        for pivot in group:
            recency = 0.5 ** ((last_index - pivot.index) / half_life_bars)
            if pivot.volume is None or avg_volume is None or avg_volume <= 0:
                vol_weight = 1.0
            else:
                # Cap the volume boost so one halt-and-reopen print cannot
                # manufacture a level on its own.
                vol_weight = min(2.5, max(0.4, pivot.volume / avg_volume))
            weights.append(recency * vol_weight)

        total_weight = sum(weights)
        centre = sum(p.price * w for p, w in zip(group, weights)) / total_weight
        prices = [p.price for p in group]
        kinds = {p.kind for p in group}
        levels.append(
            Level(
                price=centre,
                low=min(prices),
                high=max(prices),
                pivots=sorted(group, key=lambda p: p.index),
                strength=total_weight * (1.35 if len(kinds) > 1 else 1.0),
                role="pivot" if len(kinds) > 1 else "level",
            )
        )

    _normalize_strength(levels)
    return sorted(levels, key=lambda lv: lv.price)


def classify_levels(levels: list[Level], price: float) -> list[Level]:
    """Label each zone support/resistance relative to ``price``.

    A zone the price sits inside keeps its "pivot" role -- it is neither,
    and it is the one you are most likely to be chopped up by.
    """
    for level in levels:
        if level.contains(price):
            level.role = "pivot"
        elif level.price > price:
            level.role = "resistance"
        else:
            level.role = "support"
    return levels


def nearest(levels: list[Level], price: float, role: str) -> Level | None:
    """Closest zone of a given role to ``price``."""
    candidates = [lv for lv in levels if lv.role == role]
    if not candidates:
        return None
    return min(candidates, key=lambda lv: abs(lv.price - price))


def most_significant(
    levels: list[Level],
    price: float,
    role: str,
    max_distance_pct: float = 20.0,
) -> Level | None:
    """The zone most worth trading against within ``max_distance_pct``.

    Distinct from ``nearest`` on purpose. A barely-touched level 2% away is a
    worse strike reference than a heavily-defended shelf 8% away, because the
    strong one is where price actually stops. Strength is discounted by
    distance so a monster level halfway across the chart does not win.
    """
    if max_distance_pct <= 0:
        raise ValueError(f"max_distance_pct must be positive, got {max_distance_pct}")
    candidates = [
        lv
        for lv in levels
        if lv.role == role and abs(lv.distance_pct(price)) <= max_distance_pct
    ]
    if not candidates:
        return nearest(levels, price, role)
    return max(
        candidates,
        key=lambda lv: lv.strength / (1.0 + abs(lv.distance_pct(price)) / 10.0),
    )


def pivot_points(series: Series, method: str = "classic") -> dict[str, float]:
    """Floor-trader pivot points from the most recent bar.

    Mechanical intraday reference levels. They matter mainly because a
    large number of desks watch the same arithmetic, which makes them
    partly self-fulfilling on the day.
    """
    bar = series.last
    pivot = (bar.high + bar.low + bar.close) / 3.0
    span = bar.high - bar.low
    if method == "classic":
        return {
            "pivot": pivot,
            "r1": 2 * pivot - bar.low,
            "s1": 2 * pivot - bar.high,
            "r2": pivot + span,
            "s2": pivot - span,
            "r3": bar.high + 2 * (pivot - bar.low),
            "s3": bar.low - 2 * (bar.high - pivot),
        }
    if method == "fibonacci":
        return {
            "pivot": pivot,
            "r1": pivot + 0.382 * span,
            "s1": pivot - 0.382 * span,
            "r2": pivot + 0.618 * span,
            "s2": pivot - 0.618 * span,
            "r3": pivot + span,
            "s3": pivot - span,
        }
    raise ValueError(f"unknown pivot method {method!r}; use 'classic' or 'fibonacci'")


def fibonacci_retracements(series: Series, lookback: int = 120) -> dict[str, float]:
    """Retracement levels across the dominant swing in ``lookback`` bars.

    Direction is inferred from which extreme came last: if the high is more
    recent the swing is up and levels measure a pullback from it.
    """
    window = series.tail(min(lookback, len(series)))
    highs, lows = window.highs, window.lows
    hi_idx = max(range(len(highs)), key=lambda i: highs[i])
    lo_idx = min(range(len(lows)), key=lambda i: lows[i])
    hi, lo = highs[hi_idx], lows[lo_idx]
    span = hi - lo
    swing_up = hi_idx > lo_idx
    ratios = (0.236, 0.382, 0.5, 0.618, 0.786)
    out = {"swing_high": hi, "swing_low": lo, "direction": "up" if swing_up else "down"}
    for r in ratios:
        # Retrace from whichever end the swing finished at.
        out[f"fib_{r:.3f}"] = hi - span * r if swing_up else lo + span * r
    return out


def round_number_levels(price: float, count: int = 3) -> list[float]:
    """Psychologically significant round numbers bracketing ``price``.

    Real levels, not superstition: they attract resting limit orders and
    are where option strikes are listed most densely.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    magnitude = 10 ** math.floor(math.log10(price))
    step = magnitude / 2.0
    base = math.floor(price / step) * step
    out = {round(base + i * step, 10) for i in range(-count, count + 1)}
    return sorted(v for v in out if v > 0)


def _tolerance(series: Series, tolerance_atr: float) -> float:
    """Zone half-width in price units, with a percentage-based fallback."""
    if tolerance_atr <= 0:
        raise ValueError(f"tolerance_atr must be positive, got {tolerance_atr}")
    values = atr(series.highs, series.lows, series.closes, period=14)
    latest = next((v for v in reversed(values) if v is not None), None)
    if latest is None or latest <= 0:
        return series.last.close * 0.01  # too little history for ATR
    return latest * tolerance_atr


def _average_volume(series: Series) -> float | None:
    if not series.has_volume:
        return None
    recent = series.volumes[-60:]
    total = sum(recent)
    return total / len(recent) if total > 0 else None


def _normalize_strength(levels: list[Level]) -> None:
    """Rescale raw weights to 0-100 within this set of levels."""
    if not levels:
        return
    top = max(lv.strength for lv in levels)
    if top <= 0:
        return
    for level in levels:
        level.strength = 100.0 * level.strength / top
