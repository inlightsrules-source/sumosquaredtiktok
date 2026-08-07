"""Technical indicators, pure stdlib.

Every function returns a list aligned to the input series: positions with
insufficient lookback hold ``None`` rather than being dropped. Keeping the
alignment means you can zip any two indicators together without tracking
offsets, which is where most hand-rolled TA code goes wrong.

Wilder's smoothing (RSI, ATR, ADX) is seeded with a simple average of the
first ``period`` values, matching the original formulation and the major
charting packages. Other implementations seed differently, so absolute
values can differ slightly from another vendor's -- the crossings do not.
"""

from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Sequence

Number = float | None


# ---------------------------------------------------------------------------
# moving averages
# ---------------------------------------------------------------------------


def sma(values: Sequence[float], period: int) -> list[Number]:
    """Simple moving average."""
    _check_period(period)
    out: list[Number] = [None] * len(values)
    if len(values) < period:
        return out
    window = sum(values[:period])
    out[period - 1] = window / period
    for i in range(period, len(values)):
        window += values[i] - values[i - period]
        out[i] = window / period
    return out


def ema(values: Sequence[float], period: int) -> list[Number]:
    """Exponential moving average, seeded with the first SMA."""
    _check_period(period)
    out: list[Number] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    prev = fmean(values[:period])
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def wilder_smooth(values: Sequence[float], period: int) -> list[Number]:
    """Wilder's smoothing -- an EMA with alpha = 1/period."""
    _check_period(period)
    out: list[Number] = [None] * len(values)
    if len(values) < period:
        return out
    prev = fmean(values[:period])
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = prev + (values[i] - prev) / period
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------


def rsi(closes: Sequence[float], period: int = 14) -> list[Number]:
    """Relative Strength Index (Wilder).

    Reads 0-100. Conventionally >70 overbought, <30 oversold -- but in a
    strong trend RSI parks in the extreme for weeks, so treat a high reading
    as evidence of trend strength first and exhaustion only second.
    """
    _check_period(period)
    out: list[Number] = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains, losses = [], []
    for prev, cur in zip(closes, closes[1:]):
        change = cur - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = fmean(gains[:period])
    avg_loss = fmean(losses[:period])
    out[period] = _rsi_from(avg_gain, avg_loss)
    for i in range(period, len(gains)):
        avg_gain = avg_gain + (gains[i] - avg_gain) / period
        avg_loss = avg_loss + (losses[i] - avg_loss) / period
        out[i + 1] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def macd(
    closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, list[Number]]:
    """MACD line, signal line, and histogram.

    The histogram is the tell: it turns before the lines cross, so it leads
    the classic crossover signal by a bar or two.
    """
    if fast >= slow:
        raise ValueError(f"fast period {fast} must be below slow period {slow}")
    fast_line, slow_line = ema(closes, fast), ema(closes, slow)
    line: list[Number] = [
        f - s if f is not None and s is not None else None
        for f, s in zip(fast_line, slow_line)
    ]
    # The signal EMA runs over the defined stretch of the MACD line only.
    defined = [v for v in line if v is not None]
    offset = len(line) - len(defined)
    sig_defined = ema(defined, signal)
    sig: list[Number] = [None] * offset + list(sig_defined)
    hist: list[Number] = [
        m - s if m is not None and s is not None else None for m, s in zip(line, sig)
    ]
    return {"macd": line, "signal": sig, "histogram": hist}


def stochastic(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict[str, list[Number]]:
    """Stochastic oscillator (%K and %D), showing close position in range."""
    _check_period(period)
    _same_length(highs=highs, lows=lows, closes=closes)
    raw: list[Number] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        hi = max(highs[i - period + 1 : i + 1])
        lo = min(lows[i - period + 1 : i + 1])
        span = hi - lo
        # A flat window has no range to position within; 50 is the neutral read.
        raw[i] = 50.0 if span == 0 else 100.0 * (closes[i] - lo) / span
    k = _smooth_optional(raw, smooth_k)
    d = _smooth_optional(k, smooth_d)
    return {"k": k, "d": d}


def rate_of_change(values: Sequence[float], period: int = 20) -> list[Number]:
    """Percent change over ``period`` bars."""
    _check_period(period)
    out: list[Number] = [None] * len(values)
    for i in range(period, len(values)):
        base = values[i - period]
        if base != 0:
            out[i] = 100.0 * (values[i] - base) / base
    return out


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------


def true_range(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[Number]:
    """True range -- the widest of today's span and the two gap measures."""
    _same_length(highs=highs, lows=lows, closes=closes)
    out: list[Number] = [None] * len(closes)
    if not closes:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        prev_close = closes[i - 1]
        out[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
    return out


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[Number]:
    """Average True Range -- volatility in price units, not percent.

    This is the unit to size stops and strike distances in: a 1.5x ATR move
    is ordinary noise, so a stop inside that band is a donation.
    """
    tr = [v for v in true_range(highs, lows, closes) if v is not None]
    smoothed = wilder_smooth(tr, period)
    return list(smoothed)


def bollinger(
    closes: Sequence[float], period: int = 20, num_std: float = 2.0
) -> dict[str, list[Number]]:
    """Bollinger bands plus %B and bandwidth.

    ``bandwidth`` is the squeeze detector -- when it drops to the low end of
    its own recent range, the market is coiling and an expansion follows.
    That is the single most useful volatility signal for timing long premium.
    """
    _check_period(period)
    if num_std <= 0:
        raise ValueError(f"num_std must be positive, got {num_std}")
    mid = sma(closes, period)
    upper: list[Number] = [None] * len(closes)
    lower: list[Number] = [None] * len(closes)
    pct_b: list[Number] = [None] * len(closes)
    width: list[Number] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        centre = mid[i]
        assert centre is not None
        sd = pstdev(closes[i - period + 1 : i + 1])
        upper[i] = centre + num_std * sd
        lower[i] = centre - num_std * sd
        span = upper[i] - lower[i]
        pct_b[i] = 0.5 if span == 0 else (closes[i] - lower[i]) / span
        width[i] = 0.0 if centre == 0 else 100.0 * span / centre
    return {
        "middle": mid,
        "upper": upper,
        "lower": lower,
        "percent_b": pct_b,
        "bandwidth": width,
    }


def realized_volatility(
    closes: Sequence[float], period: int = 20, annualize: int = 252
) -> list[Number]:
    """Annualized close-to-close realized volatility, in percent.

    This is the number to hold implied vol up against. If IV sits well above
    a stable realized, premium is rich and selling is paid; if IV is near or
    below realized, you are being asked to sell risk at cost.
    """
    _check_period(period)
    out: list[Number] = [None] * len(closes)
    if len(closes) <= period:
        return out
    rets: list[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev <= 0:
            raise ValueError(f"non-positive close {prev}")
        rets.append(math.log(cur / prev))
    scale = math.sqrt(annualize) * 100.0
    for i in range(period - 1, len(rets)):
        out[i + 1] = pstdev(rets[i - period + 1 : i + 1]) * scale
    return out


def parkinson_volatility(
    highs: Sequence[float], lows: Sequence[float], period: int = 20, annualize: int = 252
) -> list[Number]:
    """Parkinson range-based volatility, annualized percent.

    Uses the high-low range rather than closes, so it is roughly five times
    more efficient per observation. It ignores gaps, though -- when this
    reads far below close-to-close realized, the move is happening overnight.
    """
    _check_period(period)
    _same_length(highs=highs, lows=lows)
    out: list[Number] = [None] * len(highs)
    factor = 1.0 / (4.0 * math.log(2.0))
    squares: list[float] = []
    for hi, lo in zip(highs, lows):
        if lo <= 0:
            raise ValueError(f"non-positive low {lo}")
        squares.append(factor * math.log(hi / lo) ** 2)
    scale = math.sqrt(annualize) * 100.0
    for i in range(period - 1, len(squares)):
        out[i] = math.sqrt(fmean(squares[i - period + 1 : i + 1])) * scale
    return out


# ---------------------------------------------------------------------------
# trend strength / direction
# ---------------------------------------------------------------------------


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> dict[str, list[Number]]:
    """Average Directional Index with +DI and -DI.

    ADX measures trend *strength* and says nothing about direction -- the DI
    pair carries that. Below 20 the market is ranging and trend-following
    signals are noise; above 25 a trend is real and mean-reversion trades
    are the ones that get run over.
    """
    _check_period(period)
    _same_length(highs=highs, lows=lows, closes=closes)
    n = len(closes)
    empty: list[Number] = [None] * n
    if n <= period:
        return {"adx": empty, "plus_di": list(empty), "minus_di": list(empty)}

    tr = [v for v in true_range(highs, lows, closes) if v is not None][1:]
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        # Only the larger directional move counts, and only if it is positive.
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)

    tr_s = wilder_smooth(tr, period)
    plus_s = wilder_smooth(plus_dm, period)
    minus_s = wilder_smooth(minus_dm, period)

    plus_di: list[Number] = [None] * n
    minus_di: list[Number] = [None] * n
    dx_vals: list[float] = []
    dx_index: list[int] = []
    for i, (t, p, m) in enumerate(zip(tr_s, plus_s, minus_s)):
        if t is None or p is None or m is None or t == 0:
            continue
        pdi = 100.0 * p / t
        mdi = 100.0 * m / t
        pos = i + 1  # offset back onto the original series
        plus_di[pos] = pdi
        minus_di[pos] = mdi
        total = pdi + mdi
        if total > 0:
            dx_vals.append(100.0 * abs(pdi - mdi) / total)
            dx_index.append(pos)

    adx_line: list[Number] = [None] * n
    smoothed_dx = wilder_smooth(dx_vals, period)
    for pos, value in zip(dx_index, smoothed_dx):
        adx_line[pos] = value
    return {"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di}


def linear_regression(values: Sequence[float]) -> tuple[float, float, float]:
    """Least-squares fit over the whole input.

    Returns ``(slope_per_bar, intercept, r_squared)``. R² is the honesty
    check on the slope: a steep slope with R² of 0.2 is a line drawn through
    a cloud, and reporting its angle as "the trend" is how you end up short
    gamma into a chop.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"need at least 2 points, got {n}")
    xs = list(range(n))
    mean_x = fmean(xs)
    mean_y = fmean(values)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("degenerate x-values")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    syy = sum((y - mean_y) ** 2 for y in values)
    r_squared = 1.0 if syy == 0 else max(0.0, min(1.0, (sxy * sxy) / (sxx * syy)))
    return slope, intercept, r_squared


def donchian(
    highs: Sequence[float], lows: Sequence[float], period: int = 20
) -> dict[str, list[Number]]:
    """Donchian channel -- the rolling extreme high and low."""
    _check_period(period)
    _same_length(highs=highs, lows=lows)
    n = len(highs)
    upper: list[Number] = [None] * n
    lower: list[Number] = [None] * n
    mid: list[Number] = [None] * n
    for i in range(period - 1, n):
        upper[i] = max(highs[i - period + 1 : i + 1])
        lower[i] = min(lows[i - period + 1 : i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    return {"upper": upper, "lower": lower, "middle": mid}


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------


def on_balance_volume(
    closes: Sequence[float], volumes: Sequence[float]
) -> list[Number]:
    """On-balance volume -- a running signed total of volume.

    Only the shape matters; the level is an artifact of where you started.
    OBV making new highs while price does not is accumulation.
    """
    _same_length(closes=closes, volumes=volumes)
    out: list[Number] = [None] * len(closes)
    if not closes:
        return out
    total = 0.0
    out[0] = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            total += volumes[i]
        elif closes[i] < closes[i - 1]:
            total -= volumes[i]
        out[i] = total
    return out


def volume_ratio(volumes: Sequence[float], period: int = 20) -> list[Number]:
    """Volume as a multiple of its own moving average.

    Above ~1.5 marks participation; a breakout on sub-1.0 volume is the
    signature of a move nobody is defending.
    """
    avg = sma(volumes, period)
    out: list[Number] = [None] * len(volumes)
    for i, (v, a) in enumerate(zip(volumes, avg)):
        if a is not None and a > 0:
            out[i] = v / a
    return out


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def percentile_rank(values: Sequence[Number], current: float | None = None) -> float | None:
    """Where ``current`` sits within ``values``, as 0-100.

    Used for volatility percentiles, which are far more actionable than the
    raw level -- "IV at 28" means nothing until you know 28 is this name's
    90th percentile.
    """
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    if current is None:
        current = defined[-1]
    below = sum(1 for v in defined if v < current)
    ties = sum(1 for v in defined if v == current)
    return 100.0 * (below + 0.5 * ties) / len(defined)


def _smooth_optional(values: Sequence[Number], period: int) -> list[Number]:
    """SMA over a list that may be front-padded with ``None``."""
    if period <= 1:
        return list(values)
    defined = [v for v in values if v is not None]
    offset = len(values) - len(defined)
    return [None] * offset + list(sma(defined, period))


def _check_period(period: int) -> None:
    if period < 1:
        raise ValueError(f"period must be at least 1, got {period}")


def _same_length(**arrays: Sequence) -> None:
    lengths = {name: len(a) for name, a in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"length mismatch: {lengths}")
