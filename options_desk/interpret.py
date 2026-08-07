"""Turn indicator values into stated conclusions.

Two design rules hold throughout:

**Say what it means, not what it is.** "RSI 71" is a number the caller
already had. "Momentum strong; in an established uptrend this is
confirmation rather than an exhaustion signal" is an interpretation.

**Report confidence honestly.** Where evidence conflicts, the readout says
so instead of averaging the disagreement into a false verdict. A market
whose indicators disagree is genuinely a market to size down in, and that
is more useful to know than a tidy score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import indicators as ind
from .levels import (
    Level,
    classify_levels,
    cluster_levels,
    find_pivots,
    most_significant,
    nearest,
)
from .ohlcv import Series


@dataclass
class Reading:
    """One interpreted indicator."""

    name: str
    value: float | None
    label: str  # terse classification, e.g. "overbought"
    interpretation: str  # the sentence a human reads

    def __str__(self) -> str:
        shown = "n/a" if self.value is None else f"{self.value:,.2f}"
        return f"{self.name}: {shown} [{self.label}] -- {self.interpretation}"


@dataclass
class TrendView:
    direction: str  # "up", "down", "sideways"
    strength: str  # "strong", "moderate", "weak", "absent"
    quality: float  # R^2 of the regression, 0-1
    readings: list[Reading] = field(default_factory=list)
    summary: str = ""


@dataclass
class Analysis:
    """Full technical readout for one symbol."""

    symbol: str
    price: float
    as_of: str
    trend: TrendView
    momentum: list[Reading] = field(default_factory=list)
    volatility: list[Reading] = field(default_factory=list)
    participation: list[Reading] = field(default_factory=list)
    levels: list[Level] = field(default_factory=list)
    support: Level | None = None       # closest below
    resistance: Level | None = None    # closest above
    key_support: Level | None = None      # best strike reference below
    key_resistance: Level | None = None   # best strike reference above
    regime: str = ""
    regime_note: str = ""
    conflicts: list[str] = field(default_factory=list)
    options_note: str = ""


MIN_BARS = 60


def analyze(series: Series, pivot_window: int = 5) -> Analysis:
    """Run the full analysis over a price series.

    Wants at least ``MIN_BARS`` bars; a 200-day view needs 200 and degrades
    gracefully to the shorter averages when it has less.
    """
    if len(series) < MIN_BARS:
        raise ValueError(
            f"{series.symbol}: need at least {MIN_BARS} bars for a reliable "
            f"read, got {len(series)}"
        )

    closes, highs, lows = series.closes, series.highs, series.lows
    price = series.last.close

    trend = _read_trend(series, closes, highs, lows, price)
    momentum = _read_momentum(closes, highs, lows, trend.direction)
    volatility = _read_volatility(series, closes, highs, lows)
    participation = _read_participation(series, closes)

    pivots = find_pivots(series, window=pivot_window)
    levels = classify_levels(cluster_levels(series, pivots), price)
    support = nearest(levels, price, "support")
    resistance = nearest(levels, price, "resistance")
    key_support = most_significant(levels, price, "support")
    key_resistance = most_significant(levels, price, "resistance")

    regime, regime_note = _classify_regime(trend, volatility)
    conflicts = _find_conflicts(trend, momentum, participation)

    return Analysis(
        symbol=series.symbol,
        price=price,
        as_of=str(series.last.day),
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        participation=participation,
        levels=levels,
        support=support,
        resistance=resistance,
        key_support=key_support,
        key_resistance=key_resistance,
        regime=regime,
        regime_note=regime_note,
        conflicts=conflicts,
        options_note=_options_note(
            regime, key_support, key_resistance, support, resistance, price
        ),
    )


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------


def _read_trend(
    series: Series, closes, highs, lows, price: float
) -> TrendView:
    readings: list[Reading] = []

    ma_periods = [p for p in (20, 50, 200) if len(closes) >= p]
    ma_values: dict[int, float] = {}
    for period in ma_periods:
        value = ind.sma(closes, period)[-1]
        if value is None:
            continue
        ma_values[period] = value
        side = "above" if price > value else "below"
        gap = 100.0 * (price - value) / value
        readings.append(
            Reading(
                f"SMA({period})",
                value,
                side,
                f"Price is {abs(gap):.1f}% {side} its {period}-day average.",
            )
        )

    stack = _describe_stack(ma_values, price)
    if stack:
        readings.append(stack)

    lookback = min(60, len(closes))
    slope, _, r2 = ind.linear_regression(closes[-lookback:])
    slope_pct = 100.0 * slope / price if price else 0.0
    readings.append(
        Reading(
            f"Regression slope ({lookback}d)",
            slope_pct,
            _slope_label(slope_pct),
            f"Fitted drift of {slope_pct:+.2f}% per day with R2 {r2:.2f}. "
            + (
                "The fit is tight, so the slope is a real trend."
                if r2 >= 0.6
                else "The fit is loose -- price is scattered around this line "
                "rather than following it, so treat the slope as weak evidence."
            ),
        )
    )

    adx_data = ind.adx(highs, lows, closes)
    adx_now = adx_data["adx"][-1]
    plus_di = adx_data["plus_di"][-1]
    minus_di = adx_data["minus_di"][-1]
    strength = _adx_strength(adx_now)
    if adx_now is not None:
        di_side = (
            "buyers" if (plus_di or 0) > (minus_di or 0) else "sellers"
        )
        readings.append(
            Reading(
                "ADX(14)",
                adx_now,
                strength,
                f"Trend strength is {strength}; directional pressure favours "
                f"{di_side} (+DI {plus_di:.1f} vs -DI {minus_di:.1f}). "
                + (
                    "Trend-following setups are the ones that work here."
                    if adx_now >= 25
                    else "Below 25 the tape is ranging -- fade the extremes "
                    "rather than chasing breakouts."
                ),
            )
        )

    direction = _trend_direction(price, ma_values, slope_pct, plus_di, minus_di)
    return TrendView(
        direction=direction,
        strength=strength,
        quality=r2,
        readings=readings,
        summary=_trend_summary(direction, strength, r2, ma_values, price),
    )


def _describe_stack(ma_values: dict[int, float], price: float) -> Reading | None:
    """Moving-average stacking -- the cleanest single trend read there is."""
    if len(ma_values) < 2:
        return None
    periods = sorted(ma_values)
    values = [ma_values[p] for p in periods]
    names = "/".join(str(p) for p in periods)
    if all(a > b for a, b in zip(values, values[1:])) and price > values[0]:
        return Reading(
            "MA stack",
            None,
            "bullish",
            f"Averages are stacked fast-over-slow ({names}) with price on top. "
            "This is textbook uptrend structure; pullbacks into the faster "
            "average are the higher-probability entries.",
        )
    if all(a < b for a, b in zip(values, values[1:])) and price < values[0]:
        return Reading(
            "MA stack",
            None,
            "bearish",
            f"Averages are stacked slow-over-fast ({names}) with price beneath. "
            "Downtrend structure -- rallies into the faster average are where "
            "supply tends to reappear.",
        )
    return Reading(
        "MA stack",
        None,
        "mixed",
        "Moving averages are tangled rather than stacked, which means no "
        "timeframe agrees on direction. Trend systems whipsaw in exactly "
        "this configuration.",
    )


def _trend_direction(price, ma_values, slope_pct, plus_di, minus_di) -> str:
    """Vote across three independent trend measures."""
    score = 0
    if ma_values:
        longest = ma_values[max(ma_values)]
        score += 1 if price > longest else -1
    if slope_pct > 0.05:
        score += 1
    elif slope_pct < -0.05:
        score -= 1
    if plus_di is not None and minus_di is not None:
        score += 1 if plus_di > minus_di else -1
    if score >= 2:
        return "up"
    if score <= -2:
        return "down"
    return "sideways"


def _slope_label(slope_pct: float) -> str:
    if slope_pct > 0.15:
        return "rising fast"
    if slope_pct > 0.03:
        return "rising"
    if slope_pct < -0.15:
        return "falling fast"
    if slope_pct < -0.03:
        return "falling"
    return "flat"


def _adx_strength(adx_now: float | None) -> str:
    if adx_now is None:
        return "absent"
    if adx_now >= 40:
        return "strong"
    if adx_now >= 25:
        return "moderate"
    if adx_now >= 20:
        return "weak"
    return "absent"


def _trend_summary(direction, strength, r2, ma_values, price) -> str:
    tail = ""
    if 200 in ma_values:
        side = "above" if price > ma_values[200] else "below"
        tail = f" Price sits {side} the 200-day, the usual long-term dividing line."

    if direction == "sideways":
        return (
            "No directional trend. Price is oscillating rather than "
            "progressing, so the tradeable edge is in the range boundaries, "
            "not in direction." + tail
        )

    word = "uptrend" if direction == "up" else "downtrend"
    quality = (
        "well-defined" if r2 >= 0.6 else "choppy but intact" if r2 >= 0.3 else "ragged"
    )

    # Direction and strength can genuinely disagree: price drifts one way while
    # ADX says no trend persists. Saying so beats gluing the two labels into
    # something like "absent well-defined uptrend".
    if strength in ("absent", "weak"):
        return (
            f"Price is drifting {'higher' if direction == 'up' else 'lower'} "
            f"({quality} fit, R2 {r2:.2f}), but ADX reads trend strength as "
            f"{strength} -- the direction is real while the persistence is not. "
            f"Expect the drift to keep stalling and retracing rather than "
            f"extending cleanly.{tail}"
        )
    return f"{strength.capitalize()} {quality} {word}.{tail}"


# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------


def _read_momentum(closes, highs, lows, direction: str) -> list[Reading]:
    out: list[Reading] = []

    rsi_now = ind.rsi(closes)[-1]
    if rsi_now is not None:
        out.append(Reading("RSI(14)", rsi_now, *_rsi_read(rsi_now, direction)))

    macd_data = ind.macd(closes)
    line, signal, hist = (
        macd_data["macd"][-1],
        macd_data["signal"][-1],
        macd_data["histogram"][-1],
    )
    if line is not None and signal is not None and hist is not None:
        prior = macd_data["histogram"][-2] if len(macd_data["histogram"]) > 1 else None
        widening = prior is not None and abs(hist) > abs(prior)
        posture = "bullish" if line > signal else "bearish"
        drift = (
            "and the histogram is expanding, so the move is still accelerating"
            if widening
            else "but the histogram is contracting, an early warning that the "
            "move is losing steam"
        )
        out.append(
            Reading(
                "MACD(12,26,9)",
                hist,
                posture,
                f"MACD is {posture} ({line:+.2f} vs signal {signal:+.2f}) {drift}.",
            )
        )

    stoch = ind.stochastic(highs, lows, closes)
    k, d = stoch["k"][-1], stoch["d"][-1]
    if k is not None and d is not None:
        if k > 80:
            label, note = "overbought", "closing at the top of its recent range"
        elif k < 20:
            label, note = "oversold", "closing at the bottom of its recent range"
        else:
            label, note = "neutral", "closing mid-range"
        out.append(
            Reading(
                "Stochastic %K",
                k,
                label,
                f"Price is {note} (%K {k:.0f}, %D {d:.0f}). "
                + (
                    "%K above %D adds a near-term upward bias."
                    if k > d
                    else "%K below %D adds a near-term downward bias."
                ),
            )
        )

    roc = ind.rate_of_change(closes, 20)[-1]
    if roc is not None:
        out.append(
            Reading(
                "ROC(20)",
                roc,
                "positive" if roc > 0 else "negative",
                f"Price is {roc:+.1f}% against 20 sessions ago.",
            )
        )
    return out


def _rsi_read(value: float, direction: str) -> tuple[str, str]:
    """RSI means different things depending on the trend it sits inside."""
    if value >= 70:
        if direction == "up":
            return (
                "overbought",
                f"RSI {value:.0f} is above 70, but inside an uptrend a high RSI "
                "is confirmation of strength, not a sell signal -- it can hold "
                "here for weeks. Use it to size, not to fade.",
            )
        return (
            "overbought",
            f"RSI {value:.0f} is stretched without a trend to justify it, "
            "which is the configuration that actually mean-reverts.",
        )
    if value <= 30:
        if direction == "down":
            return (
                "oversold",
                f"RSI {value:.0f} is below 30, but in a downtrend that reflects "
                "persistent selling rather than a bargain. Catching this without "
                "a reversal signal is the classic way to lose money slowly.",
            )
        return (
            "oversold",
            f"RSI {value:.0f} is washed out with no downtrend behind it -- the "
            "more reliable bounce setup.",
        )
    if value >= 55:
        return "firm", f"RSI {value:.0f} sits in the upper-neutral band; buyers have the edge."
    if value <= 45:
        return "soft", f"RSI {value:.0f} sits in the lower-neutral band; sellers have the edge."
    return "neutral", f"RSI {value:.0f} is balanced -- no momentum edge either way."


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------


def _read_volatility(series: Series, closes, highs, lows) -> list[Reading]:
    out: list[Reading] = []
    price = series.last.close

    atr_series = ind.atr(highs, lows, closes)
    atr_now = atr_series[-1]
    if atr_now is not None and price:
        atr_pct = 100.0 * atr_now / price
        out.append(
            Reading(
                "ATR(14)",
                atr_now,
                _atr_label(atr_pct),
                f"Average daily range is {atr_now:,.2f} ({atr_pct:.1f}% of price). "
                f"A normal session covers about this much, so stops inside "
                f"{atr_now:,.2f} will be hit by noise alone. For a one-week "
                f"option, expect roughly {atr_now * 2.2:,.2f} of movement.",
            )
        )

    rvol = ind.realized_volatility(closes, 20)
    rvol_now = rvol[-1]
    if rvol_now is not None:
        rank = ind.percentile_rank(rvol[-252:] if len(rvol) >= 252 else rvol)
        out.append(
            Reading(
                "Realized vol (20d)",
                rvol_now,
                _vol_rank_label(rank),
                f"20-day realized volatility is {rvol_now:.1f}% annualized, "
                f"the {rank:.0f}th percentile of its own recent history. "
                + _vol_rank_advice(rank),
            )
        )

    rvol_long = ind.realized_volatility(closes, 60)[-1]
    if rvol_now is not None and rvol_long is not None and rvol_long > 0:
        ratio = rvol_now / rvol_long
        if ratio > 1.25:
            note = (
                "Short-term vol is running hot against the 60-day baseline -- "
                "volatility is expanding, which favours owning optionality."
            )
        elif ratio < 0.8:
            note = (
                "Short-term vol has fallen below its 60-day baseline. Quiet "
                "tape, but compressed vol is a coiled spring, not a permanent state."
            )
        else:
            note = "Short and medium-term volatility agree; no regime change underway."
        out.append(Reading("Vol term structure (20d/60d)", ratio, _ratio_label(ratio), note))

    bands = ind.bollinger(closes)
    pct_b, width = bands["percent_b"][-1], bands["bandwidth"][-1]
    if pct_b is not None and width is not None:
        width_hist = [w for w in bands["bandwidth"][-252:] if w is not None]
        width_rank = ind.percentile_rank(width_hist) or 50.0
        if width_rank < 20:
            squeeze = (
                "Bandwidth is in the bottom fifth of its range -- a squeeze. "
                "These resolve into large directional moves, and long premium "
                "is at its cheapest precisely here."
            )
        elif width_rank > 80:
            squeeze = (
                "Bands are unusually wide, meaning the expansion has already "
                "happened. Premium is expensive and mean-reversion in vol favours sellers."
            )
        else:
            squeeze = "Band width is unremarkable."
        position = (
            "riding the upper band" if pct_b > 0.95
            else "riding the lower band" if pct_b < 0.05
            else f"at {pct_b:.0%} of the band range"
        )
        out.append(
            Reading("Bollinger %B", pct_b, _pct_b_label(pct_b), f"Price is {position}. {squeeze}")
        )

    park = ind.parkinson_volatility(highs, lows, 20)[-1]
    if park is not None and rvol_now is not None and park > 0:
        gap = rvol_now / park
        if gap > 1.3:
            note = (
                "Close-to-close vol far exceeds range-based vol, which means the "
                "movement is happening in gaps between sessions rather than "
                "intraday. Overnight risk dominates -- short-dated premium is "
                "underpricing it."
            )
        elif gap < 0.75:
            note = (
                "Intraday ranges are wide but closes land near each other. "
                "Whipsaw without progress: bad for directional trades, good for "
                "premium sellers who can sit through the noise."
            )
        else:
            note = "Gap risk and intraday movement are proportionate."
        out.append(
            Reading("Close-to-close / Parkinson", gap, _ratio_label(gap), note)
        )
    return out


def _atr_label(atr_pct: float) -> str:
    if atr_pct > 4:
        return "very high"
    if atr_pct > 2.5:
        return "high"
    if atr_pct > 1.2:
        return "normal"
    return "low"


def _vol_rank_label(rank: float | None) -> str:
    if rank is None:
        return "unknown"
    if rank > 80:
        return "elevated"
    if rank < 20:
        return "compressed"
    return "average"


def _vol_rank_advice(rank: float | None) -> str:
    if rank is None:
        return ""
    if rank > 80:
        return (
            "Realized vol this high usually drags implied vol up with it, so "
            "premium should be rich -- check that IV actually exceeds this "
            "number before selling, because selling vol at cost is uncompensated risk."
        )
    if rank < 20:
        return (
            "Vol this compressed means options are cheap in absolute terms, but "
            "it also means short premium is being paid very little to carry "
            "gap risk. Long-gamma structures get the better of this setup."
        )
    return "Volatility is mid-range; no strong bias between buying and selling premium."


def _ratio_label(ratio: float) -> str:
    if ratio > 1.25:
        return "expanding"
    if ratio < 0.8:
        return "contracting"
    return "stable"


def _pct_b_label(pct_b: float) -> str:
    if pct_b > 1.0:
        return "above band"
    if pct_b < 0.0:
        return "below band"
    if pct_b > 0.8:
        return "upper band"
    if pct_b < 0.2:
        return "lower band"
    return "mid band"


# ---------------------------------------------------------------------------
# participation
# ---------------------------------------------------------------------------


def _read_participation(series: Series, closes) -> list[Reading]:
    if not series.has_volume:
        return [
            Reading(
                "Volume",
                None,
                "unavailable",
                "No volume data in this series, so participation and "
                "accumulation cannot be assessed.",
            )
        ]
    out: list[Reading] = []
    volumes = series.volumes

    ratio = ind.volume_ratio(volumes, 20)[-1]
    if ratio is not None:
        if ratio > 1.5:
            label, note = "heavy", "Well above average -- real participation behind the move."
        elif ratio < 0.7:
            label, note = (
                "light",
                "Below average. Moves on thin volume are poorly defended and "
                "tend to retrace.",
            )
        else:
            label, note = "normal", "Participation is unremarkable."
        out.append(Reading("Volume vs 20d avg", ratio, label, f"{ratio:.2f}x average. {note}"))

    obv = ind.on_balance_volume(closes, volumes)
    window = min(40, len(obv))
    obv_tail = [v for v in obv[-window:] if v is not None]
    price_tail = closes[-window:]
    if len(obv_tail) >= 10:
        obv_slope, _, _ = ind.linear_regression(obv_tail)
        price_slope, _, _ = ind.linear_regression(price_tail)
        if obv_slope > 0 and price_slope < 0:
            label, note = (
                "bullish divergence",
                "Volume is flowing in while price drifts down -- accumulation "
                "under cover of weakness. One of the more reliable divergences.",
            )
        elif obv_slope < 0 and price_slope > 0:
            label, note = (
                "bearish divergence",
                "Price is rising on outflows. The rally is not being funded, "
                "which makes it fragile.",
            )
        elif obv_slope > 0:
            label, note = "confirming", "Volume flow confirms the advance."
        else:
            label, note = "confirming", "Volume flow confirms the decline."
        out.append(Reading("OBV trend (40d)", obv_slope, label, note))
    return out


# ---------------------------------------------------------------------------
# regime, conflicts, options framing
# ---------------------------------------------------------------------------


def _classify_regime(trend: TrendView, volatility: list[Reading]) -> tuple[str, str]:
    """Cross trend against volatility -- the two axes that decide structure."""
    vol_label = next(
        (r.label for r in volatility if r.name.startswith("Realized vol")), "average"
    )
    trending = trend.strength in ("strong", "moderate") and trend.direction != "sideways"

    if trending and vol_label == "elevated":
        return (
            "trending / high volatility",
            "Direction is real but the ride is violent. Position size off ATR, "
            "not off conviction -- this regime stops out correctly-directional "
            "trades more than any other.",
        )
    if trending and vol_label == "compressed":
        return (
            "trending / low volatility",
            "The most favourable regime there is: persistent direction with "
            "cheap optionality. Long-dated directional structures are "
            "underpriced relative to what the trend is delivering.",
        )
    if trending:
        return (
            "trending / normal volatility",
            "An orderly trend. Standard trend-following applies, with pullbacks "
            "to the faster moving average as the entry.",
        )
    # Directional drift without trend persistence -- distinct from a true range,
    # and calling it "range-bound" while the trend section reads "up" is the
    # kind of contradiction that makes a readout untrustworthy.
    if trend.direction != "sideways":
        return (
            f"drifting {trend.direction} / no trend persistence",
            f"Price is grinding {'higher' if trend.direction == 'up' else 'lower'} "
            "but ADX denies a durable trend, so this is a drift punctuated by "
            "stalls rather than a move to lean on. Directional exposure works "
            "here only with enough time to survive the retracements -- short-dated "
            "directional bets get chopped up.",
        )
    if vol_label == "elevated":
        return (
            "choppy / high volatility",
            "The worst regime to trade directionally: large moves that go "
            "nowhere. Rich premium and no persistent direction favours defined-risk "
            "premium selling, but only with room to be wrong.",
        )
    if vol_label == "compressed":
        return (
            "quiet range",
            "Low volatility with no trend -- a coiling market. Cheap premium "
            "ahead of an expansion that has to come eventually. Long straddles "
            "and calendars are the structures that pay here.",
        )
    return (
        "range-bound",
        "No trend and unremarkable volatility. Trade the range boundaries and "
        "keep size modest until something resolves.",
    )


def _find_conflicts(trend, momentum, participation) -> list[str]:
    """Surface disagreement between indicators rather than averaging it away."""
    out: list[str] = []
    for reading in participation:
        if "divergence" in reading.label:
            out.append(f"{reading.name} shows a {reading.label} against price.")
    rsi_reading = next((r for r in momentum if r.name.startswith("RSI")), None)
    if rsi_reading and rsi_reading.value is not None:
        if trend.direction == "up" and rsi_reading.value < 45:
            out.append("Trend reads up but RSI is soft -- the advance is losing momentum.")
        if trend.direction == "down" and rsi_reading.value > 55:
            out.append("Trend reads down but RSI is firm -- selling pressure is fading.")
    macd_reading = next((r for r in momentum if r.name.startswith("MACD")), None)
    if macd_reading:
        if trend.direction == "up" and macd_reading.label == "bearish":
            out.append("MACD has rolled bearish while the broader trend is still up.")
        if trend.direction == "down" and macd_reading.label == "bullish":
            out.append("MACD has turned bullish while the broader trend is still down.")
    if trend.direction != "sideways" and trend.quality < 0.3:
        out.append(
            "The trend has a direction but a poor regression fit -- weaker than "
            "the direction label alone suggests."
        )
    return out


def _options_note(regime, key_support, key_resistance, support, resistance, price) -> str:
    """Translate the technical picture into strike-selection guidance."""
    parts: list[str] = []
    if key_support is not None:
        parts.append(
            f"Put strikes reference {key_support.price:,.2f} "
            f"({abs(key_support.distance_pct(price)):.1f}% below, strength "
            f"{key_support.strength:.0f}/100 on {key_support.touches} touches) -- "
            f"a strike beneath a well-defended level is protected by more than "
            f"just your thesis."
        )
        if support is not None and support is not key_support:
            parts.append(
                f"Note that {support.price:,.2f} is closer but far weaker "
                f"({support.strength:.0f}/100 on {support.touches} touches), so "
                f"it is likely to give way rather than hold."
            )
    if key_resistance is not None:
        parts.append(
            f"Call strikes reference {key_resistance.price:,.2f} "
            f"({key_resistance.distance_pct(price):.1f}% above, strength "
            f"{key_resistance.strength:.0f}/100 on {key_resistance.touches} "
            f"touches) -- also where a covered call is most likely to be assigned."
        )
        if resistance is not None and resistance is not key_resistance:
            parts.append(
                f"{resistance.price:,.2f} sits closer but carries less weight "
                f"({resistance.strength:.0f}/100)."
            )
    if "low volatility" in regime or "quiet" in regime:
        parts.append(
            "With premium compressed, structures that own gamma are favoured "
            "over structures that sell it."
        )
    elif "high volatility" in regime:
        parts.append(
            "With premium elevated, selling is paid -- but keep it defined-risk, "
            "because high realized vol is exactly when undefined short options "
            "become uninsurable."
        )
    return " ".join(parts)
