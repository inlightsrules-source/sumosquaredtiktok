"""Render an Analysis as a terminal readout."""

from __future__ import annotations

from .interpret import Analysis
from .levels import Level

WIDTH = 78


def render(analysis: Analysis, max_levels: int = 8) -> str:
    a = analysis
    out: list[str] = []
    out.append(_rule("="))
    out.append(f"  {a.symbol} @ {a.price:,.2f}   as of {a.as_of}")
    out.append(_rule("="))

    out.append("")
    out.append(f"REGIME: {a.regime.upper()}")
    out.append(_wrap(a.regime_note))

    out.append("")
    out.append(_section("TREND"))
    out.append(
        f"  Direction: {a.trend.direction}   Strength: {a.trend.strength}   "
        f"Fit (R2): {a.trend.quality:.2f}"
    )
    out.append(_wrap(a.trend.summary))
    out.append("")
    for reading in a.trend.readings:
        out.extend(_reading_lines(reading))

    for title, readings in (
        ("MOMENTUM", a.momentum),
        ("VOLATILITY", a.volatility),
        ("PARTICIPATION", a.participation),
    ):
        if not readings:
            continue
        out.append("")
        out.append(_section(title))
        for reading in readings:
            out.extend(_reading_lines(reading))

    out.append("")
    out.append(_section("SUPPORT & RESISTANCE"))
    if not a.levels:
        out.append("  No levels met the touch threshold over this window.")
    else:
        ranked = sorted(a.levels, key=lambda lv: lv.strength, reverse=True)[:max_levels]
        for level in sorted(ranked, key=lambda lv: lv.price, reverse=True):
            out.append(_level_line(level, a.price))
        out.append("")
        out.append(
            "  Strength is relative within this window (0-100), combining touch "
            "count,"
        )
        out.append(
            "  recency, volume at each touch, and whether the level has flipped "
            "roles."
        )

    if a.conflicts:
        out.append("")
        out.append(_section("CONFLICTING SIGNALS"))
        for conflict in a.conflicts:
            out.append(_wrap(conflict, bullet="!"))
        out.append("")
        out.append(
            _wrap(
                "Indicators disagreeing is information, not noise to average out. "
                "It argues for smaller size until they resolve."
            )
        )

    if a.options_note:
        out.append("")
        out.append(_section("OPTIONS FRAMING"))
        out.append(_wrap(a.options_note))

    out.append("")
    out.append(_rule("="))
    return "\n".join(out)


def _level_line(level: Level, price: float) -> str:
    distance = level.distance_pct(price)
    marker = {"resistance": "^", "support": "v", "pivot": "*"}.get(level.role, " ")
    bar = "#" * max(1, round(level.strength / 10))
    return (
        f"  {marker} {level.price:>10,.2f}  {distance:>+7.1f}%  "
        f"{level.role:<10} {level.touches:>2}t  "
        f"{bar:<10} {level.strength:>5.0f}  last {level.last_touch}"
    )


def _reading_lines(reading) -> list[str]:
    shown = "n/a" if reading.value is None else f"{reading.value:,.2f}"
    head = f"  {reading.name}: {shown}  [{reading.label}]"
    return [head, _wrap(reading.interpretation, indent=6)]


def _section(title: str) -> str:
    return f"{title}\n{_rule('-')}"


def _rule(char: str) -> str:
    return char * WIDTH


def _wrap(text: str, indent: int = 4, bullet: str = "") -> str:
    import textwrap

    prefix = " " * indent
    first = f"{' ' * (indent - 2)}{bullet} " if bullet else prefix
    return textwrap.fill(
        text, width=WIDTH, initial_indent=first, subsequent_indent=prefix
    )
