"""One readout per candidate trade: IV context, Greeks, odds, and return.

Ties the pieces together so a strike can be judged on the four things that
actually decide it -- is the premium rich, what risk am I holding, what are
the odds, and what does it pay on the capital it ties up.
"""

from __future__ import annotations

from dataclasses import dataclass

from .interpret import Analysis
from .pricing import Greeks, Option, Probabilities, greeks, probabilities
from .returns import ReturnProfile
from .vol import VolContext


@dataclass
class TradeView:
    option: Option
    vol: VolContext | None
    greeks: Greeks
    odds: Probabilities
    profile: ReturnProfile | None
    level_note: str = ""
    verdict: str = ""


def evaluate_trade(
    option: Option,
    premium: float,
    vol_context: VolContext | None = None,
    profile: ReturnProfile | None = None,
    analysis: Analysis | None = None,
) -> TradeView:
    """Assemble the full picture for one strike."""
    if premium <= 0:
        raise ValueError(f"premium must be positive, got {premium}")
    view = TradeView(
        option=option,
        vol=vol_context,
        greeks=greeks(option).per_contract(),
        odds=probabilities(option, premium),
        profile=profile,
        level_note=_level_note(option, analysis),
    )
    view.verdict = _verdict(view)
    return view


def _level_note(option: Option, analysis: Analysis | None) -> str:
    """Check the strike against the technical levels and the expected move."""
    if analysis is None:
        return ""
    is_upside = option.strike > option.spot
    level = analysis.key_resistance if is_upside else analysis.key_support
    if level is None:
        return "No clustered level sits between spot and this strike."

    parts: list[str] = []
    beyond = (option.strike > level.price) if is_upside else (option.strike < level.price)
    side = "resistance" if is_upside else "support"
    if beyond:
        parts.append(
            f"The strike sits beyond {side} at {level.price:,.2f} "
            f"(strength {level.strength:.0f}/100). Price has to break a defended "
            f"level before the strike is threatened, which is the setup you want "
            f"when short."
        )
    else:
        parts.append(
            f"The strike sits inside {side} at {level.price:,.2f} "
            f"(strength {level.strength:.0f}/100) -- price can reach it without "
            f"breaking anything. Consider moving out past the level."
        )
    return " ".join(parts)


def _verdict(view: TradeView) -> str:
    """Combine the four readings into a stated conclusion."""
    parts: list[str] = []
    odds = view.odds

    parts.append(
        f"{odds.prob_otm_pct:.0f}% chance of expiring worthless, "
        f"{odds.prob_touch_pct:.0f}% chance of touching the strike first."
    )
    if odds.prob_touch_pct > 2.2 * odds.prob_itm_pct:
        parts.append(
            "Touch probability runs well above double the ITM chance, so if you "
            "manage this position on a breach rather than holding to expiry, the "
            "relevant odds are much worse than the ITM number suggests."
        )

    if view.vol is not None:
        if view.vol.edge == "sell premium":
            parts.append("Volatility context supports selling premium here.")
        elif view.vol.edge == "buy premium":
            parts.append(
                "Volatility context argues against selling: IV is not being paid "
                "enough relative to what the stock is delivering."
            )
        else:
            parts.append("Volatility offers no clear edge either way.")

    theta, vega = view.greeks.theta_per_day, view.greeks.vega_per_point
    parts.append(
        f"Carries {theta:+,.2f} of theta per day against {vega:+,.2f} per "
        f"volatility point"
        + (
            ". Decay works for you, but a vol expansion works against you faster "
            "than a day of theta pays."
            if abs(vega) > abs(theta) * 3
            else "."
        )
    )

    if view.profile is not None:
        parts.append(
            f"Pays {view.profile.annualized_simple_pct:.1f}% annualized on "
            f"{view.profile.collateral:,.0f} of collateral."
        )
    if view.level_note:
        parts.append(view.level_note)
    return " ".join(parts)


def render_trade(view: TradeView) -> str:
    """Terminal readout for one candidate trade."""
    o, g, p = view.option, view.greeks, view.odds
    width = 78
    lines = [
        "=" * width,
        f"  {o.kind.upper()} {o.strike:,.2f}  |  spot {o.spot:,.2f}  |  "
        f"{o.dte:.0f} DTE  |  IV {100 * o.volatility:.1f}%",
        "=" * width,
        "",
        "GREEKS (per contract)",
        "-" * width,
        f"  Delta {g.delta:>9.3f}   Gamma {g.gamma:>9.4f}   "
        f"Vega {g.vega_per_point:>8.3f} /pt",
        f"  Theta {g.theta_per_day:>9.3f} /day   Rho {g.rho_per_point:>8.3f} /pt",
        f"  Vanna {g.vanna:>9.4f}   Charm {g.charm:>9.4f}",
        "",
        "PROBABILITY",
        "-" * width,
        f"  Expire ITM:      {p.prob_itm_pct:>6.1f}%",
        f"  Expire OTM:      {p.prob_otm_pct:>6.1f}%",
        f"  Touch strike:    {p.prob_touch_pct:>6.1f}%",
    ]
    if p.prob_profit_pct is not None:
        lines.append(f"  Profit at expiry:{p.prob_profit_pct:>6.1f}%")
    if p.breakeven is not None:
        lines.append(f"  Breakeven:       {p.breakeven:>9,.2f}")
    lines.append(f"  Expected move:   {p.expected_move:>9,.2f}  (1 sigma by expiry)")

    if view.vol is not None:
        lines += ["", "VOLATILITY CONTEXT", "-" * width]
        lines.append(_wrap(view.vol.interpretation))

    if view.profile is not None:
        lines += ["", "RETURN ON COLLATERAL", "-" * width, str(view.profile)]

    lines += ["", "VERDICT", "-" * width, _wrap(view.verdict), "", "=" * width]
    return "\n".join(lines)


def _wrap(text: str, indent: int = 4) -> str:
    import textwrap

    pad = " " * indent
    return textwrap.fill(text, width=78, initial_indent=pad, subsequent_indent=pad)
