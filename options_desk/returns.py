"""Return on collateral, annualized, by strategy.

Two things this module insists on, because both are routinely fudged in ways
that flatter the numbers:

**Say which collateral base you mean.** A put sold for 2.00 against a 100
strike can be quoted as 2.0% (premium over full strike) or 2.04% (premium
over net cash actually tied up, 98.00). Both appear in the wild. Both are
computed here, explicitly labelled, so the comparison is never accidental.

**Say which annualization you mean.** A 2% return over 30 days is either
24.3% (simple: scale by 365/30) or 27.4% (compounded: assume you repeat the
trade twelve times and reinvest). Compounded is the larger number and the
one marketing prefers, but it assumes you can redeploy at the same terms all
year -- which is exactly what stops being true when volatility collapses or
the position goes against you. Simple is the honest default; both are
reported.

The annualized figures are what a position *would* return if the outcome
held and the trade repeated. They are not expected returns: neither one is
multiplied by the probability of actually keeping the premium. Pair them
with ``pricing.probabilities`` before comparing strikes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .collateral import CollateralAssumptions, evaluate

CONTRACT_MULTIPLIER = 100.0


@dataclass
class ReturnProfile:
    """What a position earns on the capital it ties up."""

    strategy: str
    premium: float            # credit received, per position
    collateral: float         # capital tied up, per position
    net_collateral: float     # collateral less the credit received
    dte: float
    static_return: float      # premium / collateral, as a fraction
    net_return: float         # premium / net_collateral, as a fraction
    annualized_simple: float
    annualized_compound: float
    max_profit: float
    max_loss: float | None    # None where loss is unbounded
    breakeven: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def static_return_pct(self) -> float:
        return 100.0 * self.static_return

    @property
    def annualized_simple_pct(self) -> float:
        return 100.0 * self.annualized_simple

    @property
    def annualized_compound_pct(self) -> float:
        return 100.0 * self.annualized_compound

    @property
    def return_on_risk(self) -> float | None:
        """Premium over maximum loss -- the defined-risk comparison."""
        if self.max_loss is None or self.max_loss <= 0:
            return None
        return self.premium / self.max_loss

    def __str__(self) -> str:
        lines = [
            f"{self.strategy}  ({self.dte:.0f} DTE)",
            f"  Premium:            {self.premium:>12,.2f}",
            f"  Collateral:         {self.collateral:>12,.2f}",
            f"  Net collateral:     {self.net_collateral:>12,.2f}",
            f"  Static return:      {self.static_return_pct:>11.2f}%",
            f"  Annualized (simple):   {self.annualized_simple_pct:>8.2f}%",
            f"  Annualized (compound): {self.annualized_compound_pct:>8.2f}%",
            f"  Max profit:         {self.max_profit:>12,.2f}",
            f"  Max loss:           "
            + ("  undefined (unbounded)" if self.max_loss is None else f"{self.max_loss:>12,.2f}"),
        ]
        if self.breakeven is not None:
            lines.append(f"  Breakeven:          {self.breakeven:>12,.2f}")
        if self.return_on_risk is not None:
            lines.append(f"  Return on risk:     {100 * self.return_on_risk:>11.2f}%")
        if self.notes:
            lines.append("")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


def annualize_simple(return_fraction: float, dte: float) -> float:
    """Scale a period return to a year linearly. No reinvestment assumed."""
    if dte <= 0:
        raise ValueError(f"dte must be positive to annualize, got {dte}")
    return return_fraction * 365.0 / dte


def annualize_compound(return_fraction: float, dte: float) -> float:
    """Compound a period return over a year.

    Assumes the trade repeats at identical terms all year, which it will not.
    Treat this as an upper bound rather than a forecast.
    """
    if dte <= 0:
        raise ValueError(f"dte must be positive to annualize, got {dte}")
    if return_fraction <= -1.0:
        return -1.0
    return (1.0 + return_fraction) ** (365.0 / dte) - 1.0


def cash_secured_put(
    strike: float,
    premium: float,
    dte: float,
    contracts: int = 1,
    multiplier: float = CONTRACT_MULTIPLIER,
) -> ReturnProfile:
    """Short put fully secured by cash.

    Collateral is the full strike value: assignment obliges you to buy the
    shares at the strike, so that cash must be there. The net figure backs
    out the credit received, which is cash you already hold.
    """
    _validate(strike=strike, premium=premium, dte=dte, contracts=contracts)
    credit = premium * multiplier * contracts
    collateral = strike * multiplier * contracts
    net_collateral = collateral - credit
    static = credit / collateral
    net = credit / net_collateral if net_collateral > 0 else float("inf")

    return ReturnProfile(
        strategy="Cash-secured put",
        premium=credit,
        collateral=collateral,
        net_collateral=net_collateral,
        dte=dte,
        static_return=static,
        net_return=net,
        annualized_simple=annualize_simple(static, dte),
        annualized_compound=annualize_compound(static, dte),
        max_profit=credit,
        max_loss=net_collateral,  # worst case: shares go to zero
        breakeven=strike - premium,
        notes=[
            "Max loss assumes the underlying goes to zero. Unlikely, but this "
            "is the number the collateral is actually sized against.",
            "Collateral here is idle cash, which makes this the clearest case "
            "for sweeping it into a money market -- see with_collateral_yield().",
        ],
    )


def covered_call(
    stock_basis: float,
    strike: float,
    premium: float,
    dte: float,
    contracts: int = 1,
    multiplier: float = CONTRACT_MULTIPLIER,
) -> ReturnProfile:
    """Long stock with a short call against it.

    Two returns matter and they differ: the return if the stock is unchanged
    (you keep the premium) and the return if it is called away (premium plus
    the move to the strike). Quoting only the second flatters a position
    whose upside you have just sold.
    """
    _validate(strike=strike, premium=premium, dte=dte, contracts=contracts)
    if stock_basis <= 0:
        raise ValueError(f"stock_basis must be positive, got {stock_basis}")

    credit = premium * multiplier * contracts
    collateral = stock_basis * multiplier * contracts
    static = credit / collateral
    called_away_gain = (premium + strike - stock_basis) * multiplier * contracts
    return_if_called = called_away_gain / collateral

    notes = [
        f"If called away at {strike:,.2f} the return is "
        f"{100 * return_if_called:.2f}% "
        f"({100 * annualize_simple(return_if_called, dte):.2f}% annualized) -- "
        "the better number, but it caps you there.",
        "Collateral is stock, not cash, so a money market sweep does not apply. "
        "The dividend, if any, is the yield on this collateral.",
    ]
    if strike < stock_basis:
        notes.append(
            f"The strike is below your basis, so being called away locks in a "
            f"loss of {(stock_basis - strike) * multiplier * contracts:,.2f} "
            "before premium. That may still be the right trade, but it is a "
            "decision to exit, not an income trade."
        )

    return ReturnProfile(
        strategy="Covered call",
        premium=credit,
        collateral=collateral,
        net_collateral=collateral - credit,
        dte=dte,
        static_return=static,
        net_return=credit / (collateral - credit) if collateral > credit else float("inf"),
        annualized_simple=annualize_simple(static, dte),
        annualized_compound=annualize_compound(static, dte),
        max_profit=called_away_gain,
        max_loss=collateral - credit,
        breakeven=stock_basis - premium,
        notes=notes,
    )


def credit_spread(
    short_strike: float,
    long_strike: float,
    credit: float,
    dte: float,
    contracts: int = 1,
    multiplier: float = CONTRACT_MULTIPLIER,
) -> ReturnProfile:
    """Vertical credit spread -- defined risk, so collateral is the max loss.

    Return on collateral looks far higher than a cash-secured put because the
    capital tied up is only the spread width less the credit. That is real
    leverage, and it cuts both ways: the same move that costs a cash-secured
    put a few percent takes the whole collateral here.
    """
    _validate(strike=short_strike, premium=credit, dte=dte, contracts=contracts)
    if long_strike <= 0:
        raise ValueError(f"long_strike must be positive, got {long_strike}")
    width = abs(short_strike - long_strike)
    if width == 0:
        raise ValueError("strikes must differ")
    if credit >= width:
        raise ValueError(
            f"credit {credit} cannot meet or exceed the width {width}; that "
            "would be a riskless arbitrage and means the inputs are wrong"
        )

    total_credit = credit * multiplier * contracts
    collateral = (width - credit) * multiplier * contracts
    static = total_credit / collateral
    is_put_spread = long_strike < short_strike
    breakeven = (
        short_strike - credit if is_put_spread else short_strike + credit
    )

    return ReturnProfile(
        strategy=f"{'Put' if is_put_spread else 'Call'} credit spread "
        f"({width:g} wide)",
        premium=total_credit,
        collateral=collateral,
        net_collateral=collateral,
        dte=dte,
        static_return=static,
        net_return=static,
        annualized_simple=annualize_simple(static, dte),
        annualized_compound=annualize_compound(static, dte),
        max_profit=total_credit,
        max_loss=collateral,
        breakeven=breakeven,
        notes=[
            "Collateral is the max loss, so the annualized figure is a return "
            "on risk rather than on cash. Do not compare it directly against a "
            "cash-secured put's number without adjusting for the leverage.",
            "Brokers hold this as a margin requirement, not as swept cash, so "
            "there is usually nothing here to move into a money market.",
        ],
    )


def naked_option(
    spot: float,
    strike: float,
    premium: float,
    dte: float,
    is_put: bool = True,
    contracts: int = 1,
    multiplier: float = CONTRACT_MULTIPLIER,
) -> ReturnProfile:
    """Uncovered short option under a Reg T style margin requirement.

    The requirement modelled is the standard formulation:

        premium + max(20% of spot - out-of-money amount, 10% floor)

    with the floor taken on the strike for puts and on spot for calls. Real
    requirements vary by broker and are frequently higher; portfolio margin
    computes something else entirely. Treat this as an estimate and confirm
    against your own account before sizing anything.
    """
    _validate(strike=strike, premium=premium, dte=dte, contracts=contracts)
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")

    out_of_money = max(0.0, (spot - strike) if is_put else (strike - spot))
    floor = 0.10 * (strike if is_put else spot)
    requirement = premium + max(0.20 * spot - out_of_money, floor)

    total_premium = premium * multiplier * contracts
    collateral = requirement * multiplier * contracts
    static = total_premium / collateral

    return ReturnProfile(
        strategy=f"Naked {'put' if is_put else 'call'} (Reg T estimate)",
        premium=total_premium,
        collateral=collateral,
        net_collateral=collateral - total_premium,
        dte=dte,
        static_return=static,
        net_return=static,
        annualized_simple=annualize_simple(static, dte),
        annualized_compound=annualize_compound(static, dte),
        max_profit=total_premium,
        # A naked call's loss is unbounded; a naked put's floor is the strike.
        max_loss=(strike * multiplier * contracts - total_premium) if is_put else None,
        breakeven=(strike - premium) if is_put else (strike + premium),
        notes=[
            "Margin requirement is an estimate of the Reg T formula. Confirm "
            "against your broker before sizing -- house requirements are often "
            "higher, and they rise as the position moves against you.",
            "The requirement is not static: it expands as the option goes into "
            "the money, so the return on collateral computed at entry overstates "
            "what you earn if the trade goes wrong.",
        ]
        + (
            []
            if is_put
            else [
                "Loss on a naked call is unbounded. No return-on-collateral "
                "figure captures that, and this one should not be compared "
                "against defined-risk alternatives as though it did."
            ]
        ),
    )


def with_collateral_yield(
    profile: ReturnProfile,
    assumptions: CollateralAssumptions | None = None,
    fund_net_yield: float | None = None,
) -> dict[str, float | str]:
    """Add the yield earned on collateral to the option's own return.

    This is where the two halves of the desk meet. A cash-secured put earning
    2% over 30 days is a 24.3% annualized premium yield -- but the collateral
    behind it is also sitting somewhere earning something. Total return is
    both, and ignoring the second is how a money market sweep gets dismissed
    as a rounding error when it is a third of the total.

    Pass ``assumptions`` to price the sweep decision properly (haircut and
    forced-borrow costs included), or ``fund_net_yield`` to just layer a flat
    yield on top.
    """
    premium_yield = profile.annualized_simple

    if assumptions is not None:
        scaled = CollateralAssumptions(
            **{**assumptions.__dict__, "collateral": profile.collateral}
        )
        result = evaluate(scaled)
        collateral_yield = result.net_bps / 10_000.0
        source = "money market sweep, net of haircut and forced-borrow costs"
        verdict = result.verdict
    elif fund_net_yield is not None:
        if fund_net_yield < 0:
            raise ValueError(f"fund_net_yield cannot be negative, got {fund_net_yield}")
        collateral_yield = fund_net_yield
        source = "flat assumed yield on collateral"
        verdict = ""
    else:
        raise ValueError("supply either assumptions or fund_net_yield")

    total = premium_yield + collateral_yield
    share = 0.0 if total == 0 else collateral_yield / total

    return {
        "premium_yield_annualized": premium_yield,
        "collateral_yield_annualized": collateral_yield,
        "total_yield_annualized": total,
        "collateral_share_of_total": share,
        "collateral_source": source,
        "collateral_verdict": verdict,
        "summary": (
            f"Premium yield {100 * premium_yield:.2f}% annualized plus "
            f"{100 * collateral_yield:.2f}% on the collateral gives "
            f"{100 * total:.2f}% total. The collateral contributes "
            f"{100 * share:.0f}% of it -- "
            + (
                "a large enough share that where the cash sits is a real "
                "decision, not housekeeping."
                if share > 0.15
                else "a small share, so the premium is doing the work here."
            )
        ),
    }


def _validate(strike: float, premium: float, dte: float, contracts: int) -> None:
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if premium <= 0:
        raise ValueError(f"premium must be positive, got {premium}")
    if dte <= 0:
        raise ValueError(f"dte must be positive, got {dte}")
    if contracts < 1:
        raise ValueError(f"contracts must be at least 1, got {contracts}")
