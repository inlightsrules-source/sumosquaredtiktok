"""Should collateral sit in cash at the broker, or in a money market fund?

The naive comparison -- MMF yield minus broker credit rate -- overstates the
gain, because sweeping collateral out of cash carries two costs that only
show up in bad states of the world:

**Haircut cost.** Cash pledged as collateral takes no haircut. Fund shares
typically take 0-2% (more for prime funds). That haircut is buying power you
no longer have, and its cost is whatever that buying power earns you.

**Forced-borrow cost.** Money market funds generally settle T+1. An intraday
margin call does not wait for settlement. If you cannot deliver, you borrow
at the margin debit rate or liquidate into a bad tape. Margin rates run far
above money market yields, so a single forced borrow can consume months of
yield pickup. This is the term that people leave out, and it is the one that
decides the answer for a leveraged book.

The model prices all three effects and reports a break-even, so the output is
a decision rather than a yield quote.
"""

from __future__ import annotations

from dataclasses import dataclass

# Government/treasury funds only, for collateral. Institutional prime funds
# carry discretionary liquidity fees under the 2023 SEC amendments -- fee
# machinery that can activate under exactly the market stress that produces
# your margin call. Wrong instrument for collateral at any yield.
FUND_TYPES = {
    "government": {"typical_haircut": 0.01, "gate_risk": "none"},
    "treasury": {"typical_haircut": 0.005, "gate_risk": "none"},
    "prime": {"typical_haircut": 0.02, "gate_risk": "liquidity fees possible"},
}


@dataclass
class CollateralAssumptions:
    """Inputs, all rates as decimals (0.042 == 4.2%).

    Defaults are placeholders. Replace every one with your actual broker's
    numbers before drawing conclusions -- broker credit rates in particular
    vary by an order of magnitude and are usually tiered by balance.
    """

    collateral: float                     # collateral parked, in currency units
    broker_credit_rate: float = 0.005     # what idle cash earns at the broker
    fund_gross_yield: float = 0.042       # MMF 7-day yield, before fees
    fund_expense_ratio: float = 0.0015    # fund's annual expense ratio
    haircut: float = 0.01                 # fraction of MMF value not lendable
    buying_power_return: float = 0.0      # annual return earned on buying power
    margin_debit_rate: float = 0.085      # rate charged on a forced borrow
    prob_forced_borrow: float = 0.10      # annual probability of a settlement squeeze
    forced_borrow_fraction: float = 0.25  # share of collateral borrowed against
    forced_borrow_days: int = 3           # days the borrow stays open
    state_tax_rate: float = 0.0           # marginal state rate, for the exemption
    treasury_exempt_fraction: float = 0.0 # share of fund income exempt from state tax

    def __post_init__(self) -> None:
        if self.collateral <= 0:
            raise ValueError(f"collateral must be positive, got {self.collateral}")
        if not 0.0 <= self.haircut < 1.0:
            raise ValueError(f"haircut must be in [0, 1), got {self.haircut}")
        if not 0.0 <= self.prob_forced_borrow <= 1.0:
            raise ValueError(
                f"prob_forced_borrow must be in [0, 1], got {self.prob_forced_borrow}"
            )
        if not 0.0 <= self.forced_borrow_fraction <= 1.0:
            raise ValueError(
                f"forced_borrow_fraction must be in [0, 1], got "
                f"{self.forced_borrow_fraction}"
            )
        if self.forced_borrow_days < 0:
            raise ValueError(
                f"forced_borrow_days cannot be negative, got {self.forced_borrow_days}"
            )
        if not 0.0 <= self.state_tax_rate < 1.0:
            raise ValueError(f"state_tax_rate must be in [0, 1), got {self.state_tax_rate}")
        if not 0.0 <= self.treasury_exempt_fraction <= 1.0:
            raise ValueError(
                f"treasury_exempt_fraction must be in [0, 1], got "
                f"{self.treasury_exempt_fraction}"
            )
        if self.fund_expense_ratio < 0:
            raise ValueError(
                f"fund_expense_ratio cannot be negative, got {self.fund_expense_ratio}"
            )


@dataclass
class CollateralResult:
    gross_pickup: float          # annual currency gain, ignoring costs
    haircut_cost: float          # annual cost of lost buying power
    expected_borrow_cost: float  # probability-weighted forced-borrow cost
    state_tax_saving: float      # value of the treasury state-tax exemption
    net_benefit: float           # what actually lands in the account
    net_bps: float               # net benefit in basis points on collateral
    breakeven_prob: float | None # forced-borrow probability that zeroes the trade
    survives_certain_borrow: bool  # still positive at a 100% chance of borrowing
    verdict: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            f"Gross yield pickup:      {self.gross_pickup:>12,.2f}",
            f"Haircut cost:            {-self.haircut_cost:>12,.2f}",
            f"Expected borrow cost:    {-self.expected_borrow_cost:>12,.2f}",
            f"State tax saving:        {self.state_tax_saving:>12,.2f}",
            f"{'-' * 38}",
            f"Net benefit:             {self.net_benefit:>12,.2f}  ({self.net_bps:+.0f} bps)",
            "",
            f"Verdict: {self.verdict}",
        ]
        if self.breakeven_prob is not None:
            lines.insert(
                -2,
                f"Break-even forced-borrow probability: {self.breakeven_prob:.1%}",
            )
        if self.notes:
            lines.append("")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


def evaluate(assumptions: CollateralAssumptions, day_count: int = 360) -> CollateralResult:
    """Price the sweep decision.

    ``day_count`` is 360 because broker margin interest is conventionally
    quoted on a 360-day basis; pass 365 if yours is not.
    """
    a = assumptions
    net_fund_yield = a.fund_gross_yield - a.fund_expense_ratio
    spread = net_fund_yield - a.broker_credit_rate
    gross_pickup = a.collateral * spread

    # Buying power surrendered to the haircut, valued at what it would earn.
    haircut_cost = a.collateral * a.haircut * a.buying_power_return

    # A forced borrow costs the spread between the debit rate and what the
    # collateral keeps earning while pledged -- not the full debit rate.
    borrowed = a.collateral * a.forced_borrow_fraction
    net_borrow_rate = max(0.0, a.margin_debit_rate - net_fund_yield)
    borrow_cost_if_hit = borrowed * net_borrow_rate * a.forced_borrow_days / day_count
    expected_borrow_cost = a.prob_forced_borrow * borrow_cost_if_hit

    state_tax_saving = (
        a.collateral * net_fund_yield * a.treasury_exempt_fraction * a.state_tax_rate
    )

    net = gross_pickup - haircut_cost - expected_borrow_cost + state_tax_saving
    net_bps = 10_000.0 * net / a.collateral

    # The probability of a forced borrow at which the whole trade nets zero.
    # Only meaningful if it lands inside [0, 1]: a "break-even" of 140% is not
    # a probability, it means no amount of borrowing risk kills the trade, and
    # reporting a clamped 100% there would understate the margin of safety.
    breakeven: float | None = None
    survives_certain_borrow = False
    if borrow_cost_if_hit > 0:
        raw = (gross_pickup - haircut_cost + state_tax_saving) / borrow_cost_if_hit
        if raw > 1.0:
            survives_certain_borrow = True
        elif raw >= 0.0:
            breakeven = raw
    else:
        survives_certain_borrow = net > 0

    return CollateralResult(
        gross_pickup=gross_pickup,
        haircut_cost=haircut_cost,
        expected_borrow_cost=expected_borrow_cost,
        state_tax_saving=state_tax_saving,
        net_benefit=net,
        net_bps=net_bps,
        breakeven_prob=breakeven,
        survives_certain_borrow=survives_certain_borrow,
        verdict=_verdict(net_bps, spread, breakeven, a),
        notes=_notes(a, spread, survives_certain_borrow),
    )


def _verdict(net_bps, spread, breakeven, a: CollateralAssumptions) -> str:
    if spread <= 0:
        return (
            "Do not sweep. The fund's net yield does not beat what the broker "
            "already credits on cash, so there is nothing to capture."
        )
    if net_bps <= 0:
        return (
            "Do not sweep. The yield pickup is real but haircut and forced-borrow "
            "costs consume it entirely at these assumptions."
        )
    if net_bps < 25:
        return (
            f"Marginal ({net_bps:+.0f} bps). Positive, but thin enough that it "
            "turns on assumptions you are estimating rather than observing. "
            "Not worth the operational complexity unless the balance is large."
        )
    if breakeven is not None and breakeven < 0.25:
        return (
            f"Sweep, with a cash buffer ({net_bps:+.0f} bps). The pickup is "
            f"worthwhile but fragile: it breaks even at a {breakeven:.0%} annual "
            "chance of a forced borrow. Hold back enough cash to meet a bad "
            "one-day call without touching the fund."
        )
    return (
        f"Sweep ({net_bps:+.0f} bps). The pickup comfortably survives the "
        "haircut and settlement risk at these assumptions."
    )


def _notes(a: CollateralAssumptions, spread: float, survives_certain_borrow: bool) -> list[str]:
    notes: list[str] = []
    if spread > 0.02:
        notes.append(
            f"The {spread:.2%} spread over broker credit is wide, which usually "
            "means the broker is keeping most of the short rate. Worth "
            "negotiating the credit tier as well as sweeping -- the two are "
            "not mutually exclusive."
        )
    if a.buying_power_return <= 0 and a.haircut > 0:
        notes.append(
            "Buying power return is set to zero, so the haircut is being priced "
            "as free. That is only true if you are not buying-power constrained. "
            "If you ever hit a margin limit, set this to your marginal return on "
            "capital and re-run -- the answer can flip."
        )
    if a.prob_forced_borrow < 0.05:
        notes.append(
            "A sub-5% annual chance of a settlement squeeze is optimistic for a "
            "book that is short options. Assignment and gap risk both produce "
            "same-day obligations; stress this input before trusting the verdict."
        )
    if survives_certain_borrow:
        notes.append(
            "The pickup stays positive even at a certainty of forced borrowing "
            "at this size and duration, so there is no break-even probability -- "
            "the borrow assumption is not what decides this case. Stress the "
            "borrowed fraction and duration instead."
        )
    notes.append(
        "Use a government or treasury fund. Institutional prime funds can impose "
        "liquidity fees under stress, which is precisely when collateral must be "
        "available."
    )
    notes.append(
        "Confirm your clearer accepts fund shares as collateral directly. Under "
        "portfolio margin or SPAN many do, and then you keep both the yield and "
        "the buying power -- which makes the haircut term above moot."
    )
    return notes


def compare_scenarios(
    base: CollateralAssumptions, probabilities: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
) -> list[tuple[float, CollateralResult]]:
    """Re-run across forced-borrow probabilities.

    The single most important sensitivity, because it is the input you know
    least well and the one most able to change the sign of the answer.
    """
    out = []
    for prob in probabilities:
        variant = CollateralAssumptions(**{**base.__dict__, "prob_forced_borrow": prob})
        out.append((prob, evaluate(variant)))
    return out
