# options_desk

Technical analysis and collateral tooling for the options desk. Pure standard
library — no numpy, pandas, or vendor SDK required, so it drops into any
Python 3.10+ environment.

```bash
python3 -m unittest discover -s tests
```

## Technical analysis

Everything works on a `Series` of OHLCV bars and nothing else, so the data
source is a detail you swap at the edge.

```python
from options_desk import adapters, analyze, render

series = adapters.from_csv("data/AAPL.csv")   # or from_dicts / from_columns / from_yfinance
print(render(analyze(series)))
```

The readout covers:

| Section | Contents |
|---|---|
| Regime | Trend crossed against volatility — the two axes that decide structure |
| Trend | MA stack, regression slope with R², ADX/DI |
| Momentum | RSI, MACD, stochastic, rate of change |
| Volatility | ATR, realized vol with percentile, 20d/60d term structure, Bollinger squeeze, close-to-close vs Parkinson |
| Participation | Volume ratio, OBV with divergence detection |
| Support & resistance | Clustered, strength-scored zones |
| Conflicting signals | Where indicators disagree, stated rather than averaged away |
| Options framing | Which levels to hang strikes on |

Every value ships with an interpretation. `RSI 71` is a number you already
had; what the module reports is that a high RSI inside an uptrend is
confirmation rather than an exhaustion signal.

### Support and resistance

Three steps: find swing pivots, cluster them within an ATR-scaled tolerance,
then score each zone. Scoring weighs **touches**, **recency** (exponential
decay, tunable half-life), **volume at each touch**, and a bonus for zones
that have **flipped** between support and resistance.

The output is a ranked list of zones with reasons attached, not one "the
resistance is X" number — price does not respect single lines, and pretending
otherwise makes for confident bad trades.

Two accessors, deliberately different:

- `nearest()` — closest zone. What price meets first.
- `most_significant()` — best strike reference, discounting strength by
  distance. A barely-touched level 2% away is a worse place to sell a put than
  a heavily-defended shelf 8% away, because the strong one is where price
  actually stops.

Also available: `pivot_points()` (classic and Fibonacci),
`fibonacci_retracements()`, and `round_number_levels()`.

### Alignment

Indicators return lists aligned to the input, with `None` where lookback is
insufficient. Zip any two together without tracking offsets — this is where
most hand-rolled TA code goes wrong.

Wilder-smoothed indicators (RSI, ATR, ADX) seed with a simple average of the
first `period` values, matching the original formulation. Other packages seed
differently, so absolute values can differ slightly from another vendor's;
the crossings do not.

## Collateral

Prices the money-market sweep decision, including the two costs the naive
yield comparison omits.

```python
from options_desk import CollateralAssumptions, compare_scenarios, evaluate

print(evaluate(CollateralAssumptions(
    collateral=1_000_000,
    broker_credit_rate=0.005,
    fund_gross_yield=0.042,
    haircut=0.01,
    prob_forced_borrow=0.10,
)))
```

Three terms drive the answer:

1. **Gross pickup** — fund yield net of expenses, minus what the broker
   already credits on idle cash.
2. **Haircut cost** — cash pledged as collateral takes no haircut; fund shares
   take 0–2%. That is buying power you no longer have, worth whatever it earns
   you. Free if you are not buying-power constrained, expensive if you are.
3. **Forced-borrow cost** — money market funds settle T+1; an intraday margin
   call does not wait. Margin debit rates run well above fund yields, so one
   forced borrow can consume months of pickup. This is the term people leave
   out and the one that decides the answer for a leveraged book.

`compare_scenarios()` sweeps the forced-borrow probability — the input you
know least well and the one most able to flip the sign.

Two things the model assumes you have already checked:

- **Government or treasury funds only.** Institutional prime funds can impose
  liquidity fees under stress, which is exactly when collateral must be
  available.
- **Whether your clearer accepts fund shares as collateral directly.** Under
  portfolio margin or SPAN many do, and then you keep both the yield and the
  buying power, which makes the haircut term moot.

Defaults are placeholders. Replace every one with your broker's actual
numbers before drawing conclusions — broker credit rates in particular vary by
an order of magnitude and are usually tiered by balance.

## Screening: which option to sell

`screen.py` ranks candidate contracts and returns the best, or `None` when
nothing qualifies.

```python
from options_desk import Candidate, Option, PUT, best, render_screen, score_candidates

candidates = [Candidate(option=Option(spot=100, strike=92, dte=30, volatility=0.31,
                                      kind=PUT),
                        bid=0.71, ask=0.74, open_interest=1800, volume=260)]
print(render_screen(score_candidates(candidates, realized_vol=22.0)))
```

The ranking is **not** a premium sort. The central quantity is:

```
edge = credit_received − BlackScholes_price(realized_vol)
```

You collect premium priced at *implied* vol and bear risk that plays out at
*realized* vol; the difference is the variance risk premium captured on that
specific contract. Sorting on raw yield instead puts the richest-looking
contract on top exactly when the market has correctly identified a dangerous
name.

Five weighted components, all exposed in `DEFAULT_WEIGHTS` because the
ranking is only as defensible as they are: `edge` (0.35), `probability`
(0.20), `liquidity` (0.20), `technical` (0.15), `vol_context` (0.10).

Hard filters run before scoring — spread, open interest, delta, DTE bounds,
minimum credit, and positive edge. These are disqualifiers, not penalties: a
contract that cannot be exited is not a worse trade, it is one that should
not be on the list. In testing on a sample chain, the filters correctly
excluded the *highest-IV* contract because its spread was 20% of mid.

Two limits, stated in the module and in the rendered output:

- **Realized vol is backward-looking.** The edge calculation assumes
  volatility persists. It usually does, but it breaks around earnings and
  other scheduled events — in the dangerous direction.
- **A ranking is not a recommendation.** The weights are judgment calls. Two
  reasonable desks would weight these differently and pick different winners.

Credits are taken at the **bid** by default, not the mid. Mid fills are not
guaranteed, and pricing at mid overstates every edge the screen finds — worst
on exactly the illiquid contracts it should be avoiding.

`best()` returns `None` when nothing passes. That is a real answer: a screen
that always returns something will hand you a trade on the day when nothing
is worth doing.

## Scope

This is a technical and cash-management layer, not a signal generator or a
recommendation engine. It describes what price has done and what it costs to
hold collateral one way versus another. Nothing here forecasts returns, and
the interpretations are conventional readings of well-known indicators rather
than tested edges.
