"""Technical analysis and collateral tooling for the options desk.

Pure stdlib -- no numpy, no pandas, no vendor SDK required.

    from options_desk import adapters, analyze, render

    series = adapters.from_csv("data/AAPL.csv")
    print(render(analyze(series)))

Collateral:

    from options_desk import CollateralAssumptions, evaluate

    print(evaluate(CollateralAssumptions(collateral=500_000)))
"""

from .collateral import (
    FUND_TYPES,
    CollateralAssumptions,
    CollateralResult,
    compare_scenarios,
    evaluate,
)
from .interpret import Analysis, Reading, TrendView, analyze
from .levels import (
    Level,
    Pivot,
    cluster_levels,
    fibonacci_retracements,
    find_pivots,
    most_significant,
    nearest,
    pivot_points,
    round_number_levels,
)
from .ohlcv import Bar, Series, from_rows
from .report import render

__all__ = [
    "FUND_TYPES",
    "Analysis",
    "Bar",
    "CollateralAssumptions",
    "CollateralResult",
    "Level",
    "Pivot",
    "Reading",
    "Series",
    "TrendView",
    "analyze",
    "cluster_levels",
    "compare_scenarios",
    "evaluate",
    "fibonacci_retracements",
    "find_pivots",
    "from_rows",
    "most_significant",
    "nearest",
    "pivot_points",
    "render",
    "round_number_levels",
]
