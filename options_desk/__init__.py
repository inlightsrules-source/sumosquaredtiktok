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
from .pricing import (
    CALL,
    PUT,
    Greeks,
    Option,
    Probabilities,
    delta_to_strike,
    expected_move,
    greeks,
    implied_volatility,
    price,
    probabilities,
    probability_of_touch,
)
from .report import render
from .returns import (
    ReturnProfile,
    annualize_compound,
    annualize_simple,
    cash_secured_put,
    covered_call,
    credit_spread,
    naked_option,
    with_collateral_yield,
)
from .trade import TradeView, evaluate_trade, render_trade
from .vol import VolContext, build_context, iv_percentile, iv_rank, term_structure_slope

__all__ = [
    "CALL",
    "FUND_TYPES",
    "PUT",
    "Analysis",
    "Bar",
    "CollateralAssumptions",
    "CollateralResult",
    "Greeks",
    "Level",
    "Option",
    "Pivot",
    "Probabilities",
    "Reading",
    "ReturnProfile",
    "Series",
    "TradeView",
    "TrendView",
    "VolContext",
    "analyze",
    "annualize_compound",
    "annualize_simple",
    "build_context",
    "cash_secured_put",
    "cluster_levels",
    "compare_scenarios",
    "covered_call",
    "credit_spread",
    "delta_to_strike",
    "evaluate",
    "evaluate_trade",
    "expected_move",
    "fibonacci_retracements",
    "find_pivots",
    "from_rows",
    "greeks",
    "implied_volatility",
    "iv_percentile",
    "iv_rank",
    "most_significant",
    "naked_option",
    "nearest",
    "pivot_points",
    "price",
    "probabilities",
    "probability_of_touch",
    "render",
    "render_trade",
    "round_number_levels",
    "term_structure_slope",
    "with_collateral_yield",
]
