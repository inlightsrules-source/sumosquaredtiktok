"""Data-source adapters.

Everything above this module works on a ``Series`` and nothing else, so
supporting a new provider means writing one function here. Adapters that
need a third-party client import it lazily, keeping the core dependency-free.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .ohlcv import Bar, Series

# Header spellings seen across common vendor exports.
_FIELD_ALIASES = {
    "day": ("date", "day", "timestamp", "time", "datetime"),
    "open": ("open", "o", "adj open", "adj_open"),
    "high": ("high", "h", "adj high", "adj_high"),
    "low": ("low", "l", "adj low", "adj_low"),
    "close": ("close", "c", "adj close", "adj_close", "close/last", "adjusted_close"),
    "volume": ("volume", "v", "vol"),
}


def from_csv(path: str | Path, symbol: str | None = None) -> Series:
    """Load a CSV with a header row, matching columns case-insensitively.

    Handles the usual vendor spellings (``Adj Close``, ``Close/Last``, ...)
    and strips currency symbols and thousands separators from prices.
    """
    path = Path(path)
    symbol = symbol or path.stem.upper()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    mapping = _resolve_columns(rows[0].keys())
    return Series(symbol, (_row_to_bar(row, mapping) for row in rows))


def from_dicts(symbol: str, rows: Iterable[Mapping]) -> Series:
    """Load from dicts -- the shape most JSON APIs return."""
    rows = list(rows)
    if not rows:
        raise ValueError(f"{symbol}: no rows")
    mapping = _resolve_columns(rows[0].keys())
    return Series(symbol, (_row_to_bar(row, mapping) for row in rows))


def from_columns(
    symbol: str,
    dates: Sequence,
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float] | None = None,
) -> Series:
    """Load from parallel arrays -- the shape dataframe libraries hand back."""
    lengths = {
        "dates": len(dates), "opens": len(opens), "highs": len(highs),
        "lows": len(lows), "closes": len(closes),
    }
    if volumes is not None:
        lengths["volumes"] = len(volumes)
    if len(set(lengths.values())) > 1:
        raise ValueError(f"{symbol}: column length mismatch: {lengths}")
    bars = []
    for i in range(len(dates)):
        bars.append(
            Bar(
                day=_coerce_date(dates[i]),
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                volume=float(volumes[i]) if volumes is not None else None,
            )
        )
    return Series(symbol, bars)


def from_yfinance(symbol: str, period: str = "2y", interval: str = "1d") -> Series:
    """Fetch via ``yfinance`` if it is installed.

    Convenient for exploration. Its chain data is not reliable enough to
    trade implied vol off, so use a real feed for anything that touches
    execution.
    """
    try:
        import yfinance  # noqa: PLC0415  (optional dependency, imported on use)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "from_yfinance needs the yfinance package (pip install yfinance), "
            "or use from_csv/from_columns with your own feed."
        ) from exc
    frame = yfinance.Ticker(symbol).history(period=period, interval=interval)
    if frame.empty:
        raise ValueError(f"{symbol}: yfinance returned no rows")
    return from_columns(
        symbol,
        [d.date() for d in frame.index],
        frame["Open"].tolist(),
        frame["High"].tolist(),
        frame["Low"].tolist(),
        frame["Close"].tolist(),
        frame["Volume"].tolist() if "Volume" in frame else None,
    )


def _resolve_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    """Map canonical field names onto this file's actual headers."""
    lookup = {name.strip().lower(): name for name in fieldnames if name}
    resolved: dict[str, str] = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                resolved[canonical] = lookup[alias]
                break
    missing = {"day", "open", "high", "low", "close"} - resolved.keys()
    if missing:
        raise ValueError(
            f"missing required columns {sorted(missing)}; saw {sorted(lookup)}"
        )
    return resolved


def _row_to_bar(row: Mapping, mapping: dict[str, str]) -> Bar:
    volume_key = mapping.get("volume")
    raw_volume = row.get(volume_key) if volume_key else None
    return Bar(
        day=_coerce_date(row[mapping["day"]]),
        open=_coerce_price(row[mapping["open"]]),
        high=_coerce_price(row[mapping["high"]]),
        low=_coerce_price(row[mapping["low"]]),
        close=_coerce_price(row[mapping["close"]]),
        volume=_coerce_volume(raw_volume),
    )


def _coerce_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(f"unrecognised date {value!r}") from exc


def _coerce_price(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().lstrip("$").replace(",", "")
    if not cleaned:
        raise ValueError("empty price field")
    return float(cleaned)


def _coerce_volume(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().replace(",", "")
    return float(cleaned) if cleaned else None
