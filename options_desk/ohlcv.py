"""Price series container.

Deliberately vendor-neutral: every analysis function in this package takes a
``Series`` and nothing else, so the data source is a detail you swap at the
edge. Adapters live in ``adapters.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar. Volume is optional -- some feeds omit it for indices."""

    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"{self.day}: high {self.high} below low {self.low}")
        for name in ("open", "close"):
            price = getattr(self, name)
            if not (self.low <= price <= self.high):
                raise ValueError(
                    f"{self.day}: {name} {price} outside range [{self.low}, {self.high}]"
                )
        if self.volume is not None and self.volume < 0:
            raise ValueError(f"{self.day}: negative volume {self.volume}")

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0


class Series:
    """An ordered, deduplicated run of daily bars for one symbol.

    Bars are sorted by date on construction and duplicate dates are rejected
    rather than silently collapsed -- a duplicate almost always means two
    feeds got stitched together badly, and quietly dropping one produces
    indicator values that look plausible and are wrong.
    """

    def __init__(self, symbol: str, bars: Iterable[Bar]):
        ordered = sorted(bars, key=lambda b: b.day)
        seen: set[date] = set()
        for bar in ordered:
            if bar.day in seen:
                raise ValueError(f"{symbol}: duplicate bar for {bar.day}")
            seen.add(bar.day)
        if not ordered:
            raise ValueError(f"{symbol}: no bars")
        self.symbol = symbol
        self.bars: tuple[Bar, ...] = tuple(ordered)

    def __len__(self) -> int:
        return len(self.bars)

    def __getitem__(self, index):  # supports both int and slice
        if isinstance(index, slice):
            return Series(self.symbol, self.bars[index])
        return self.bars[index]

    def __iter__(self):
        return iter(self.bars)

    def __repr__(self) -> str:
        return (
            f"<Series {self.symbol} {len(self)} bars "
            f"{self.bars[0].day}..{self.bars[-1].day}>"
        )

    # -- column accessors -------------------------------------------------

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    @property
    def highs(self) -> list[float]:
        return [b.high for b in self.bars]

    @property
    def lows(self) -> list[float]:
        return [b.low for b in self.bars]

    @property
    def opens(self) -> list[float]:
        return [b.open for b in self.bars]

    @property
    def volumes(self) -> list[float]:
        """Missing volume reads as 0.0 so volume indicators stay total.

        Check ``has_volume`` before trusting anything derived from this.
        """
        return [0.0 if b.volume is None else b.volume for b in self.bars]

    @property
    def dates(self) -> list[date]:
        return [b.day for b in self.bars]

    @property
    def has_volume(self) -> bool:
        return all(b.volume is not None for b in self.bars)

    @property
    def last(self) -> Bar:
        return self.bars[-1]

    def tail(self, n: int) -> "Series":
        return Series(self.symbol, self.bars[-n:])

    def returns(self, log: bool = True) -> list[float]:
        """Bar-over-bar returns. Length is ``len(self) - 1``."""
        import math

        out: list[float] = []
        closes = self.closes
        for prev, cur in zip(closes, closes[1:]):
            if prev <= 0:
                raise ValueError(f"{self.symbol}: non-positive close {prev}")
            out.append(math.log(cur / prev) if log else (cur / prev) - 1.0)
        return out


def from_rows(symbol: str, rows: Sequence[Sequence]) -> Series:
    """Build a Series from ``(date, open, high, low, close[, volume])`` rows.

    The generic escape hatch: whatever your provider returns, shape it into
    rows and this takes it.
    """
    bars = []
    for row in rows:
        if len(row) not in (5, 6):
            raise ValueError(f"row needs 5 or 6 fields, got {len(row)}: {row!r}")
        day = row[0]
        if isinstance(day, str):
            day = date.fromisoformat(day)
        volume = float(row[5]) if len(row) == 6 and row[5] is not None else None
        bars.append(
            Bar(
                day=day,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=volume,
            )
        )
    return Series(symbol, bars)
