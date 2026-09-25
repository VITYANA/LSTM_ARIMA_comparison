"""Data-quality diagnostics and basic return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Build one quality-summary row for each ticker."""
    dated = prices.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    records: list[dict[str, object]] = []
    for ticker, group in dated.groupby("ticker", sort=True):
        records.append(
            {
                "ticker": str(ticker),
                "observations": len(group),
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
                "missing_values": int(group.isna().sum().sum()),
                "duplicate_dates": int(group["date"].duplicated().sum()),
                "dividend_events": int(group["dividends"].ne(0).sum()),
                "split_events": int(group["stock_splits"].ne(0).sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def shared_calendar(prices: pd.DataFrame) -> bool:
    """Return whether all tickers contain the same dates."""
    calendars = [
        pd.Index(group["date"].drop_duplicates()).sort_values()
        for _, group in prices.groupby("ticker", sort=True)
    ]
    return not calendars or all(calendar.equals(calendars[0]) for calendar in calendars[1:])


def add_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add adjusted-price log returns within each ticker."""
    if prices["adj_close"].le(0).any():
        raise ValueError("Adjusted prices must be positive for logarithmic returns")

    result = prices.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    log_prices = np.log(result["adj_close"])
    result["log_return"] = log_prices.groupby(result["ticker"]).diff()
    return result
