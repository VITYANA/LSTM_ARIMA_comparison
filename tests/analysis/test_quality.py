"""Tests for reusable data-quality calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.quality import add_log_returns, build_quality_report, shared_calendar

PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
]


@pytest.fixture
def prices() -> pd.DataFrame:
    """Return two small ticker histories with aligned dates"""
    return pd.DataFrame(
        [
            ["2020-01-02", "AAA", 10, 12, 9, 11, 11, 100, 0.0, 0.0],
            ["2020-01-03", "AAA", 11, 13, 10, 12, 12, 110, 0.2, 0.0],
            ["2020-01-02", "BBB", 20, 22, 19, 21, 21, 200, 0.0, 0.0],
            ["2020-01-03", "BBB", 21, 23, 20, 22, 22, 210, 0.0, 2.0],
        ],
        columns=PRICE_COLUMNS,
    )


def test_build_quality_report_summarizes_each_ticker(prices: pd.DataFrame) -> None:
    """Summarizes dates, observations, actions, and defects"""
    prices.loc[1, "volume"] = np.nan
    prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)

    report = build_quality_report(prices)

    assert report.to_dict(orient="records") == [
        {
            "ticker": "AAA",
            "observations": 3,
            "first_date": pd.Timestamp("2020-01-02"),
            "last_date": pd.Timestamp("2020-01-03"),
            "missing_values": 1,
            "duplicate_dates": 1,
            "dividend_events": 1,
            "split_events": 0,
        },
        {
            "ticker": "BBB",
            "observations": 2,
            "first_date": pd.Timestamp("2020-01-02"),
            "last_date": pd.Timestamp("2020-01-03"),
            "missing_values": 0,
            "duplicate_dates": 0,
            "dividend_events": 0,
            "split_events": 1,
        },
    ]


@pytest.mark.parametrize(
    ("dates_by_ticker", "expected"),
    [
        (
            {
                "AAA": ["2020-01-02", "2020-01-03"],
                "BBB": ["2020-01-02", "2020-01-03"],
            },
            True,
        ),
        (
            {
                "AAA": ["2020-01-02", "2020-01-03"],
                "BBB": ["2020-01-02", "2020-01-06"],
            },
            False,
        ),
        ({"AAA": ["2020-01-02", "2020-01-03"]}, True),
        ({}, True),
    ],
)
def test_shared_calendar_detects_date_mismatches(
    dates_by_ticker: dict[str, list[str]], expected: bool
) -> None:
    """Detects whether ticker trading calendars match"""
    frame = pd.DataFrame(
        [(ticker, date) for ticker, dates in dates_by_ticker.items() for date in dates],
        columns=["ticker", "date"],
    )
    frame["date"] = pd.to_datetime(frame["date"])

    assert shared_calendar(frame) is expected


def test_add_log_returns_calculates_within_each_ticker(prices: pd.DataFrame) -> None:
    """Calculates adjusted-price returns without crossing tickers"""
    result = add_log_returns(prices)

    expected = [np.nan, np.log(12 / 11), np.nan, np.log(22 / 21)]
    np.testing.assert_allclose(result["log_return"], expected, equal_nan=True)
    assert "log_return" not in prices.columns


@pytest.mark.parametrize("invalid_price", [0.0, -1.0])
def test_add_log_returns_rejects_non_positive_prices(
    prices: pd.DataFrame, invalid_price: float
) -> None:
    """Rejects prices incompatible with logarithmic returns"""
    prices.loc[0, "adj_close"] = invalid_price

    with pytest.raises(ValueError, match="positive"):
        add_log_returns(prices)
