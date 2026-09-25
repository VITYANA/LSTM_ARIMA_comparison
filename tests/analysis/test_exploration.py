"""Tests for reusable exploratory-analysis calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.exploration import summarize_returns


@pytest.fixture
def returns() -> pd.DataFrame:
    """Return deterministic histories for two assets"""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "date": [*dates, *dates],
            "ticker": ["AAA"] * 5 + ["BBB"] * 5,
            "adj_close": [100, 101, 100, 102, 104, 50, 49, 51, 50, 52],
            "log_return": [
                np.nan,
                0.01,
                -0.01,
                0.02,
                0.02,
                np.nan,
                -0.02,
                0.04,
                -0.02,
                0.04,
            ],
        }
    )


def test_summarize_returns_reports_each_ticker(returns: pd.DataFrame) -> None:
    """Summarize distributions volatility and drawdowns by ticker"""
    result = summarize_returns(
        returns.sample(frac=1, random_state=7),
        annualization_periods=4,
    )

    assert result["ticker"].tolist() == ["AAA", "BBB"]
    assert result["observations"].tolist() == [4, 4]
    assert result.loc[0, "mean_daily"] == pytest.approx(0.01)
    assert result.loc[0, "mean_annualized"] == pytest.approx(0.04)
    assert result.loc[0, "volatility_annualized"] == pytest.approx(
        np.std([0.01, -0.01, 0.02, 0.02], ddof=1) * 2
    )
    assert result.loc[0, "maximum_drawdown"] == pytest.approx(100 / 101 - 1)


@pytest.mark.parametrize("annualization_periods", [0, -252])
def test_summarize_returns_rejects_invalid_annualization(
    returns: pd.DataFrame,
    annualization_periods: int,
) -> None:
    """Reject nonpositive annualization constants"""
    with pytest.raises(ValueError, match="annualization_periods"):
        summarize_returns(returns, annualization_periods)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_summarize_returns_rejects_unusable_ticker_returns(
    returns: pd.DataFrame,
    invalid_value: float,
) -> None:
    """Reject missing or infinite ticker return histories"""
    returns.loc[returns["ticker"].eq("AAA"), "log_return"] = invalid_value

    with pytest.raises(ValueError, match="AAA: log returns must be finite"):
        summarize_returns(returns)


def test_summarize_returns_does_not_mutate_source(returns: pd.DataFrame) -> None:
    """Preserve the source frame and row ordering"""
    original = returns.copy(deep=True)

    summarize_returns(returns)

    pd.testing.assert_frame_equal(returns, original)
