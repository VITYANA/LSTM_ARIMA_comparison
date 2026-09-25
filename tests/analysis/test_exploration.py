"""Tests for reusable exploratory-analysis calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.exploration import (
    add_rolling_volatility,
    return_correlation,
    summarize_returns,
)


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


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("pearson", -1 / np.sqrt(6)),
        ("spearman", -1 / np.sqrt(18)),
    ],
)
def test_return_correlation_calculates_aligned_asset_relationships(
    returns: pd.DataFrame,
    method: str,
    expected: float,
) -> None:
    """Calculate hand-checked aligned return correlations"""
    result = return_correlation(returns.sample(frac=1, random_state=11), method=method)

    assert result.index.tolist() == ["AAA", "BBB"]
    assert result.columns.tolist() == ["AAA", "BBB"]
    assert result.loc["AAA", "AAA"] == pytest.approx(1.0)
    assert result.loc["BBB", "BBB"] == pytest.approx(1.0)
    assert result.loc["AAA", "BBB"] == pytest.approx(expected)
    assert result.loc["BBB", "AAA"] == pytest.approx(expected)


@pytest.mark.parametrize("method", ["kendall", "invalid"])
def test_return_correlation_rejects_unsupported_method(
    returns: pd.DataFrame,
    method: str,
) -> None:
    """Reject correlation methods outside the protocol"""
    with pytest.raises(ValueError, match="pearson or spearman"):
        return_correlation(returns, method=method)


def test_add_rolling_volatility_does_not_cross_tickers(
    returns: pd.DataFrame,
) -> None:
    """Calculate trailing volatility independently per ticker"""
    result = add_rolling_volatility(
        returns.sample(frac=1, random_state=7),
        windows=(2, 3),
        annualization_periods=4,
    )

    first_rows = result.groupby("ticker", sort=False).head(1)
    assert first_rows[["volatility_2d", "volatility_3d"]].isna().all().all()
    aaa = result.loc[result["ticker"].eq("AAA")].reset_index(drop=True)
    assert aaa.loc[2, "volatility_2d"] == pytest.approx(np.std([0.01, -0.01], ddof=1) * 2)
    assert aaa.loc[3, "volatility_3d"] == pytest.approx(np.std([0.01, -0.01, 0.02], ddof=1) * 2)


def test_add_rolling_volatility_is_prefix_invariant(returns: pd.DataFrame) -> None:
    """Prevent future observations changing earlier volatility"""
    prefix = returns.groupby("ticker", group_keys=False).head(4)
    prefix_result = add_rolling_volatility(prefix, windows=(2,), annualization_periods=1)
    full_result = add_rolling_volatility(returns, windows=(2,), annualization_periods=1)
    comparable = full_result.groupby("ticker", group_keys=False).head(4)

    pd.testing.assert_series_equal(
        prefix_result["volatility_2d"].reset_index(drop=True),
        comparable["volatility_2d"].reset_index(drop=True),
    )


@pytest.mark.parametrize("annualization_periods", [0, -252])
def test_add_rolling_volatility_rejects_invalid_annualization(
    returns: pd.DataFrame,
    annualization_periods: int,
) -> None:
    """Reject nonpositive volatility annualization constants"""
    with pytest.raises(ValueError, match="annualization_periods"):
        add_rolling_volatility(
            returns,
            annualization_periods=annualization_periods,
        )


@pytest.mark.parametrize("windows", [(), (1,), (2, 2)])
def test_add_rolling_volatility_rejects_invalid_windows(
    returns: pd.DataFrame,
    windows: tuple[int, ...],
) -> None:
    """Reject empty short or duplicate rolling windows"""
    with pytest.raises(ValueError, match="windows"):
        add_rolling_volatility(returns, windows=windows)


def test_add_rolling_volatility_does_not_mutate_source(returns: pd.DataFrame) -> None:
    """Preserve source values columns and row ordering"""
    original = returns.copy(deep=True)

    add_rolling_volatility(returns, windows=(2,))

    pd.testing.assert_frame_equal(returns, original)
