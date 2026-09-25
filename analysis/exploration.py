"""Reusable exploratory analysis for financial return series."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
SUPPORTED_CORRELATIONS = ("pearson", "spearman")
CorrelationMethod = Literal["pearson", "spearman"]


def summarize_returns(
    frame: pd.DataFrame,
    annualization_periods: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Summarize return distributions and price drawdowns by ticker."""
    if annualization_periods < 1:
        raise ValueError("annualization_periods must be positive")

    records: list[dict[str, float | int | str]] = []
    ordered = frame.copy().sort_values(["ticker", "date"])
    for ticker, group in ordered.groupby("ticker", sort=True):
        values = group["log_return"].dropna()
        if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{ticker}: log returns must be finite")
        drawdown = group["adj_close"].div(group["adj_close"].cummax()).sub(1)
        records.append(
            {
                "ticker": str(ticker),
                "observations": len(values),
                "mean_daily": float(values.mean()),
                "median_daily": float(values.median()),
                "volatility_daily": float(values.std()),
                "mean_annualized": float(values.mean() * annualization_periods),
                "volatility_annualized": float(values.std() * np.sqrt(annualization_periods)),
                "skewness": cast(float, values.skew()),
                "excess_kurtosis": cast(float, values.kurt()),
                "minimum": float(values.min()),
                "q01": float(values.quantile(0.01)),
                "q05": float(values.quantile(0.05)),
                "median": float(values.quantile(0.50)),
                "q95": float(values.quantile(0.95)),
                "q99": float(values.quantile(0.99)),
                "maximum": float(values.max()),
                "maximum_drawdown": float(drawdown.min()),
            }
        )
    return pd.DataFrame.from_records(records)


def return_correlation(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Calculate an aligned cross-asset return-correlation matrix."""
    if method not in SUPPORTED_CORRELATIONS:
        raise ValueError("method must be pearson or spearman")
    wide = frame.pivot(index="date", columns="ticker", values="log_return")
    return wide.corr(method=cast(CorrelationMethod, method))


def add_rolling_volatility(
    frame: pd.DataFrame,
    windows: Sequence[int] = (21, 63),
    annualization_periods: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Add trailing annualized volatility without crossing tickers."""
    if annualization_periods < 1:
        raise ValueError("annualization_periods must be positive")
    if not windows or any(window < 2 for window in windows):
        raise ValueError("windows must contain integers greater than one")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must not contain duplicates")

    result = frame.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = result.groupby("ticker", sort=False)["log_return"]
    for window in windows:
        rolling = grouped.rolling(window=window, min_periods=window).std()
        result[f"volatility_{window}d"] = rolling.reset_index(
            level=0, drop=True
        ).sort_index() * np.sqrt(annualization_periods)
    return result
