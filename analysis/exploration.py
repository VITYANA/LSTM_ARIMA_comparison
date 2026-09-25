"""Reusable exploratory analysis for financial return series."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


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
