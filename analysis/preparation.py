"""Build leakage-safe targets and chronological research splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.quality import add_log_returns


@dataclass(frozen=True)
class SplitConfig:
    """Inclusive target-date boundaries for chronological evaluation."""

    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


DEFAULT_SPLIT_CONFIG = SplitConfig(
    train_start="2006-01-03",
    train_end="2015-12-31",
    validation_start="2016-01-04",
    validation_end="2019-12-31",
    test_start="2020-01-02",
    test_end="2025-12-31",
)


def _validated_boundaries(config: SplitConfig) -> tuple[pd.Timestamp, ...]:
    """Parse boundaries and reject overlaps or reversed periods."""
    boundaries = tuple(pd.Timestamp(value) for value in vars(config).values())
    if list(boundaries) != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
        raise ValueError("split boundaries must be strictly increasing")
    return boundaries


def assign_time_split(
    frame: pd.DataFrame,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> pd.DataFrame:
    """Assign train, validation, or test from each target date."""
    train_start, train_end, validation_start, validation_end, test_start, test_end = (
        _validated_boundaries(config)
    )
    result = frame.copy()
    target_dates = pd.to_datetime(result["target_date"])
    split = pd.Series(pd.NA, index=result.index, dtype="string")
    split.loc[target_dates.between(train_start, train_end)] = "train"
    split.loc[target_dates.between(validation_start, validation_end)] = "validation"
    split.loc[target_dates.between(test_start, test_end)] = "test"
    if split.isna().any():
        dates = sorted(target_dates.loc[split.isna()].dt.strftime("%Y-%m-%d").unique())
        raise ValueError(f"target dates outside configured splits: {dates}")
    result["target_date"] = target_dates
    result["split"] = split
    return result


def build_model_dataset(
    prices: pd.DataFrame,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> pd.DataFrame:
    """Create current-return features and next-trading-day targets."""
    result = add_log_returns(prices).sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = result.groupby("ticker", sort=False)
    result["target_date"] = grouped["date"].shift(-1)
    result["target_return"] = grouped["log_return"].shift(-1)
    result = result.dropna(subset=["log_return", "target_date", "target_return"]).copy()
    numeric = result[["log_return", "target_return"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("model returns must be finite")
    result = assign_time_split(result, config)
    columns = [
        "date",
        "target_date",
        "ticker",
        "adj_close",
        "log_return",
        "target_return",
        "split",
    ]
    return result.loc[:, columns].reset_index(drop=True)
