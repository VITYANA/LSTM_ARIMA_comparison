"""Naive forecasting baselines."""

from __future__ import annotations

import pandas as pd


class NaiveModel:
    """Forecast zero next-period return for every observation."""

    @property
    def name(self) -> str:
        """Return the stable baseline identifier."""
        return "naive_zero"

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Predict a zero return for every observation."""
        return pd.Series(
            0.0,
            index=observations.index,
            dtype="float64",
            name="predicted_return",
        )
