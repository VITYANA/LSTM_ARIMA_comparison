"""Naive forecasting baselines."""

from __future__ import annotations

import pandas as pd


def predict_zero(observations: pd.DataFrame) -> pd.Series:
    """Predict a zero return for every observation."""
    return pd.Series(
        0.0,
        index=observations.index,
        dtype="float64",
        name="predicted_return",
    )
