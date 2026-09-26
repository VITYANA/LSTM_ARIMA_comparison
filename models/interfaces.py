"""Shared structural interfaces for forecasting models."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


def validate_model_name(name: object) -> str:
    """Return a non-empty model identifier."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name must be a non-empty string")
    return name


class ForecastModel(Protocol):
    """Model capable of producing aligned return forecasts."""

    @property
    def name(self) -> str:
        """Return the stable model identifier."""
        ...

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Return predictions aligned with the observation index."""
        ...
