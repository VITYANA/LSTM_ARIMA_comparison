"""Shared structural interfaces for forecasting models."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class ForecastModel(Protocol):
    """Model capable of producing aligned return forecasts."""

    @property
    def name(self) -> str:
        """Return the stable model identifier."""
        ...

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Return predictions aligned with the observation index."""
        ...
