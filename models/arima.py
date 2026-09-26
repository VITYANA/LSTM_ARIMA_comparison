"""Fitted ARIMA forecasting models and fitters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Self, cast

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA  # type: ignore[import-untyped]

ARIMAOrder = tuple[int, int, int]
PREDICTION_NAME = "predicted_return"
RETURN_COLUMN = "log_return"


class _ARIMAResult(Protocol):
    """Minimal fitted-result surface required for forecasting."""

    @property
    def mle_retvals(self) -> Mapping[str, object]:
        """Return optimizer metadata."""
        ...

    def forecast(self, steps: int) -> object:
        """Return forecasts for the requested horizon."""
        ...


def _validate_model_name(name: object) -> str:
    """Return a non-empty model identifier."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name must be a non-empty string")
    return name


def _validate_order(order: object) -> ARIMAOrder:
    """Return a supported stationary ARIMA order."""
    if not isinstance(order, tuple) or len(order) != 3:
        raise ValueError("ARIMA order must contain exactly three integers")
    if any(type(value) is not int for value in order):
        raise ValueError("ARIMA order values must be integers")
    p, d, q = cast(ARIMAOrder, order)
    if p < 0 or d < 0 or q < 0:
        raise ValueError("ARIMA order values must be non-negative")
    if d != 0:
        raise ValueError("ARIMA order must use d=0 for stationary returns")
    return p, d, q


def _return_values(history: pd.DataFrame) -> np.ndarray:
    """Extract a finite numeric return history."""
    if RETURN_COLUMN not in history.columns:
        raise ValueError(f"history missing column {RETURN_COLUMN!r}")
    if history.empty:
        raise ValueError("history must not be empty")
    returns = history[RETURN_COLUMN]
    if returns.dtype.kind not in "iuf":
        raise ValueError(f"{RETURN_COLUMN} must be numeric")
    values = returns.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError(f"{RETURN_COLUMN} must contain only finite values")
    return values


class ARIMAModel:
    """Adapt one fitted statsmodels result to ForecastModel."""

    def __init__(self, fitted_result: _ARIMAResult, *, name: str = "arima") -> None:
        self._fitted_result = fitted_result
        self._name = _validate_model_name(name)

    @classmethod
    def fit(
        cls,
        history: pd.DataFrame,
        order: ARIMAOrder,
        *,
        name: str = "arima",
    ) -> Self:
        """Fit and return one stationary ARIMA model."""
        validated_order = _validate_order(order)
        validated_name = _validate_model_name(name)
        values = _return_values(history)

        try:
            fitted = StatsmodelsARIMA(
                values,
                order=validated_order,
                trend="c",
            ).fit()
        except Exception as error:
            raise ValueError(f"ARIMA order {validated_order} fit failed: {error}") from error

        result = cast(_ARIMAResult, fitted)
        if result.mle_retvals.get("converged") is not True:
            raise ValueError(f"ARIMA order {validated_order} did not converge")
        return cls(result, name=validated_name)

    @property
    def name(self) -> str:
        """Return the stable model identifier."""
        return self._name

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Forecast returns aligned with observation rows."""
        if len(observations) == 0:
            return pd.Series(
                index=observations.index,
                dtype="float64",
                name=PREDICTION_NAME,
            )

        forecast = np.asarray(self._fitted_result.forecast(steps=len(observations)))
        values = forecast.reshape(-1)
        if len(values) != len(observations):
            raise ValueError("forecast length must match observations")
        if values.dtype.kind not in "iuf":
            raise ValueError("forecast values must be numeric")
        numeric = values.astype("float64")
        if not np.isfinite(numeric).all():
            raise ValueError("forecast values must contain only finite values")
        return pd.Series(
            numeric,
            index=observations.index,
            dtype="float64",
            name=PREDICTION_NAME,
        )


class ARIMAFitter:
    """Fit ticker-specific fixed ARIMA orders."""

    def __init__(
        self,
        orders: Mapping[str, ARIMAOrder],
        model_name: str = "arima",
    ) -> None:
        self._orders = {ticker: _validate_order(order) for ticker, order in orders.items()}
        self._model_name = _validate_model_name(model_name)

    def __call__(self, ticker: str, history: pd.DataFrame) -> ARIMAModel:
        """Fit the configured order for one ticker."""
        try:
            order = self._orders[ticker]
        except KeyError as error:
            raise ValueError(f"ticker {ticker!r} has no configured ARIMA order") from error
        return ARIMAModel.fit(history, order, name=self._model_name)
