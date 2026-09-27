"""Compose aligned ARIMA and residual LSTM forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.preparation import DEFAULT_SPLIT_CONFIG
from evaluation.contracts import PREDICTION_COLUMNS, require_finite_numeric
from models.interfaces import validate_model_name

HYBRID_MODEL_NAME = "arima_lstm"
RESIDUAL_MODEL_NAME = "residual_lstm"
ARIMA_MODEL_NAME = "arima"
ALIGNMENT_COLUMNS = ("date", "target_date", "ticker", "split")
SORT_COLUMNS = ("ticker", "target_date", "date", "split")
RESIDUAL_IDENTITY_ATOL = 1e-12


def _validated_validation_end(value: str | pd.Timestamp) -> pd.Timestamp:
    """Return one finite timezone-naive validation boundary."""
    try:
        boundary = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("validation end must be a valid datetime") from error
    if pd.isna(boundary) or boundary.tz is not None:
        raise ValueError("validation end must be a finite timezone-naive datetime")
    return boundary


def _validated_component(
    predictions: pd.DataFrame,
    *,
    label: str,
    expected_model: str,
) -> pd.DataFrame:
    """Return one isolated sorted validation component."""
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError(f"{label} predictions must use standard columns in standard order")
    if predictions.empty:
        raise ValueError(f"{label} predictions must not be empty")
    for column in ("date", "target_date"):
        if not pd.api.types.is_datetime64_any_dtype(predictions[column]):
            raise ValueError(f"{label} prediction {column} must be datetime")
        if predictions[column].isna().any():
            raise ValueError(f"{label} predictions contain missing dates")
    if not predictions["split"].eq("validation").all():
        raise ValueError(f"{label} predictions must contain validation rows only")
    if not predictions["model"].eq(expected_model).all():
        raise ValueError(f"{label} predictions must use model {expected_model!r}")
    if predictions.duplicated(list(ALIGNMENT_COLUMNS)).any():
        raise ValueError(f"{label} predictions contain duplicate observation keys")
    require_finite_numeric(predictions["actual_return"], f"{label} actual_return")
    require_finite_numeric(predictions["predicted_return"], f"{label} predicted_return")
    return (
        predictions.loc[:, PREDICTION_COLUMNS]
        .sort_values(list(SORT_COLUMNS))
        .reset_index(drop=True)
        .copy(deep=True)
    )


def build_arima_lstm_predictions(
    arima_predictions: pd.DataFrame,
    residual_predictions: pd.DataFrame,
    model_name: str = HYBRID_MODEL_NAME,
    *,
    validation_end: str | pd.Timestamp = DEFAULT_SPLIT_CONFIG.validation_end,
) -> pd.DataFrame:
    """Add aligned residual forecasts to raw ARIMA forecasts."""
    hybrid_name = validate_model_name(model_name)
    validated_end = _validated_validation_end(validation_end)
    arima = _validated_component(
        arima_predictions,
        label="ARIMA",
        expected_model=ARIMA_MODEL_NAME,
    )
    residual = _validated_component(
        residual_predictions,
        label="residual",
        expected_model=RESIDUAL_MODEL_NAME,
    )
    if arima["target_date"].gt(validated_end).any():
        raise ValueError("ARIMA predictions contain targets after validation end")

    arima_keys = arima.loc[:, ALIGNMENT_COLUMNS]
    residual_keys = residual.loc[:, ALIGNMENT_COLUMNS]
    if not arima_keys.equals(residual_keys):
        raise ValueError("ARIMA and residual prediction keys do not align")

    actual = arima["actual_return"].to_numpy(dtype="float64")
    arima_forecast = arima["predicted_return"].to_numpy(dtype="float64")
    realized_residual = residual["actual_return"].to_numpy(dtype="float64")
    expected_residual = actual - arima_forecast
    if not np.allclose(
        realized_residual,
        expected_residual,
        rtol=0.0,
        atol=RESIDUAL_IDENTITY_ATOL,
    ):
        raise ValueError("residual identity does not match actual_return minus ARIMA forecast")

    residual_forecast = residual["predicted_return"].to_numpy(dtype="float64")
    with np.errstate(over="ignore", invalid="ignore"):
        hybrid_forecast = arima_forecast + residual_forecast
    if not np.isfinite(hybrid_forecast).all():
        raise ValueError("hybrid predictions must contain only finite values")

    result = arima.loc[:, ALIGNMENT_COLUMNS].copy()
    result["model"] = hybrid_name
    result["actual_return"] = actual
    result["predicted_return"] = hybrid_forecast
    return result.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)
