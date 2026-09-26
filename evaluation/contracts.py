"""Shared contracts for forecast evaluation tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.loading import REQUIRED_DATASET_COLUMNS

ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
BPS_FACTOR = 10_000.0
EVALUATION_SPLITS = frozenset({"train", "validation"})
OBSERVATION_KEY = ("ticker", "target_date", "split")
PREDICTION_COLUMNS = (
    "date",
    "target_date",
    "ticker",
    "split",
    "model",
    "actual_return",
    "predicted_return",
)
PREDICTION_NAME = "predicted_return"
TARGET_COLUMN = "target_return"


def require_finite_numeric(series: pd.Series, label: str) -> None:
    """Require a numeric series containing only finite values."""
    if series.dtype.kind not in "iufc":
        raise ValueError(f"{label} must be numeric")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{label} must contain only finite values")


def validate_model_dataset(dataset: pd.DataFrame) -> None:
    """Validate the source dataset contract needed for evaluation."""
    missing_columns = set(REQUIRED_DATASET_COLUMNS) - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"dataset missing columns: {sorted(missing_columns)}")
    if dataset.empty:
        raise ValueError("dataset must not be empty")
    if not dataset["split"].isin(ALLOWED_SPLITS).all():
        raise ValueError("dataset contains unknown split values")
    if dataset.duplicated(list(OBSERVATION_KEY)).any():
        raise ValueError("dataset contains duplicate observation keys")
    require_finite_numeric(dataset[TARGET_COLUMN], TARGET_COLUMN)


def build_model_input(observations: pd.DataFrame) -> pd.DataFrame:
    """Return an isolated model input without future target values."""
    return observations.drop(columns=TARGET_COLUMN).copy(deep=True)


def validate_prediction_series(
    predictions: object,
    observations: pd.DataFrame,
) -> pd.Series:
    """Validate and return predictions produced by a model."""
    if not isinstance(predictions, pd.Series):
        raise ValueError("predictions must be a pandas Series")
    if predictions.name != PREDICTION_NAME:
        raise ValueError(f"prediction Series name must be {PREDICTION_NAME!r}")
    if len(predictions) != len(observations):
        raise ValueError("prediction length must match observations")
    if not predictions.index.equals(observations.index):
        raise ValueError("prediction index must match observations")
    require_finite_numeric(predictions, "predictions")
    return predictions
