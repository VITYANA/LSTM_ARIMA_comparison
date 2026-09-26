"""Build leakage-safe standardized prediction tables."""

from __future__ import annotations

import pandas as pd

from evaluation.contracts import (
    EVALUATION_SPLITS,
    PREDICTION_COLUMNS,
    require_finite_numeric,
)
from evaluation.loading import REQUIRED_DATASET_COLUMNS
from models.interfaces import ForecastModel

ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
PREDICTION_NAME = "predicted_return"
OBSERVATION_KEY = ("ticker", "target_date", "split")


def _validate_dataset(dataset: pd.DataFrame) -> None:
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
    require_finite_numeric(dataset["target_return"], "target_return")


def _validate_predictions(predictions: object, observations: pd.DataFrame) -> pd.Series:
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


def build_predictions(dataset: pd.DataFrame, model: ForecastModel) -> pd.DataFrame:
    """Run a model on train and validation rows and standardize its output."""
    _validate_dataset(dataset)

    observations = dataset.loc[dataset["split"].isin(EVALUATION_SPLITS)].copy(deep=True)
    if observations.empty:
        raise ValueError("dataset must contain train or validation rows")
    observations = observations.sort_values(["split", "ticker", "target_date"])

    model_name = model.name
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model name must be a non-empty string")

    model_input = observations.copy(deep=True)
    predictions = _validate_predictions(model.predict(model_input), observations)

    result = observations.loc[:, ["date", "target_date", "ticker", "split"]].copy()
    result["model"] = model_name
    result["actual_return"] = observations["target_return"]
    result[PREDICTION_NAME] = predictions
    return result.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)
