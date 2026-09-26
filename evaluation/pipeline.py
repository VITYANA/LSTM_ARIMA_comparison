"""Build leakage-safe standardized prediction tables."""

from __future__ import annotations

import pandas as pd

from evaluation.contracts import (
    EVALUATION_SPLITS,
    PREDICTION_COLUMNS,
    PREDICTION_NAME,
    TARGET_COLUMN,
    build_model_input,
    validate_model_dataset,
    validate_prediction_series,
)
from models.interfaces import ForecastModel, validate_model_name


def build_predictions(dataset: pd.DataFrame, model: ForecastModel) -> pd.DataFrame:
    """Run a model on train and validation rows and standardize its output."""
    validate_model_dataset(dataset)

    observations = dataset.loc[dataset["split"].isin(EVALUATION_SPLITS)].copy(deep=True)
    if observations.empty:
        raise ValueError("dataset must contain train or validation rows")
    observations = observations.sort_values(["split", "ticker", "target_date"])

    model_name = validate_model_name(model.name)

    model_input = build_model_input(observations)
    predictions = validate_prediction_series(model.predict(model_input), model_input)

    result = observations.loc[:, ["date", "target_date", "ticker", "split"]].copy()
    result["model"] = model_name
    result["actual_return"] = observations[TARGET_COLUMN]
    result[PREDICTION_NAME] = predictions
    return result.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)
