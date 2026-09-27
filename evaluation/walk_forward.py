"""Run leakage-safe expanding-window validation forecasts."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from evaluation.contracts import (
    PREDICTION_COLUMNS,
    PREDICTION_NAME,
    TARGET_COLUMN,
    build_model_input,
    require_finite_numeric,
    validate_model_dataset,
    validate_prediction_series,
)
from models.interfaces import ForecastModel, validate_model_name

ModelFitter = Callable[[str, pd.DataFrame], ForecastModel]
ProgressCallback = Callable[[int, int], None]
RETURN_COLUMN = "log_return"
VALIDATION_SPLIT = "validation"


def build_walk_forward_predictions(
    dataset: pd.DataFrame,
    fitter: ModelFitter,
    *,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Build validation forecasts from daily expanding ticker histories."""
    validate_model_dataset(dataset)
    require_finite_numeric(dataset[RETURN_COLUMN], RETURN_COLUMN)

    validation = dataset.loc[dataset["split"].eq(VALIDATION_SPLIT)].copy()
    if validation.empty:
        raise ValueError("dataset must contain validation rows")
    validation = validation.sort_values(["ticker", "target_date", "date"])

    expected_model_name: str | None = None
    rows: list[dict[str, object]] = []
    for position in range(len(validation)):
        observation = validation.iloc[[position]].copy()
        ticker = str(observation["ticker"].iloc[0])
        origin_date = pd.Timestamp(observation["date"].iloc[0])
        history = dataset.loc[
            dataset["ticker"].eq(ticker)
            & dataset["split"].ne("test")
            & dataset["date"].le(origin_date)
        ].sort_values("date")

        history_input = build_model_input(history)
        model_observation = build_model_input(observation)
        try:
            model = fitter(ticker, history_input)
            model_name = validate_model_name(model.name)
            if expected_model_name is None:
                expected_model_name = model_name
            elif model_name != expected_model_name:
                raise ValueError("model name must not change between validation dates")
            predictions = validate_prediction_series(
                model.predict(model_observation),
                model_observation,
            )
            if validate_model_name(model.name) != model_name:
                raise ValueError("model name must not change during prediction")
        except Exception as error:
            context = f"ticker {ticker!r} at origin {origin_date.date()}"
            raise ValueError(f"walk-forward failed for {context}: {error}") from error

        rows.append(
            {
                "date": observation["date"].iloc[0],
                "target_date": observation["target_date"].iloc[0],
                "ticker": ticker,
                "split": VALIDATION_SPLIT,
                "model": model_name,
                "actual_return": observation[TARGET_COLUMN].iloc[0],
                PREDICTION_NAME: predictions.iloc[0],
            }
        )
        if progress is not None:
            progress(position + 1, len(validation))

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS).reset_index(drop=True)
