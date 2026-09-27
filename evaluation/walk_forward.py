"""Run leakage-safe expanding-window validation forecasts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from evaluation.contracts import (
    EVALUATION_SPLITS,
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


def _validate_target_splits(target_splits: object) -> tuple[str, ...]:
    """Return unique requested train and validation split names."""
    if isinstance(target_splits, (str, bytes)) or not isinstance(target_splits, Sequence):
        raise ValueError("target_splits must be a non-empty sequence")
    values = tuple(target_splits)
    if not values or any(not isinstance(value, str) for value in values):
        raise ValueError("target_splits must contain split names")
    if len(set(values)) != len(values):
        raise ValueError("target_splits must not contain duplicates")
    if not set(values).issubset(EVALUATION_SPLITS):
        raise ValueError("target_splits must contain only train and validation")
    return values


def _validate_min_history(min_history: object) -> int:
    """Return a positive minimum history length."""
    if type(min_history) is not int or min_history <= 0:
        raise ValueError("min_history must be a positive integer")
    return min_history


def _eligible_observations(
    dataset: pd.DataFrame,
    target_splits: tuple[str, ...],
    min_history: int,
) -> pd.DataFrame:
    """Return requested observations meeting the ticker history boundary."""
    observations = dataset.loc[dataset["split"].isin(target_splits)].copy()
    observations = observations.sort_values(["ticker", "target_date", "date"])
    history_dates_by_ticker = {
        ticker: pd.DatetimeIndex(
            dataset.loc[
                dataset["ticker"].eq(ticker) & dataset["split"].isin(EVALUATION_SPLITS),
                "date",
            ].sort_values()
        )
        for ticker in observations["ticker"].drop_duplicates()
    }
    eligible_positions: list[int] = []
    for position in range(len(observations)):
        ticker = observations["ticker"].iloc[position]
        origin_date = pd.Timestamp(observations["date"].iloc[position])
        available = int(history_dates_by_ticker[ticker].searchsorted(origin_date, side="right"))
        if available >= min_history:
            eligible_positions.append(position)
    return observations.iloc[eligible_positions].copy()


def build_walk_forward_predictions(
    dataset: pd.DataFrame,
    fitter: ModelFitter,
    *,
    target_splits: Sequence[str] = (VALIDATION_SPLIT,),
    min_history: int = 1,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Build requested forecasts from daily expanding ticker histories."""
    validate_model_dataset(dataset)
    require_finite_numeric(dataset[RETURN_COLUMN], RETURN_COLUMN)
    validated_splits = _validate_target_splits(target_splits)
    validated_min_history = _validate_min_history(min_history)
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")

    observations = _eligible_observations(dataset, validated_splits, validated_min_history)
    if observations.empty:
        raise ValueError(f"dataset has no eligible rows for target_splits {validated_splits}")

    expected_model_name: str | None = None
    rows: list[dict[str, object]] = []
    for position in range(len(observations)):
        observation = observations.iloc[[position]].copy()
        ticker = str(observation["ticker"].iloc[0])
        origin_date = pd.Timestamp(observation["date"].iloc[0])
        history = dataset.loc[
            dataset["ticker"].eq(ticker)
            & dataset["split"].isin(EVALUATION_SPLITS)
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
                "split": observation["split"].iloc[0],
                "model": model_name,
                "actual_return": observation[TARGET_COLUMN].iloc[0],
                PREDICTION_NAME: predictions.iloc[0],
            }
        )
        if progress is not None:
            progress(position + 1, len(observations))

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS).reset_index(drop=True)
