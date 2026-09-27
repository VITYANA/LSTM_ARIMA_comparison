"""Build leakage-safe datasets of realized ARIMA residuals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from evaluation.contracts import (
    EVALUATION_SPLITS,
    PREDICTION_COLUMNS,
    require_finite_numeric,
    validate_model_dataset,
)
from evaluation.loading import REQUIRED_DATASET_COLUMNS
from models.arima import ARIMAOrder

ARIMA_RESIDUAL_WARMUP = 252
FIXED_ARIMA_ORDERS: Mapping[str, ARIMAOrder] = MappingProxyType(
    {
        "AAPL": (0, 0, 0),
        "JPM": (0, 0, 1),
        "SPY": (0, 0, 2),
        "XOM": (2, 0, 1),
    }
)
SOURCE_KEY = ("date", "target_date", "ticker", "split")


@dataclass(frozen=True)
class ARIMAResidualData:
    """Contain validated ARIMA forecasts and aligned residual rows."""

    arima_predictions: pd.DataFrame
    residual_dataset: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    """Require every column used by an input table."""
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _require_datetime_keys(frame: pd.DataFrame, label: str) -> None:
    """Require native datetime values for both temporal keys."""
    for column in ("date", "target_date"):
        if not pd.api.types.is_datetime64_any_dtype(frame[column]):
            raise ValueError(f"{label} {column} must be datetime")


def _key_set(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    """Return source identity keys without index dependence."""
    return set(frame.loc[:, SOURCE_KEY].itertuples(index=False, name=None))


def _expected_source_tail(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return complete source tails beginning at first forecasts."""
    source = dataset.loc[
        dataset["split"].isin(EVALUATION_SPLITS) & dataset["ticker"].isin(FIXED_ARIMA_ORDERS)
    ].copy()
    if set(source["ticker"]) - set(predictions["ticker"]):
        raise ValueError("missing prediction keys for source tickers")

    expected_frames: list[pd.DataFrame] = []
    for ticker, ticker_predictions in predictions.groupby("ticker", sort=True):
        ticker_source = source.loc[source["ticker"].eq(ticker)].sort_values(["target_date", "date"])
        first_key = next(
            iter(
                ticker_predictions.sort_values(["target_date", "date"])
                .loc[:, SOURCE_KEY]
                .itertuples(index=False, name=None)
            )
        )
        source_keys = list(ticker_source.loc[:, SOURCE_KEY].itertuples(index=False, name=None))
        expected_frames.append(ticker_source.iloc[source_keys.index(first_key) :])
    return pd.concat(expected_frames, ignore_index=True)


def _validated_inputs(
    dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and join forecasts to their source observations."""
    validate_model_dataset(dataset)
    _require_columns(dataset, REQUIRED_DATASET_COLUMNS, "dataset")
    _require_columns(arima_predictions, PREDICTION_COLUMNS, "predictions")
    _require_datetime_keys(dataset, "dataset")
    _require_datetime_keys(arima_predictions, "predictions")

    predictions = arima_predictions.loc[:, PREDICTION_COLUMNS].copy(deep=True)
    if predictions["split"].eq("test").any():
        raise ValueError("predictions must not contain test split rows")
    if not predictions["split"].isin(EVALUATION_SPLITS).all():
        raise ValueError("predictions contain unknown split values")
    if not predictions["model"].eq("arima").all():
        raise ValueError("predictions model must be 'arima'")
    if not predictions["ticker"].isin(FIXED_ARIMA_ORDERS).all():
        raise ValueError("predictions contain unknown ticker values")
    if predictions.duplicated(list(SOURCE_KEY)).any():
        raise ValueError("predictions contain duplicate observation keys")
    require_finite_numeric(predictions["actual_return"], "actual_return")
    require_finite_numeric(predictions["predicted_return"], "predicted_return")

    test_target_dates = set(dataset.loc[dataset["split"].eq("test"), "target_date"])
    if predictions["target_date"].isin(test_target_dates).any():
        raise ValueError("predictions contain a test target date")

    source = dataset.loc[
        dataset["split"].isin(EVALUATION_SPLITS) & dataset["ticker"].isin(FIXED_ARIMA_ORDERS)
    ]
    prediction_keys = _key_set(predictions)
    source_keys = _key_set(source)
    if prediction_keys - source_keys:
        raise ValueError("extra prediction keys not present in source data")
    expected_keys = _key_set(_expected_source_tail(dataset, predictions))
    if expected_keys - prediction_keys:
        raise ValueError("missing prediction keys from source data")

    source_values = source.loc[
        :,
        [*SOURCE_KEY, "adj_close", "target_return"],
    ].rename(
        columns={
            "adj_close": "source_adj_close",
            "target_return": "source_target_return",
        }
    )
    merged = predictions.merge(
        source_values,
        on=list(SOURCE_KEY),
        how="left",
        validate="one_to_one",
    )
    if not np.allclose(
        merged["actual_return"].to_numpy(dtype="float64"),
        merged["source_target_return"].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("prediction actual_return does not match source target_return")
    require_finite_numeric(merged["source_adj_close"], "adj_close")
    return merged.sort_values(["ticker", "target_date", "date"]).reset_index(drop=True)


def build_residual_dataset(
    dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build consecutive realized residual features and targets."""
    predictions = _validated_inputs(dataset, arima_predictions)
    rows: list[dict[str, object]] = []
    for ticker, ticker_predictions in predictions.groupby("ticker", sort=True):
        ordered = ticker_predictions.reset_index(drop=True)
        for position in range(len(ordered) - 1):
            current = ordered.iloc[position]
            following = ordered.iloc[position + 1]
            if current["target_date"] != following["date"]:
                current_target = pd.Timestamp(current["target_date"]).date()
                following_origin = pd.Timestamp(following["date"]).date()
                raise ValueError(
                    f"residual chronology gap for ticker {ticker!r}: "
                    f"current target {current_target} != next origin {following_origin}"
                )
            rows.append(
                {
                    "date": current["target_date"],
                    "target_date": following["target_date"],
                    "ticker": ticker,
                    "adj_close": following["source_adj_close"],
                    "log_return": current["actual_return"] - current["predicted_return"],
                    "target_return": following["actual_return"] - following["predicted_return"],
                    "split": following["split"],
                }
            )

    result = pd.DataFrame(rows, columns=REQUIRED_DATASET_COLUMNS)
    validate_model_dataset(result)
    require_finite_numeric(result["adj_close"], "adj_close")
    require_finite_numeric(result["log_return"], "log_return")
    return result.sort_values(["ticker", "target_date", "date"]).reset_index(drop=True)


def validate_arima_residual_data(
    dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> ARIMAResidualData:
    """Return isolated validated forecasts and residual dataset."""
    residual_dataset = build_residual_dataset(dataset, arima_predictions)
    predictions = (
        arima_predictions.loc[:, PREDICTION_COLUMNS]
        .sort_values(["ticker", "target_date", "date"])
        .reset_index(drop=True)
        .copy(deep=True)
    )
    return ARIMAResidualData(
        arima_predictions=predictions,
        residual_dataset=residual_dataset,
    )
