"""Run one leakage-safe LSTM candidate on validation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from data.sequences import (
    build_return_sequences,
    fit_return_scaler,
    lag_columns,
    split_sequence_batch,
    transform_features,
    transform_targets,
)
from evaluation.contracts import (
    BPS_FACTOR,
    EVALUATION_SPLITS,
    PREDICTION_COLUMNS,
    require_finite_numeric,
    validate_model_dataset,
    validate_prediction_series,
)
from models.lstm import MAX_SEED, MIN_SEED, LSTMConfig, LSTMTrainer

CORE_TICKERS = ("AAPL", "JPM", "SPY", "XOM")
RunStatus = Literal["ok", "failed", "non_finite"]


@dataclass(frozen=True)
class LSTMRunKey:
    """Identify one ticker, configuration and seed run."""

    ticker: str
    config: LSTMConfig
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or self.ticker not in CORE_TICKERS:
            raise ValueError(f"ticker must be one of {CORE_TICKERS}")
        if not isinstance(self.config, LSTMConfig):
            raise ValueError("config must be an LSTMConfig")
        if type(self.seed) is not int or not MIN_SEED <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be an integer from {MIN_SEED} through {MAX_SEED}")

    @property
    def run_id(self) -> str:
        """Return a stable checkpoint-safe run identifier."""
        return f"{self.ticker}_{_model_name(self)}"


@dataclass(frozen=True)
class LSTMRunResult:
    """Contain one terminal LSTM run and its validation output."""

    key: LSTMRunKey
    status: RunStatus
    best_epoch: int | None
    parameter_count: int | None
    mae_bps: float
    rmse_bps: float
    predictions: pd.DataFrame
    error: str | None


class _NonFinitePredictionError(ValueError):
    """Mark a completed model that emitted nonfinite forecasts."""


def _model_name(key: LSTMRunKey) -> str:
    """Return the stable evaluation model name for one run."""
    dropout_percent = int(round(key.config.dropout * 100))
    return (
        f"lstm_w{key.config.window}_u{key.config.units}_d{dropout_percent:02d}"
        f"_seed_{key.seed:02d}"
    )


def _empty_predictions() -> pd.DataFrame:
    """Return an empty table following the prediction contract."""
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _terminal_error(
    key: LSTMRunKey,
    error: Exception,
    best_epoch: int | None,
    parameter_count: int | None,
) -> LSTMRunResult:
    """Convert an independent runtime error into one terminal result."""
    non_finite = isinstance(error, _NonFinitePredictionError) or (
        isinstance(error, ValueError) and "finite" in str(error).lower()
    )
    status: RunStatus = "non_finite" if non_finite else "failed"
    return LSTMRunResult(
        key=key,
        status=status,
        best_epoch=best_epoch,
        parameter_count=parameter_count,
        mae_bps=float("nan"),
        rmse_bps=float("nan"),
        predictions=_empty_predictions(),
        error=f"{key.run_id}: {type(error).__name__}: {error}",
    )


def _build_prediction_table(
    keys: pd.DataFrame,
    targets: pd.Series,
    predicted: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Build one aligned standard validation prediction table."""
    predictions = keys.copy()
    predictions["model"] = model_name
    predictions["actual_return"] = targets
    predictions["predicted_return"] = predicted
    return predictions.loc[:, PREDICTION_COLUMNS]


def run_lstm_candidate(
    dataset: pd.DataFrame,
    key: LSTMRunKey,
    trainer: LSTMTrainer,
) -> LSTMRunResult:
    """Train and validate one ticker, configuration and seed candidate."""
    if not isinstance(key, LSTMRunKey):
        raise ValueError("key must be an LSTMRunKey")
    validate_model_dataset(dataset)
    evaluation_data = dataset.loc[dataset["split"].isin(EVALUATION_SPLITS)].copy()

    train_batch = build_return_sequences(
        evaluation_data,
        key.ticker,
        key.config.window,
        "train",
    )
    validation_batch = build_return_sequences(
        evaluation_data,
        key.ticker,
        key.config.window,
        "validation",
    )
    inner_train, inner_validation = split_sequence_batch(train_batch)
    inner_cutoff = inner_train.keys["date"].max()
    inner_scaler = fit_return_scaler(evaluation_data, key.ticker, inner_cutoff)
    inner_train_features = transform_features(inner_train.features, inner_scaler)
    inner_train_targets = transform_targets(inner_train.targets, inner_scaler)
    inner_validation_features = transform_features(
        inner_validation.features,
        inner_scaler,
    )
    inner_validation_targets = transform_targets(
        inner_validation.targets,
        inner_scaler,
    )

    best_epoch: int | None = None
    parameter_count: int | None = None
    try:
        best_epoch = trainer.determine_best_epoch(
            key.config,
            key.seed,
            inner_train_features,
            inner_train_targets,
            inner_validation_features,
            inner_validation_targets,
        )
        full_cutoff = train_batch.keys["date"].max()
        full_scaler = fit_return_scaler(evaluation_data, key.ticker, full_cutoff)
        full_features = transform_features(train_batch.features, full_scaler)
        full_targets = transform_targets(train_batch.targets, full_scaler)
        model_name = _model_name(key)
        model = trainer.fit(
            key.config,
            key.seed,
            full_features,
            full_targets,
            full_scaler,
            lag_columns(key.config.window),
            best_epoch,
            model_name,
        )
        parameter_count = model.parameter_count
        predicted = model.predict(validation_batch.features)
        if (
            isinstance(predicted, pd.Series)
            and predicted.dtype.kind in "iufc"
            and not np.isfinite(predicted.to_numpy()).all()
        ):
            raise _NonFinitePredictionError("predictions must contain only finite values")
        validated = validate_prediction_series(predicted, validation_batch.features)
        predictions = _build_prediction_table(
            validation_batch.keys,
            validation_batch.targets,
            validated,
            model_name,
        )
        require_finite_numeric(predictions["actual_return"], "actual_return")
        require_finite_numeric(predictions["predicted_return"], "predicted_return")
        errors = predictions["actual_return"].to_numpy(dtype="float64") - predictions[
            "predicted_return"
        ].to_numpy(dtype="float64")
        mae_bps = float(np.mean(np.abs(errors))) * BPS_FACTOR
        rmse_bps = float(np.sqrt(np.mean(np.square(errors)))) * BPS_FACTOR
    except Exception as error:
        return _terminal_error(key, error, best_epoch, parameter_count)

    return LSTMRunResult(
        key=key,
        status="ok",
        best_epoch=best_epoch,
        parameter_count=parameter_count,
        mae_bps=mae_bps,
        rmse_bps=rmse_bps,
        predictions=predictions,
        error=None,
    )
