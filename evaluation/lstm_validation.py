"""Run one leakage-safe LSTM candidate on validation data."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
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
from models.lstm import (
    MAX_SEED,
    MIN_SEED,
    LSTMConfig,
    LSTMTrainer,
    build_lstm_grid,
)

CORE_TICKERS = ("AAPL", "JPM", "SPY", "XOM")
RunStatus = Literal["ok", "failed", "non_finite"]
ProgressCallback = Callable[[int, int, str], None]


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


def _validate_grid_configs(
    configs: Sequence[LSTMConfig] | None,
) -> tuple[LSTMConfig, ...]:
    """Return a nonempty sequence of unique configurations."""
    values = build_lstm_grid() if configs is None else tuple(configs)
    if not values:
        raise ValueError("configs must not be empty")
    if not all(isinstance(config, LSTMConfig) for config in values):
        raise ValueError("configs must contain LSTMConfig values")
    if len(set(values)) != len(values):
        raise ValueError("duplicate config values are not allowed")
    return values


def _validate_grid_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Return a nonempty sequence of unique supported seeds."""
    values = tuple(seeds)
    if not values:
        raise ValueError("seeds must not be empty")
    if any(type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED for seed in values):
        raise ValueError(f"seed must be an integer from {MIN_SEED} through {MAX_SEED}")
    if len(set(values)) != len(values):
        raise ValueError("duplicate seed values are not allowed")
    return values


def _grid_evaluation_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return core train and validation rows after structural checks."""
    validate_model_dataset(dataset)
    evaluation_data = dataset.loc[
        dataset["ticker"].isin(CORE_TICKERS) & dataset["split"].isin(EVALUATION_SPLITS)
    ].copy()
    if evaluation_data.empty:
        raise ValueError("dataset must contain train and validation rows")
    for ticker in CORE_TICKERS:
        ticker_splits = set(evaluation_data.loc[evaluation_data["ticker"].eq(ticker), "split"])
        if ticker_splits != set(EVALUATION_SPLITS):
            raise ValueError(f"ticker {ticker!r} must contain train and validation rows")
    return evaluation_data


def run_lstm_grid(
    dataset: pd.DataFrame,
    dataset_sha256: str,
    checkpoint_dir: Path,
    trainer: LSTMTrainer,
    configs: Sequence[LSTMConfig] | None = None,
    seeds: Sequence[int] = tuple(range(10)),
    progress: ProgressCallback | None = None,
) -> tuple[LSTMRunResult, ...]:
    """Resume and execute the deterministic core-ticker LSTM grid."""
    from evaluation import lstm_checkpoint

    config_values = _validate_grid_configs(configs)
    seed_values = _validate_grid_seeds(seeds)
    evaluation_data = _grid_evaluation_data(dataset)
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")

    fingerprint = lstm_checkpoint.build_lstm_fingerprint(
        dataset_sha256,
        config_values,
        seed_values,
    )
    scheduled_keys = tuple(
        LSTMRunKey(ticker, config, seed)
        for ticker in CORE_TICKERS
        for config in config_values
        for seed in seed_values
    )
    expected_ids = {key.run_id for key in scheduled_keys}
    cached = lstm_checkpoint.load_lstm_runs(checkpoint_dir, fingerprint)
    unexpected_ids = set(cached) - expected_ids
    if unexpected_ids:
        raise ValueError(f"checkpoint contains unexpected runs: {sorted(unexpected_ids)}")
    completed = len(cached)
    total = len(scheduled_keys)
    if progress is not None:
        progress(completed, total, "")

    results = dict(cached)
    for key in scheduled_keys:
        if key.run_id in results:
            continue
        result = run_lstm_candidate(evaluation_data, key, trainer)
        lstm_checkpoint.save_lstm_run(checkpoint_dir, fingerprint, result)
        results[key.run_id] = result
        completed += 1
        if progress is not None:
            progress(completed, total, key.run_id)
    return tuple(results[key.run_id] for key in scheduled_keys)
