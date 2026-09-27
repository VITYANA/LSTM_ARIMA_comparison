"""Frozen model choices and contracts for independent final evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.preparation import DEFAULT_SPLIT_CONFIG, SplitConfig
from evaluation.contracts import OBSERVATION_KEY
from evaluation.loading import REQUIRED_DATASET_COLUMNS
from models.arima import ARIMAOrder
from models.lstm import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    LSTMConfig,
)

ARIMA_WARMUP = 252
ANNUALIZATION_DAYS = 252
VOLATILITY_WINDOW = 21
MODEL_COMPARISONS = (
    ("arima", "naive_zero"),
    ("lstm", "naive_zero"),
    ("arima_lstm", "naive_zero"),
    ("arima_lstm", "arima"),
    ("arima_lstm", "lstm"),
)
NUMERIC_DATASET_COLUMNS = ("adj_close", "log_return", "target_return")


@dataclass(frozen=True)
class FinalTickerSpec:
    """Store the selected models for one final-test ticker."""

    ticker: str
    arima_order: ARIMAOrder
    lstm_config: LSTMConfig
    residual_config: LSTMConfig

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ValueError("ticker must be a nonempty string")
        if (
            not isinstance(self.arima_order, tuple)
            or len(self.arima_order) != 3
            or any(type(value) is not int or value < 0 for value in self.arima_order)
        ):
            raise ValueError("arima_order must contain three nonnegative integers")
        if not isinstance(self.lstm_config, LSTMConfig):
            raise ValueError("lstm_config must be an LSTMConfig")
        if not isinstance(self.residual_config, LSTMConfig):
            raise ValueError("residual_config must be an LSTMConfig")


FINAL_TICKER_SPECS = (
    FinalTickerSpec("AAPL", (0, 0, 0), LSTMConfig(5, 32, 0.0), LSTMConfig(5, 32, 0.0)),
    FinalTickerSpec("JPM", (0, 0, 1), LSTMConfig(63, 16, 0.2), LSTMConfig(63, 32, 0.2)),
    FinalTickerSpec("SPY", (0, 0, 2), LSTMConfig(21, 32, 0.2), LSTMConfig(21, 32, 0.0)),
    FinalTickerSpec("XOM", (2, 0, 1), LSTMConfig(5, 16, 0.2), LSTMConfig(5, 32, 0.2)),
)


@dataclass(frozen=True)
class FinalTrainingSpec:
    """Store the fixed two-stage Keras training procedure."""

    inner_train_fraction: float
    max_epochs: int
    batch_size: int
    early_stopping_patience: int
    min_delta: float
    learning_rate: float
    shuffle: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.inner_train_fraction, bool)
            or not isinstance(self.inner_train_fraction, (int, float))
            or not 0.0 < self.inner_train_fraction < 1.0
        ):
            raise ValueError("inner_train_fraction must be between zero and one")
        expected = (
            MAX_EPOCHS,
            BATCH_SIZE,
            EARLY_STOPPING_PATIENCE,
            0.0,
            LEARNING_RATE,
            False,
        )
        actual = (
            self.max_epochs,
            self.batch_size,
            self.early_stopping_patience,
            self.min_delta,
            self.learning_rate,
            self.shuffle,
        )
        if actual != expected:
            raise ValueError("training parameters must match the fixed LSTM procedure")
        object.__setattr__(self, "inner_train_fraction", float(self.inner_train_fraction))
        object.__setattr__(self, "min_delta", float(self.min_delta))
        object.__setattr__(self, "learning_rate", float(self.learning_rate))


FINAL_TRAINING_SPEC = FinalTrainingSpec(
    inner_train_fraction=0.8,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    min_delta=0.0,
    learning_rate=LEARNING_RATE,
    shuffle=False,
)


def _require_positive_integer(value: object, label: str) -> None:
    """Require a strictly positive integer protocol value."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _require_probability(value: object, label: str) -> None:
    """Require a numeric probability strictly between zero and one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value < 1.0:
        raise ValueError(f"{label} must be between zero and one")


@dataclass(frozen=True)
class FinalTestProtocol:
    """Store every choice frozen before independent test evaluation."""

    ticker_specs: tuple[FinalTickerSpec, ...]
    split_config: SplitConfig
    seeds: tuple[int, ...]
    training: FinalTrainingSpec
    arima_warmup: int
    volatility_window: int
    annualization_days: int
    model_comparisons: tuple[tuple[str, str], ...]
    dm_lag: int
    holm_alpha: float
    bootstrap_block: int
    bootstrap_repetitions: int
    bootstrap_seed: int
    bootstrap_confidence: float

    def __post_init__(self) -> None:
        if self.ticker_specs != FINAL_TICKER_SPECS:
            raise ValueError("ticker_specs must match the frozen ticker specifications")
        if self.split_config != DEFAULT_SPLIT_CONFIG:
            raise ValueError("split_config must match the canonical research boundaries")
        if self.seeds != tuple(range(10)):
            raise ValueError("seeds must contain ordered values from 0 through 9")
        if not isinstance(self.training, FinalTrainingSpec):
            raise ValueError("training must be a FinalTrainingSpec")
        _require_positive_integer(self.arima_warmup, "arima_warmup")
        _require_positive_integer(self.volatility_window, "volatility_window")
        _require_positive_integer(self.annualization_days, "annualization_days")
        if self.model_comparisons != MODEL_COMPARISONS:
            raise ValueError("model_comparisons must match the frozen comparisons")
        if type(self.dm_lag) is not int or self.dm_lag < 0:
            raise ValueError("dm_lag must be a nonnegative integer")
        _require_probability(self.holm_alpha, "holm_alpha")
        _require_positive_integer(self.bootstrap_block, "bootstrap_block")
        if self.dm_lag != self.bootstrap_block - 1:
            raise ValueError("dm_lag must be one less than bootstrap_block")
        _require_positive_integer(self.bootstrap_repetitions, "bootstrap_repetitions")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be a nonnegative integer")
        _require_probability(self.bootstrap_confidence, "bootstrap_confidence")
        object.__setattr__(self, "holm_alpha", float(self.holm_alpha))
        object.__setattr__(self, "bootstrap_confidence", float(self.bootstrap_confidence))


FINAL_TEST_PROTOCOL = FinalTestProtocol(
    ticker_specs=FINAL_TICKER_SPECS,
    split_config=DEFAULT_SPLIT_CONFIG,
    seeds=tuple(range(10)),
    training=FINAL_TRAINING_SPEC,
    arima_warmup=ARIMA_WARMUP,
    volatility_window=VOLATILITY_WINDOW,
    annualization_days=ANNUALIZATION_DAYS,
    model_comparisons=MODEL_COMPARISONS,
    dm_lag=20,
    holm_alpha=0.05,
    bootstrap_block=21,
    bootstrap_repetitions=10_000,
    bootstrap_seed=42,
    bootstrap_confidence=0.95,
)


def _parse_date_column(dataset: pd.DataFrame, column: str) -> None:
    """Parse one required date column in place."""
    try:
        parsed = pd.to_datetime(dataset[column], format="mixed", errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must contain valid dates") from error
    if parsed.isna().any():
        raise ValueError(f"{column} must contain valid dates")
    dataset[column] = parsed


def _validate_numeric_columns(dataset: pd.DataFrame) -> None:
    """Require finite numeric model values."""
    for column in NUMERIC_DATASET_COLUMNS:
        series = dataset[column]
        if series.dtype.kind not in "iufc":
            raise ValueError(f"{column} must be numeric")
        if not np.isfinite(series.to_numpy()).all():
            raise ValueError(f"{column} must contain only finite values")


def _expected_splits(target_dates: pd.Series, config: SplitConfig) -> np.ndarray:
    """Return configured split labels for target dates."""
    train = target_dates.between(config.train_start, config.train_end)
    validation = target_dates.between(config.validation_start, config.validation_end)
    test = target_dates.between(config.test_start, config.test_end)
    return np.select(
        [train.to_numpy(), validation.to_numpy(), test.to_numpy()],
        ["train", "validation", "test"],
        default="",
    )


def validate_final_dataset(
    dataset: pd.DataFrame,
    protocol: FinalTestProtocol = FINAL_TEST_PROTOCOL,
) -> pd.DataFrame:
    """Return an isolated canonical snapshot for final-test processing."""
    if not isinstance(protocol, FinalTestProtocol):
        raise ValueError("protocol must be a FinalTestProtocol")
    missing_columns = set(REQUIRED_DATASET_COLUMNS) - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"dataset missing columns: {sorted(missing_columns)}")
    if dataset.empty:
        raise ValueError("dataset must not be empty")

    validated = dataset.copy(deep=True)
    _parse_date_column(validated, "date")
    _parse_date_column(validated, "target_date")
    _validate_numeric_columns(validated)

    expected_tickers = {spec.ticker for spec in protocol.ticker_specs}
    if set(validated["ticker"]) != expected_tickers:
        raise ValueError("dataset tickers must match the final-test protocol")
    known_splits = {"train", "validation", "test"}
    actual_splits = set(validated["split"])
    if not actual_splits <= known_splits:
        raise ValueError("dataset contains unknown split values")
    if actual_splits != known_splits:
        raise ValueError("dataset splits must contain train, validation and test")
    if validated.duplicated(list(OBSERVATION_KEY)).any():
        raise ValueError("dataset contains duplicate observation keys")
    if not validated["date"].lt(validated["target_date"]).all():
        raise ValueError("date must precede target_date")

    expected_splits = _expected_splits(validated["target_date"], protocol.split_config)
    if not np.array_equal(expected_splits, validated["split"].to_numpy()):
        raise ValueError("dataset split values must match canonical split boundaries")

    observed_pairs = set(zip(validated["ticker"], validated["split"], strict=True))
    expected_pairs = {(ticker, split) for ticker in expected_tickers for split in known_splits}
    if observed_pairs != expected_pairs:
        raise ValueError("dataset must contain every required ticker/split pair")

    test_dates = validated.loc[validated["split"].eq("test"), "target_date"]
    if test_dates.min() != pd.Timestamp(
        protocol.split_config.test_start
    ) or test_dates.max() != pd.Timestamp(protocol.split_config.test_end):
        raise ValueError("dataset test interval must include both canonical boundaries")

    return validated.sort_values(["ticker", "target_date"], kind="mergesort").reset_index(drop=True)
