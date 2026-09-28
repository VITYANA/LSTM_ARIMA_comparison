"""Frozen model choices and contracts for independent final evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.preparation import DEFAULT_SPLIT_CONFIG, SplitConfig
from data.sequences import (
    SequenceBatch,
    fit_return_scaler,
    lag_columns,
    split_sequence_batch,
    transform_features,
    transform_targets,
)
from evaluation.contracts import (
    OBSERVATION_KEY,
    PREDICTION_COLUMNS,
    require_finite_numeric,
    validate_prediction_series,
)
from evaluation.loading import REQUIRED_DATASET_COLUMNS
from models.arima import ARIMAOrder
from models.interfaces import validate_model_name
from models.lstm import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_SEED,
    MIN_SEED,
    LSTMConfig,
    LSTMTrainer,
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
FINAL_KEY_COLUMNS = ("date", "target_date", "ticker", "split")
FINAL_SORT_COLUMNS = ("ticker", "target_date", "date", "split")


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


@dataclass(frozen=True)
class FinalSeedResult:
    """Contain one successful fixed-model final seed run."""

    ticker: str
    config: LSTMConfig
    seed: int
    model: str
    best_epoch: int
    parameter_count: int
    predictions: pd.DataFrame


def _selected_ticker_spec(ticker: object) -> FinalTickerSpec:
    """Return the frozen specification for one core ticker."""
    for spec in FINAL_TEST_PROTOCOL.ticker_specs:
        if spec.ticker == ticker:
            return spec
    raise ValueError("ticker must belong to the final-test protocol")


def _validated_final_identity(
    ticker: object,
    config: object,
    seed: object,
    model_name: object,
) -> tuple[str, LSTMConfig, int, str]:
    """Validate one final LSTM run identity."""
    spec = _selected_ticker_spec(ticker)
    if not isinstance(config, LSTMConfig) or config not in (
        spec.lstm_config,
        spec.residual_config,
    ):
        raise ValueError("config must match a frozen ticker configuration")
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer from {MIN_SEED} through {MAX_SEED}")
    return spec.ticker, config, seed, validate_model_name(model_name)


def _build_final_sequences(
    dataset: pd.DataFrame,
    ticker: str,
    window: int,
    splits: frozenset[str],
) -> SequenceBatch:
    """Build complete chronological sequences for selected final splits."""
    rows = dataset.loc[dataset["ticker"].eq(ticker)].sort_values(
        ["date", "target_date"],
        kind="mergesort",
    )
    columns = lag_columns(window)
    features = pd.DataFrame(
        {
            column: rows["log_return"].shift(lag)
            for column, lag in zip(columns, range(window - 1, -1, -1), strict=True)
        },
        index=rows.index,
    )
    selected = rows["split"].isin(splits) & features.notna().all(axis=1)
    if not selected.any():
        raise ValueError("insufficient history for final LSTM sequences")
    targets = rows.loc[selected, "target_return"].astype("float64").copy()
    targets.name = "target_return"
    return SequenceBatch(
        features=features.loc[selected].astype("float64"),
        targets=targets,
        keys=rows.loc[selected, FINAL_KEY_COLUMNS].copy(),
    )


def _seed_model_name(model_name: str, seed: int) -> str:
    """Return a stable final seed model identifier."""
    return f"{model_name}_seed_{seed:02d}"


def run_final_lstm_seed(
    dataset: pd.DataFrame,
    ticker: str,
    config: LSTMConfig,
    seed: int,
    trainer: LSTMTrainer,
    model_name: str,
) -> FinalSeedResult:
    """Train one fixed LSTM seed and forecast the complete final test."""
    validated_ticker, validated_config, validated_seed, base_name = _validated_final_identity(
        ticker,
        config,
        seed,
        model_name,
    )
    validated = validate_final_dataset(dataset)
    training_splits = frozenset({"train", "validation"})
    training_batch = _build_final_sequences(
        validated,
        validated_ticker,
        validated_config.window,
        training_splits,
    )
    test_batch = _build_final_sequences(
        validated,
        validated_ticker,
        validated_config.window,
        frozenset({"test"}),
    )
    inner_train, inner_validation = split_sequence_batch(
        training_batch,
        FINAL_TEST_PROTOCOL.training.inner_train_fraction,
    )
    training_data = validated.loc[validated["split"].isin(training_splits)].copy()

    inner_cutoff = pd.Timestamp(inner_train.keys["date"].max())
    inner_scaler = fit_return_scaler(training_data, validated_ticker, inner_cutoff)
    best_epoch = trainer.determine_best_epoch(
        validated_config,
        validated_seed,
        transform_features(inner_train.features, inner_scaler),
        transform_targets(inner_train.targets, inner_scaler),
        transform_features(inner_validation.features, inner_scaler),
        transform_targets(inner_validation.targets, inner_scaler),
    )
    if type(best_epoch) is not int or not 1 <= best_epoch <= MAX_EPOCHS:
        raise ValueError(f"best epoch must be an integer from 1 through {MAX_EPOCHS}")

    full_cutoff = pd.Timestamp(training_batch.keys["date"].max())
    full_scaler = fit_return_scaler(training_data, validated_ticker, full_cutoff)
    seed_name = _seed_model_name(base_name, validated_seed)
    model = trainer.fit(
        validated_config,
        validated_seed,
        transform_features(training_batch.features, full_scaler),
        transform_targets(training_batch.targets, full_scaler),
        full_scaler,
        lag_columns(validated_config.window),
        best_epoch,
        seed_name,
    )
    if model.name != seed_name or model.config != validated_config:
        raise ValueError("fitted model identity must match the requested final seed")
    if type(model.parameter_count) is not int or model.parameter_count <= 0:
        raise ValueError("parameter_count must be a positive integer")

    predicted = validate_prediction_series(model.predict(test_batch.features), test_batch.features)
    predictions = test_batch.keys.copy()
    predictions["model"] = seed_name
    predictions["actual_return"] = test_batch.targets
    predictions["predicted_return"] = predicted
    predictions = predictions.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)
    require_finite_numeric(predictions["actual_return"], "actual_return")
    require_finite_numeric(predictions["predicted_return"], "predicted_return")
    return FinalSeedResult(
        ticker=validated_ticker,
        config=validated_config,
        seed=validated_seed,
        model=seed_name,
        best_epoch=best_epoch,
        parameter_count=model.parameter_count,
        predictions=predictions,
    )


def _validated_seed_predictions(
    result: FinalSeedResult,
    base_name: str,
) -> pd.DataFrame:
    """Return one isolated canonical seed prediction table."""
    expected_name = _seed_model_name(base_name, result.seed)
    if result.model != expected_name:
        raise ValueError("seed result model name does not match its seed")
    predictions = result.predictions
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError("seed predictions must use standard columns in standard order")
    if predictions.empty:
        raise ValueError("seed predictions must not be empty")
    for column in ("date", "target_date"):
        if not pd.api.types.is_datetime64_any_dtype(predictions[column]):
            raise ValueError(f"seed prediction {column} must be datetime")
        if predictions[column].isna().any():
            raise ValueError("seed predictions must contain finite dates")
    if not predictions["ticker"].eq(result.ticker).all():
        raise ValueError("seed prediction ticker must match its result")
    if not predictions["split"].eq("test").all():
        raise ValueError("seed predictions must contain test rows only")
    if not predictions["model"].eq(result.model).all():
        raise ValueError("seed prediction model must match its result")
    if predictions.duplicated(list(FINAL_KEY_COLUMNS)).any():
        raise ValueError("seed predictions contain duplicate observation keys")
    require_finite_numeric(predictions["actual_return"], "seed actual_return")
    require_finite_numeric(predictions["predicted_return"], "seed predicted_return")
    return (
        predictions.loc[:, PREDICTION_COLUMNS]
        .sort_values(list(FINAL_SORT_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )


def average_final_seed_predictions(
    results: Sequence[FinalSeedResult],
    model_name: str,
) -> pd.DataFrame:
    """Average exactly ten aligned final seed forecasts."""
    base_name = validate_model_name(model_name)
    values = tuple(results)
    if not all(isinstance(result, FinalSeedResult) for result in values):
        raise ValueError("results must contain FinalSeedResult values")
    if len(values) != len(FINAL_TEST_PROTOCOL.seeds) or {result.seed for result in values} != set(
        FINAL_TEST_PROTOCOL.seeds
    ):
        raise ValueError("results must contain exactly seeds 0 through 9")
    if len({result.ticker for result in values}) != 1:
        raise ValueError("seed results must share one ticker")
    if len({result.config for result in values}) != 1:
        raise ValueError("seed results must share one config")

    ordered = tuple(sorted(values, key=lambda result: result.seed))
    tables = tuple(_validated_seed_predictions(result, base_name) for result in ordered)
    expected_keys = tables[0].loc[:, FINAL_KEY_COLUMNS]
    expected_actual = tables[0]["actual_return"]
    for table in tables[1:]:
        if not table.loc[:, FINAL_KEY_COLUMNS].equals(expected_keys):
            raise ValueError("seed prediction keys do not align")
        if not table["actual_return"].equals(expected_actual):
            raise ValueError("seed actual returns do not align")

    prediction_values = np.column_stack(
        [table["predicted_return"].to_numpy(dtype="float64") for table in tables]
    )
    averaged = tables[0].loc[:, FINAL_KEY_COLUMNS].copy()
    averaged["model"] = base_name
    averaged["actual_return"] = expected_actual.to_numpy(dtype="float64")
    averaged["predicted_return"] = prediction_values.mean(axis=1)
    return averaged.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)
