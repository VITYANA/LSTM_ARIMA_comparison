"""Tests for the frozen independent final-test protocol."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

import evaluation.final_models as final_models_module
from analysis.preparation import DEFAULT_SPLIT_CONFIG
from data.sequences import fit_return_scaler, lag_columns, transform_features, transform_targets
from evaluation.contracts import PREDICTION_COLUMNS, PREDICTION_NAME
from evaluation.final_models import (
    FINAL_TEST_PROTOCOL,
    FinalSeedResult,
    FinalTestProtocol,
    FinalTickerSpec,
    FinalTrainingSpec,
    average_final_seed_predictions,
    run_final_lstm_seed,
    validate_final_dataset,
)
from models.lstm import LSTMConfig, LSTMTrainer

EXPECTED_COMPARISONS = (
    ("arima", "naive_zero"),
    ("lstm", "naive_zero"),
    ("arima_lstm", "naive_zero"),
    ("arima_lstm", "arima"),
    ("arima_lstm", "lstm"),
)
EXPECTED_TICKER_SPECS = (
    FinalTickerSpec("AAPL", (0, 0, 0), LSTMConfig(5, 32, 0.0), LSTMConfig(5, 32, 0.0)),
    FinalTickerSpec("JPM", (0, 0, 1), LSTMConfig(63, 16, 0.2), LSTMConfig(63, 32, 0.2)),
    FinalTickerSpec("SPY", (0, 0, 2), LSTMConfig(21, 32, 0.2), LSTMConfig(21, 32, 0.0)),
    FinalTickerSpec("XOM", (2, 0, 1), LSTMConfig(5, 16, 0.2), LSTMConfig(5, 32, 0.2)),
)
EXPECTED_TRAINING = FinalTrainingSpec(
    inner_train_fraction=0.8,
    max_epochs=200,
    batch_size=32,
    early_stopping_patience=15,
    min_delta=0.0,
    learning_rate=0.001,
    shuffle=False,
)


def valid_dataset() -> pd.DataFrame:
    """Return shuffled rows spanning every required split"""
    schedule = (
        ("2015-12-30", "2015-12-31", "train"),
        ("2019-12-30", "2019-12-31", "validation"),
        ("2019-12-31", "2020-01-02", "test"),
        ("2025-12-30", "2025-12-31", "test"),
    )
    rows = []
    for ticker_position, ticker in enumerate(("AAPL", "JPM", "SPY", "XOM"), start=1):
        for date, target_date, split in schedule:
            rows.append(
                {
                    "date": date,
                    "target_date": target_date,
                    "ticker": ticker,
                    "adj_close": 100.0 + ticker_position,
                    "log_return": ticker_position / 1_000,
                    "target_return": ticker_position / 2_000,
                    "split": split,
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=17).reset_index(drop=True)


def replace_protocol(field: str, value: object) -> FinalTestProtocol:
    """Replace one protocol field for negative tests"""
    return replace(FINAL_TEST_PROTOCOL, **cast(Any, {field: value}))


def test_default_protocol_matches_frozen_research_choices() -> None:
    """Expose every choice fixed before test evaluation"""
    assert FINAL_TEST_PROTOCOL.ticker_specs == EXPECTED_TICKER_SPECS
    assert FINAL_TEST_PROTOCOL.split_config == DEFAULT_SPLIT_CONFIG
    assert FINAL_TEST_PROTOCOL.seeds == tuple(range(10))
    assert FINAL_TEST_PROTOCOL.training == EXPECTED_TRAINING
    assert FINAL_TEST_PROTOCOL.arima_warmup == 252
    assert FINAL_TEST_PROTOCOL.volatility_window == 21
    assert FINAL_TEST_PROTOCOL.annualization_days == 252
    assert FINAL_TEST_PROTOCOL.model_comparisons == EXPECTED_COMPARISONS
    assert FINAL_TEST_PROTOCOL.dm_lag == 20
    assert FINAL_TEST_PROTOCOL.holm_alpha == pytest.approx(0.05)
    assert FINAL_TEST_PROTOCOL.bootstrap_block == 21
    assert FINAL_TEST_PROTOCOL.bootstrap_repetitions == 10_000
    assert FINAL_TEST_PROTOCOL.bootstrap_seed == 42
    assert FINAL_TEST_PROTOCOL.bootstrap_confidence == pytest.approx(0.95)


@pytest.mark.parametrize(
    ("value", "attribute"),
    [
        pytest.param(FINAL_TEST_PROTOCOL, "seeds", id="protocol"),
        pytest.param(EXPECTED_TICKER_SPECS[0], "ticker", id="ticker-spec"),
        pytest.param(EXPECTED_TRAINING, "max_epochs", id="training-spec"),
    ],
)
def test_protocol_objects_are_immutable(value: object, attribute: str) -> None:
    """Prevent post-freeze changes to research choices"""
    with pytest.raises(FrozenInstanceError):
        setattr(value, attribute, True)


@pytest.mark.parametrize(
    ("ticker", "order", "lstm_config", "residual_config"),
    [
        pytest.param("", (0, 0, 0), LSTMConfig(5, 16, 0.0), LSTMConfig(5, 16, 0.0), id="ticker"),
        pytest.param(
            "AAPL", (0, 0), LSTMConfig(5, 16, 0.0), LSTMConfig(5, 16, 0.0), id="order-length"
        ),
        pytest.param(
            "AAPL", (0, -1, 0), LSTMConfig(5, 16, 0.0), LSTMConfig(5, 16, 0.0), id="negative-order"
        ),
        pytest.param("AAPL", (0, 0, 0), "invalid", LSTMConfig(5, 16, 0.0), id="lstm-config"),
        pytest.param("AAPL", (0, 0, 0), LSTMConfig(5, 16, 0.0), "invalid", id="residual-config"),
    ],
)
def test_ticker_spec_rejects_invalid_fields(
    ticker: object,
    order: object,
    lstm_config: object,
    residual_config: object,
) -> None:
    """Reject malformed frozen model specifications"""
    with pytest.raises(ValueError):
        FinalTickerSpec(
            cast(Any, ticker),
            cast(Any, order),
            cast(Any, lstm_config),
            cast(Any, residual_config),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("inner_train_fraction", 0.0, id="zero-fraction"),
        pytest.param("inner_train_fraction", 1.0, id="full-fraction"),
        pytest.param("inner_train_fraction", True, id="boolean-fraction"),
        pytest.param("max_epochs", 199, id="epochs"),
        pytest.param("batch_size", 16, id="batch-size"),
        pytest.param("early_stopping_patience", 14, id="patience"),
        pytest.param("min_delta", 0.1, id="minimum-delta"),
        pytest.param("learning_rate", 0.01, id="learning-rate"),
        pytest.param("shuffle", True, id="shuffle"),
    ],
)
def test_training_spec_rejects_changed_procedure(field: str, value: object) -> None:
    """Reject changes to the fixed Keras procedure"""
    values = dict(vars(EXPECTED_TRAINING))
    values[field] = value

    with pytest.raises(ValueError):
        FinalTrainingSpec(**cast(Any, values))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("ticker_specs", EXPECTED_TICKER_SPECS[:-1], id="missing-ticker"),
        pytest.param(
            "ticker_specs",
            (*EXPECTED_TICKER_SPECS[:-1], EXPECTED_TICKER_SPECS[0]),
            id="duplicate-ticker",
        ),
        pytest.param("ticker_specs", tuple(reversed(EXPECTED_TICKER_SPECS)), id="ticker-order"),
        pytest.param("seeds", tuple(range(9)), id="missing-seed"),
        pytest.param("seeds", tuple(reversed(range(10))), id="seed-order"),
        pytest.param(
            "split_config",
            replace(DEFAULT_SPLIT_CONFIG, test_end="2025-12-30"),
            id="split-boundary",
        ),
        pytest.param("arima_warmup", 0, id="warmup"),
        pytest.param("volatility_window", 0, id="volatility-window"),
        pytest.param("annualization_days", 0, id="annualization"),
        pytest.param("model_comparisons", EXPECTED_COMPARISONS[:-1], id="missing-comparison"),
        pytest.param(
            "model_comparisons", tuple(reversed(EXPECTED_COMPARISONS)), id="comparison-order"
        ),
        pytest.param("dm_lag", -1, id="negative-lag"),
        pytest.param("dm_lag", 19, id="lag-block-mismatch"),
        pytest.param("holm_alpha", 0.0, id="zero-alpha"),
        pytest.param("holm_alpha", 1.0, id="full-alpha"),
        pytest.param("bootstrap_block", 0, id="bootstrap-block"),
        pytest.param("bootstrap_repetitions", 0, id="bootstrap-repetitions"),
        pytest.param("bootstrap_seed", -1, id="bootstrap-seed"),
        pytest.param("bootstrap_confidence", 0.0, id="zero-confidence"),
        pytest.param("bootstrap_confidence", 1.0, id="full-confidence"),
    ],
)
def test_protocol_rejects_invalid_research_choices(field: str, value: object) -> None:
    """Reject incomplete or inconsistent protocol choices"""
    with pytest.raises(ValueError):
        replace_protocol(field, value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("arima_warmup", True, id="boolean-warmup"),
        pytest.param("volatility_window", 21.0, id="float-window"),
        pytest.param("annualization_days", True, id="boolean-annualization"),
        pytest.param("dm_lag", 20.0, id="float-lag"),
        pytest.param("bootstrap_block", True, id="boolean-block"),
        pytest.param("bootstrap_repetitions", 10_000.0, id="float-repetitions"),
        pytest.param("bootstrap_seed", 42.0, id="float-seed"),
    ],
)
def test_protocol_rejects_noninteger_count_fields(field: str, value: object) -> None:
    """Reject booleans and floats in count fields"""
    with pytest.raises(ValueError):
        replace_protocol(field, value)


def test_protocol_allows_structurally_valid_inference_settings() -> None:
    """Permit compact inference settings for controlled tests"""
    protocol = replace(
        FINAL_TEST_PROTOCOL,
        dm_lag=1,
        bootstrap_block=2,
        bootstrap_repetitions=5,
        bootstrap_seed=7,
        holm_alpha=0.1,
        bootstrap_confidence=0.8,
    )

    assert protocol.dm_lag == 1
    assert protocol.bootstrap_block == 2
    assert protocol.bootstrap_repetitions == 5


def test_protocol_rejects_invalid_training_spec_type() -> None:
    """Reject training objects outside frozen contract"""
    with pytest.raises(ValueError, match="training"):
        replace_protocol("training", "invalid")


def test_validate_final_dataset_returns_sorted_isolated_copy() -> None:
    """Normalize dates and isolate deterministic validated data"""
    source = valid_dataset()
    original = source.copy(deep=True)

    result = validate_final_dataset(source)

    expected_order = result.sort_values(["ticker", "target_date"]).index
    assert expected_order.tolist() == result.index.tolist()
    assert pd.api.types.is_datetime64_ns_dtype(result["date"])
    assert pd.api.types.is_datetime64_ns_dtype(result["target_date"])
    assert set(result["ticker"]) == {"AAPL", "JPM", "SPY", "XOM"}
    assert set(result["split"]) == {"train", "validation", "test"}
    assert result.loc[result["split"].eq("test"), "target_date"].min() == pd.Timestamp("2020-01-02")
    assert result.loc[result["split"].eq("test"), "target_date"].max() == pd.Timestamp("2025-12-31")
    pd.testing.assert_frame_equal(source, original)

    source.loc[:, "ticker"] = "CHANGED"
    assert set(result["ticker"]) == {"AAPL", "JPM", "SPY", "XOM"}


@pytest.mark.parametrize(
    "column",
    ["date", "target_date", "ticker", "adj_close", "log_return", "target_return", "split"],
)
def test_validate_final_dataset_rejects_missing_columns(column: str) -> None:
    """Reject every missing required dataset column"""
    with pytest.raises(ValueError, match="missing columns"):
        validate_final_dataset(valid_dataset().drop(columns=column))


@pytest.mark.parametrize("column", ["adj_close", "log_return", "target_return"])
@pytest.mark.parametrize("value", [np.nan, np.inf, "invalid"])
def test_validate_final_dataset_rejects_invalid_numeric_values(
    column: str,
    value: object,
) -> None:
    """Reject nonnumeric missing or infinite model values"""
    dataset = valid_dataset()
    if isinstance(value, str):
        dataset[column] = dataset[column].astype(object)
    dataset.loc[0, column] = cast(Any, value)

    with pytest.raises(ValueError, match=column):
        validate_final_dataset(dataset)


@pytest.mark.parametrize("column", ["date", "target_date"])
def test_validate_final_dataset_rejects_invalid_dates(column: str) -> None:
    """Reject unparseable source and target dates"""
    dataset = valid_dataset()
    dataset.loc[0, column] = "not-a-date"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=column):
            validate_final_dataset(dataset)


@pytest.mark.parametrize("column", ["date", "target_date"])
def test_validate_final_dataset_rejects_missing_dates(column: str) -> None:
    """Reject missing source and target dates"""
    dataset = valid_dataset()
    dataset.loc[0, column] = None

    with pytest.raises(ValueError, match=rf"^{column} must contain valid dates$"):
        validate_final_dataset(dataset)


def test_validate_final_dataset_rejects_empty_data() -> None:
    """Reject empty final evaluation snapshots"""
    with pytest.raises(ValueError, match="must not be empty"):
        validate_final_dataset(valid_dataset().iloc[0:0])


def test_validate_final_dataset_rejects_invalid_protocol_type() -> None:
    """Reject protocol objects outside final contract"""
    with pytest.raises(ValueError, match="protocol"):
        validate_final_dataset(valid_dataset(), protocol="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param(
            lambda frame: frame[frame["ticker"].ne("XOM")], "tickers", id="missing-ticker"
        ),
        pytest.param(lambda frame: frame.assign(ticker="TSLA"), "tickers", id="unknown-ticker"),
        pytest.param(
            lambda frame: frame[frame["split"].ne("validation")], "splits", id="missing-split"
        ),
        pytest.param(lambda frame: frame.assign(split="unknown"), "split", id="unknown-split"),
        pytest.param(
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate",
            id="duplicate-key",
        ),
    ],
)
def test_validate_final_dataset_rejects_incomplete_identity(
    mutation: Any,
    message: str,
) -> None:
    """Reject incomplete ticker split and key identity"""
    with pytest.raises(ValueError, match=message):
        validate_final_dataset(mutation(valid_dataset()))


def test_validate_final_dataset_requires_each_ticker_split() -> None:
    """Require every ticker in every research split"""
    dataset = valid_dataset()
    dataset = dataset.loc[~(dataset["ticker"].eq("AAPL") & dataset["split"].eq("validation"))]

    with pytest.raises(ValueError, match="ticker/split"):
        validate_final_dataset(dataset)


@pytest.mark.parametrize(
    ("row_filter", "message"),
    [
        pytest.param(
            lambda frame: frame.loc[
                ~(frame["split"].eq("test") & frame["target_date"].eq("2020-01-02"))
            ],
            "test interval",
            id="missing-test-start",
        ),
        pytest.param(
            lambda frame: frame.loc[
                ~(frame["split"].eq("test") & frame["target_date"].eq("2025-12-31"))
            ],
            "test interval",
            id="missing-test-end",
        ),
    ],
)
def test_validate_final_dataset_requires_complete_test_boundaries(
    row_filter: Any,
    message: str,
) -> None:
    """Require both canonical final-test boundary dates"""
    with pytest.raises(ValueError, match=message):
        validate_final_dataset(row_filter(valid_dataset()))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        pytest.param("target_date", "2019-12-31", "split boundaries", id="wrong-test-label"),
        pytest.param("target_date", "2016-01-02", "split boundaries", id="gap-date"),
        pytest.param("date", "2025-12-31", "precede", id="noncausal-origin"),
    ],
)
def test_validate_final_dataset_rejects_temporal_inconsistency(
    column: str,
    value: object,
    message: str,
) -> None:
    """Reject mislabeled or noncausal observation dates"""
    dataset = valid_dataset()
    row = int(dataset.index[dataset["split"].eq("test")][0])
    if column == "target_date":
        dataset.loc[row, "date"] = "2015-12-30"
    dataset.loc[row, column] = cast(Any, value)

    with pytest.raises(ValueError, match=message):
        validate_final_dataset(dataset)


def final_training_dataset() -> pd.DataFrame:
    """Return enough canonical rows for final LSTM tests"""
    train_targets = pd.bdate_range("2015-12-18", "2015-12-31")
    validation_targets = pd.bdate_range("2019-12-24", "2019-12-31")
    test_targets = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2025-12-31"])
    schedule = [
        *((target - pd.offsets.BDay(1), target, "train") for target in train_targets),
        *((target - pd.offsets.BDay(1), target, "validation") for target in validation_targets),
        *((target - pd.offsets.BDay(1), target, "test") for target in test_targets),
    ]
    rows: list[dict[str, object]] = []
    for ticker_position, ticker in enumerate(("AAPL", "JPM", "SPY", "XOM"), start=1):
        for row_position, (date, target_date, split) in enumerate(schedule, start=1):
            current_return = ticker_position / 100 + row_position / 1_000
            rows.append(
                {
                    "date": date,
                    "target_date": target_date,
                    "ticker": ticker,
                    "adj_close": 100.0 + ticker_position + row_position,
                    "log_return": current_return,
                    "target_return": current_return + 0.0005,
                    "split": split,
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=29).reset_index(drop=True)


class FinalRecordingModel:
    """Record final observations and emit controlled predictions."""

    def __init__(
        self,
        config: LSTMConfig,
        name: str,
        prediction: float | Callable[[pd.DataFrame], pd.Series] = 0.01,
        *,
        parameter_count: int = 73,
        error: Exception | None = None,
    ) -> None:
        self.config = config
        self.name = name
        self.parameter_count = parameter_count
        self.prediction = prediction
        self.error = error
        self.received: list[pd.DataFrame] = []

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Capture raw test lags and return forecasts"""
        self.received.append(observations.copy(deep=True))
        if self.error is not None:
            raise self.error
        if callable(self.prediction):
            return self.prediction(observations)
        return pd.Series(
            self.prediction,
            index=observations.index,
            dtype="float64",
            name=PREDICTION_NAME,
        )


class FinalRecordingTrainer:
    """Record both stages of final LSTM training."""

    def __init__(
        self,
        *,
        best_epoch: int = 3,
        prediction: float | Callable[[pd.DataFrame], pd.Series] = 0.01,
        returned_name: str | None = None,
        returned_config: LSTMConfig | None = None,
        parameter_count: int = 73,
        determine_error: Exception | None = None,
        fit_error: Exception | None = None,
        predict_error: Exception | None = None,
    ) -> None:
        self.best_epoch = best_epoch
        self.prediction = prediction
        self.returned_name = returned_name
        self.returned_config = returned_config
        self.parameter_count = parameter_count
        self.determine_error = determine_error
        self.fit_error = fit_error
        self.predict_error = predict_error
        self.determine_calls: list[dict[str, object]] = []
        self.fit_calls: list[dict[str, object]] = []
        self.models: list[FinalRecordingModel] = []

    def determine_best_epoch(
        self,
        config: LSTMConfig,
        seed: int,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
    ) -> int:
        """Record chronological inner partitions and return epoch"""
        self.determine_calls.append(
            {
                "config": config,
                "seed": seed,
                "train_features": train_features.copy(),
                "train_targets": train_targets.copy(),
                "validation_features": validation_features.copy(),
                "validation_targets": validation_targets.copy(),
            }
        )
        if self.determine_error is not None:
            raise self.determine_error
        return self.best_epoch

    def fit(
        self,
        config: LSTMConfig,
        seed: int,
        features: np.ndarray,
        targets: np.ndarray,
        scaler: StandardScaler,
        feature_columns: tuple[str, ...],
        epochs: int,
        name: str,
    ) -> FinalRecordingModel:
        """Record final fit and return controlled model"""
        self.fit_calls.append(
            {
                "config": config,
                "seed": seed,
                "features": features.copy(),
                "targets": targets.copy(),
                "scaler": scaler,
                "feature_columns": feature_columns,
                "epochs": epochs,
                "name": name,
            }
        )
        if self.fit_error is not None:
            raise self.fit_error
        model = FinalRecordingModel(
            self.returned_config or config,
            self.returned_name or name,
            self.prediction,
            parameter_count=self.parameter_count,
            error=self.predict_error,
        )
        self.models.append(model)
        return model


def expected_sequence_parts(
    dataset: pd.DataFrame,
    ticker: str,
    window: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build independent training and test sequence expectations"""
    rows = validate_final_dataset(dataset)
    rows = rows.loc[rows["ticker"].eq(ticker)].sort_values(["date", "target_date"])
    columns = lag_columns(window)
    features = pd.DataFrame(
        {
            column: rows["log_return"].shift(lag)
            for column, lag in zip(columns, range(window - 1, -1, -1), strict=True)
        },
        index=rows.index,
    )
    complete = features.notna().all(axis=1)
    training = complete & rows["split"].isin(("train", "validation"))
    test = complete & rows["split"].eq("test")
    keys = ["date", "target_date", "ticker", "split"]
    return (
        features.loc[training].astype("float64"),
        rows.loc[training, "target_return"].astype("float64"),
        rows.loc[training, keys].copy(),
        features.loc[test].astype("float64"),
        rows.loc[test, "target_return"].astype("float64"),
        rows.loc[test, keys].copy(),
    )


def recorded_array(call: dict[str, object], key: str) -> np.ndarray:
    """Return one recorded numeric trainer argument"""
    return cast(np.ndarray, call[key])


def test_run_final_lstm_seed_executes_two_stage_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train chronologically then refit all permitted history"""
    dataset = final_training_dataset()
    original = dataset.copy(deep=True)
    config = LSTMConfig(5, 32, 0.0)
    trainer = FinalRecordingTrainer(best_epoch=3)
    scaler_calls: list[tuple[pd.DataFrame, str, pd.Timestamp]] = []

    def record_scaler(
        frame: pd.DataFrame,
        ticker: str,
        through_date: pd.Timestamp,
    ) -> StandardScaler:
        scaler_calls.append((frame.copy(deep=True), ticker, pd.Timestamp(through_date)))
        return fit_return_scaler(frame, ticker, through_date)

    monkeypatch.setattr(final_models_module, "fit_return_scaler", record_scaler)

    result = run_final_lstm_seed(
        dataset,
        "AAPL",
        config,
        4,
        cast(LSTMTrainer, trainer),
        "lstm",
    )

    pd.testing.assert_frame_equal(dataset, original)
    features, targets, keys, test_features, test_targets, test_keys = expected_sequence_parts(
        dataset, "AAPL", config.window
    )
    boundary = int(len(features) * 0.8)
    inner_cutoff = keys.iloc[:boundary]["date"].max()
    full_cutoff = keys["date"].max()
    training_data = validate_final_dataset(dataset).loc[
        lambda frame: frame["split"].isin(("train", "validation"))
    ]
    inner_scaler = fit_return_scaler(training_data, "AAPL", inner_cutoff)
    full_scaler = fit_return_scaler(training_data, "AAPL", full_cutoff)

    assert len(trainer.determine_calls) == 1
    determine_call = trainer.determine_calls[0]
    assert determine_call["config"] == config
    assert determine_call["seed"] == 4
    np.testing.assert_allclose(
        recorded_array(determine_call, "train_features"),
        transform_features(features.iloc[:boundary], inner_scaler),
    )
    np.testing.assert_allclose(
        recorded_array(determine_call, "train_targets"),
        transform_targets(targets.iloc[:boundary], inner_scaler),
    )
    np.testing.assert_allclose(
        recorded_array(determine_call, "validation_features"),
        transform_features(features.iloc[boundary:], inner_scaler),
    )
    np.testing.assert_allclose(
        recorded_array(determine_call, "validation_targets"),
        transform_targets(targets.iloc[boundary:], inner_scaler),
    )

    assert len(trainer.fit_calls) == 1
    fit_call = trainer.fit_calls[0]
    assert fit_call["config"] == config
    assert fit_call["seed"] == 4
    assert fit_call["epochs"] == 3
    assert fit_call["name"] == "lstm_seed_04"
    assert fit_call["feature_columns"] == lag_columns(config.window)
    np.testing.assert_allclose(
        recorded_array(fit_call, "features"),
        transform_features(features, full_scaler),
    )
    np.testing.assert_allclose(
        recorded_array(fit_call, "targets"),
        transform_targets(targets, full_scaler),
    )
    assert [call[2] for call in scaler_calls] == [inner_cutoff, full_cutoff]
    assert all(call[1] == "AAPL" for call in scaler_calls)
    assert all(set(call[0]["split"]) == {"train", "validation"} for call in scaler_calls)

    assert len(trainer.models) == 1
    model = trainer.models[0]
    assert len(model.received) == 1
    pd.testing.assert_frame_equal(model.received[0], test_features)
    assert model.received[0].columns.tolist() == list(lag_columns(config.window))
    assert result.ticker == "AAPL"
    assert result.config == config
    assert result.seed == 4
    assert result.model == "lstm_seed_04"
    assert result.best_epoch == 3
    assert result.parameter_count == 73
    assert result.predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert result.predictions["split"].eq("test").all()
    assert result.predictions["model"].eq("lstm_seed_04").all()
    assert result.predictions["actual_return"].tolist() == pytest.approx(test_targets.tolist())
    assert result.predictions["predicted_return"].tolist() == pytest.approx([0.01] * len(test_keys))
    pd.testing.assert_frame_equal(
        result.predictions.loc[:, ["date", "target_date", "ticker", "split"]],
        test_keys.reset_index(drop=True),
    )


def test_run_final_lstm_seed_uses_realized_test_returns_only() -> None:
    """Exclude test targets and future returns from forecasts"""

    def latest_return(observations: pd.DataFrame) -> pd.Series:
        return observations["lag_0"].rename(PREDICTION_NAME)

    dataset = final_training_dataset()
    baseline_trainer = FinalRecordingTrainer(prediction=latest_return)
    baseline = run_final_lstm_seed(
        dataset,
        "AAPL",
        LSTMConfig(5, 32, 0.0),
        0,
        cast(LSTMTrainer, baseline_trainer),
        "lstm",
    )
    changed_target = dataset.copy(deep=True)
    test_mask = changed_target["ticker"].eq("AAPL") & changed_target["split"].eq("test")
    last_test = pd.to_datetime(changed_target.loc[test_mask, "target_date"]).idxmax()
    changed_target.loc[last_test, "target_return"] = 99.0
    target_trainer = FinalRecordingTrainer(prediction=latest_return)
    target_result = run_final_lstm_seed(
        changed_target,
        "AAPL",
        LSTMConfig(5, 32, 0.0),
        0,
        cast(LSTMTrainer, target_trainer),
        "lstm",
    )

    assert baseline.predictions["predicted_return"].tolist() == pytest.approx(
        target_result.predictions["predicted_return"].tolist()
    )
    assert baseline.predictions["actual_return"].iloc[:-1].tolist() == pytest.approx(
        target_result.predictions["actual_return"].iloc[:-1].tolist()
    )
    assert target_result.predictions["actual_return"].iloc[-1] == pytest.approx(99.0)
    assert len(target_trainer.determine_calls) == len(target_trainer.fit_calls) == 1
    assert target_trainer.models[0].received[0].columns.tolist() == list(lag_columns(5))

    changed_return = dataset.copy(deep=True)
    changed_return.loc[last_test, "log_return"] = 77.0
    return_result = run_final_lstm_seed(
        changed_return,
        "AAPL",
        LSTMConfig(5, 32, 0.0),
        0,
        cast(LSTMTrainer, FinalRecordingTrainer(prediction=latest_return)),
        "lstm",
    )
    assert baseline.predictions["predicted_return"].iloc[:-1].tolist() == pytest.approx(
        return_result.predictions["predicted_return"].iloc[:-1].tolist()
    )
    assert return_result.predictions["predicted_return"].iloc[-1] == pytest.approx(77.0)


@pytest.mark.parametrize("stage", ["determine", "fit", "predict"])
def test_run_final_lstm_seed_propagates_failed_seed(stage: str) -> None:
    """Stop final evaluation when any seed fails"""
    error = RuntimeError(f"{stage} failed")
    trainer = FinalRecordingTrainer(
        determine_error=error if stage == "determine" else None,
        fit_error=error if stage == "fit" else None,
        predict_error=error if stage == "predict" else None,
    )

    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        run_final_lstm_seed(
            final_training_dataset(),
            "AAPL",
            LSTMConfig(5, 32, 0.0),
            0,
            cast(LSTMTrainer, trainer),
            "lstm",
        )


@pytest.mark.parametrize(
    ("ticker", "config", "seed", "model_name", "message"),
    [
        pytest.param("TSLA", LSTMConfig(5, 32, 0.0), 0, "lstm", "ticker", id="ticker"),
        pytest.param("AAPL", LSTMConfig(21, 32, 0.0), 0, "lstm", "config", id="config"),
        pytest.param("AAPL", LSTMConfig(5, 32, 0.0), -1, "lstm", "seed", id="seed"),
        pytest.param("AAPL", LSTMConfig(5, 32, 0.0), 0, "", "model", id="model"),
    ],
)
def test_run_final_lstm_seed_rejects_invalid_identity(
    ticker: str,
    config: LSTMConfig,
    seed: int,
    model_name: str,
    message: str,
) -> None:
    """Reject runs outside the frozen protocol"""
    with pytest.raises(ValueError, match=message):
        run_final_lstm_seed(
            final_training_dataset(),
            ticker,
            config,
            seed,
            cast(LSTMTrainer, FinalRecordingTrainer()),
            model_name,
        )


@pytest.mark.parametrize("best_epoch", [0, 201, True])
def test_run_final_lstm_seed_rejects_invalid_selected_epoch(best_epoch: object) -> None:
    """Reject epochs outside the fixed training range"""
    trainer = FinalRecordingTrainer(best_epoch=cast(int, best_epoch))

    with pytest.raises(ValueError, match="epoch"):
        run_final_lstm_seed(
            final_training_dataset(),
            "AAPL",
            LSTMConfig(5, 32, 0.0),
            0,
            cast(LSTMTrainer, trainer),
            "lstm",
        )


def test_run_final_lstm_seed_rejects_insufficient_sequence_history() -> None:
    """Reject ticker history shorter than selected window"""
    dataset = final_training_dataset()
    aapl = dataset.loc[dataset["ticker"].eq("AAPL")]
    retained = pd.concat(
        [
            aapl.loc[aapl["split"].eq("train")].sort_values("target_date").iloc[[0]],
            aapl.loc[aapl["split"].eq("validation")].sort_values("target_date").iloc[[0]],
            aapl.loc[aapl["split"].eq("test")].sort_values("target_date").iloc[[0, -1]],
        ],
        ignore_index=True,
    )
    dataset = pd.concat(
        [dataset.loc[dataset["ticker"].ne("AAPL")], retained],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="insufficient history"):
        run_final_lstm_seed(
            dataset,
            "AAPL",
            LSTMConfig(5, 32, 0.0),
            0,
            cast(LSTMTrainer, FinalRecordingTrainer()),
            "lstm",
        )


@pytest.mark.parametrize(
    ("trainer", "message"),
    [
        pytest.param(
            FinalRecordingTrainer(returned_name="wrong"),
            "identity",
            id="model-name",
        ),
        pytest.param(
            FinalRecordingTrainer(returned_config=LSTMConfig(5, 16, 0.0)),
            "identity",
            id="model-config",
        ),
        pytest.param(
            FinalRecordingTrainer(parameter_count=0),
            "parameter_count",
            id="parameter-count",
        ),
    ],
)
def test_run_final_lstm_seed_rejects_invalid_fitted_model(
    trainer: FinalRecordingTrainer,
    message: str,
) -> None:
    """Reject fitted models violating requested run identity"""
    with pytest.raises(ValueError, match=message):
        run_final_lstm_seed(
            final_training_dataset(),
            "AAPL",
            LSTMConfig(5, 32, 0.0),
            0,
            cast(LSTMTrainer, trainer),
            "lstm",
        )


def seed_prediction_table(seed: int, *, model_name: str = "lstm") -> pd.DataFrame:
    """Return two aligned seed-level final predictions"""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "ticker": ["AAPL", "AAPL"],
            "split": ["test", "test"],
            "model": [f"{model_name}_seed_{seed:02d}"] * 2,
            "actual_return": [0.01, -0.02],
            "predicted_return": [seed / 1_000, -seed / 2_000],
        }
    )
    return frame.loc[:, PREDICTION_COLUMNS]


def seed_result(
    seed: int,
    *,
    ticker: str = "AAPL",
    config: LSTMConfig | None = None,
    model_name: str = "lstm",
) -> FinalSeedResult:
    """Return one controlled successful final seed result"""
    selected_config = config or LSTMConfig(5, 32, 0.0)
    return FinalSeedResult(
        ticker=ticker,
        config=selected_config,
        seed=seed,
        model=f"{model_name}_seed_{seed:02d}",
        best_epoch=2,
        parameter_count=73,
        predictions=seed_prediction_table(seed, model_name=model_name),
    )


def test_average_final_seed_predictions_averages_exact_ten_seeds() -> None:
    """Average aligned seeds without mutating their tables"""
    results = tuple(seed_result(seed) for seed in reversed(range(10)))
    originals = [result.predictions.copy(deep=True) for result in results]

    averaged = average_final_seed_predictions(results, "lstm")

    assert averaged.columns.tolist() == list(PREDICTION_COLUMNS)
    assert averaged["model"].eq("lstm").all()
    assert averaged["split"].eq("test").all()
    assert averaged["actual_return"].tolist() == pytest.approx([0.01, -0.02])
    assert averaged["predicted_return"].tolist() == pytest.approx([0.0045, -0.00225])
    for result, original in zip(results, originals, strict=True):
        pd.testing.assert_frame_equal(result.predictions, original)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        pytest.param(lambda values: values[:-1], "seeds", id="missing-seed"),
        pytest.param(
            lambda values: (*values[:-1], seed_result(0)),
            "seeds",
            id="duplicate-seed",
        ),
        pytest.param(
            lambda values: (*values[:-1], seed_result(9, ticker="JPM")),
            "ticker",
            id="ticker",
        ),
        pytest.param(
            lambda values: (*values[:-1], seed_result(9, config=LSTMConfig(5, 16, 0.0))),
            "config",
            id="config",
        ),
    ],
)
def test_average_final_seed_predictions_rejects_incomplete_ensemble(
    mutator: Callable[[tuple[FinalSeedResult, ...]], tuple[FinalSeedResult, ...]],
    message: str,
) -> None:
    """Reject incomplete mixed or duplicate seed ensembles"""
    results = tuple(seed_result(seed) for seed in range(10))

    with pytest.raises(ValueError, match=message):
        average_final_seed_predictions(mutator(results), "lstm")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("key", "align", id="key"),
        pytest.param("actual", "actual", id="actual"),
        pytest.param("prediction", "finite", id="prediction"),
        pytest.param("split", "test", id="split"),
        pytest.param("table-model", "model", id="table-model"),
        pytest.param("result-model", "model", id="result-model"),
        pytest.param("duplicate", "duplicate", id="duplicate"),
        pytest.param("schema", "columns", id="schema"),
        pytest.param("date", "datetime", id="date"),
        pytest.param("empty", "empty", id="empty"),
        pytest.param("missing-date", "finite dates", id="missing-date"),
        pytest.param("ticker", "ticker", id="ticker"),
    ],
)
def test_average_final_seed_predictions_rejects_invalid_tables(
    case: str,
    message: str,
) -> None:
    """Reject malformed or misaligned seed prediction tables"""
    results = list(seed_result(seed) for seed in range(10))
    invalid = results[-1]
    predictions = invalid.predictions.copy(deep=True)
    model = invalid.model
    if case == "key":
        predictions.loc[predictions.index[0], "target_date"] = pd.Timestamp("2020-01-06")
    elif case == "actual":
        predictions.loc[predictions.index[0], "actual_return"] = 9.0
    elif case == "prediction":
        predictions.loc[predictions.index[0], "predicted_return"] = np.inf
    elif case == "split":
        predictions.loc[predictions.index[0], "split"] = "validation"
    elif case == "table-model":
        predictions.loc[predictions.index[0], "model"] = "wrong"
    elif case == "result-model":
        model = "wrong"
    elif case == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif case == "schema":
        predictions = predictions.drop(columns="date")
    elif case == "date":
        predictions["date"] = predictions["date"].astype(str)
    elif case == "empty":
        predictions = predictions.iloc[0:0]
    elif case == "missing-date":
        predictions.loc[predictions.index[0], "date"] = pd.NaT
    else:
        predictions.loc[predictions.index[0], "ticker"] = "JPM"
    results[-1] = replace(invalid, model=model, predictions=predictions)

    with pytest.raises(ValueError, match=message):
        average_final_seed_predictions(tuple(results), "lstm")


def test_average_final_seed_predictions_rejects_invalid_result_type() -> None:
    """Reject values outside final seed result contract"""
    results: tuple[object, ...] = (*tuple(seed_result(seed) for seed in range(9)), object())

    with pytest.raises(ValueError, match="FinalSeedResult"):
        average_final_seed_predictions(cast(Any, results), "lstm")


def test_average_final_seed_predictions_rejects_invalid_model_name() -> None:
    """Reject empty ensemble model identifiers"""
    with pytest.raises(ValueError, match="model"):
        average_final_seed_predictions(tuple(seed_result(seed) for seed in range(10)), "")
