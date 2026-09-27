"""Tests for the frozen independent final-test protocol."""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from analysis.preparation import DEFAULT_SPLIT_CONFIG
from evaluation.final_models import (
    FINAL_TEST_PROTOCOL,
    FinalTestProtocol,
    FinalTickerSpec,
    FinalTrainingSpec,
    validate_final_dataset,
)
from models.lstm import LSTMConfig

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
