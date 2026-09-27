"""Tests for aligned expanding ARIMA residual data."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pytest

from evaluation.arima_residuals import (
    ARIMA_RESIDUAL_WARMUP,
    FIXED_ARIMA_ORDERS,
    ARIMAResidualData,
    build_residual_dataset,
    validate_arima_residual_data,
)
from evaluation.contracts import PREDICTION_COLUMNS

DATASET_COLUMNS = [
    "date",
    "target_date",
    "ticker",
    "adj_close",
    "log_return",
    "target_return",
    "split",
]


@pytest.fixture
def source_dataset() -> pd.DataFrame:
    """Return unsorted core rows spanning every split"""
    dates = pd.to_datetime(["2019-12-25", "2019-12-26", "2019-12-27", "2019-12-30", "2019-12-31"])
    target_dates = pd.to_datetime(
        ["2019-12-26", "2019-12-27", "2019-12-30", "2019-12-31", "2020-01-02"]
    )
    frames = []
    for ticker_position, ticker in enumerate(("AAPL", "JPM")):
        offset = ticker_position / 100.0
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "target_date": target_dates,
                    "ticker": ticker,
                    "adj_close": 100.0 + 10 * ticker_position + np.arange(5),
                    "log_return": offset + np.arange(5, dtype="float64") / 1_000,
                    "target_return": offset + np.arange(1, 6, dtype="float64") / 1_000,
                    "split": ["train", "train", "validation", "validation", "test"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=17)


@pytest.fixture
def arima_predictions(source_dataset: pd.DataFrame) -> pd.DataFrame:
    """Return unsorted ARIMA forecasts for evaluation rows"""
    evaluation = source_dataset.loc[source_dataset["split"].ne("test")].copy()
    predictions = evaluation.loc[:, ["date", "target_date", "ticker", "split"]]
    predictions["model"] = "arima"
    predictions["actual_return"] = evaluation["target_return"]
    predictions["predicted_return"] = np.where(
        predictions["ticker"].eq("AAPL"),
        0.00025,
        -0.0005,
    )
    return predictions.loc[:, PREDICTION_COLUMNS].sample(frac=1, random_state=23)


def test_residual_protocol_constants_are_frozen() -> None:
    """Expose fixed warmup and approved ticker orders"""
    assert ARIMA_RESIDUAL_WARMUP == 252
    assert dict(FIXED_ARIMA_ORDERS) == {
        "AAPL": (0, 0, 0),
        "JPM": (0, 0, 1),
        "SPY": (0, 0, 2),
        "XOM": (2, 0, 1),
    }
    with pytest.raises(TypeError):
        cast(dict[str, tuple[int, int, int]], FIXED_ARIMA_ORDERS)["AAPL"] = (1, 0, 0)


def test_build_residual_dataset_aligns_values_and_boundaries(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Align consecutive residuals across split boundaries"""
    original_dataset = source_dataset.copy(deep=True)
    original_predictions = arima_predictions.copy(deep=True)

    result = build_residual_dataset(source_dataset, arima_predictions)

    pd.testing.assert_frame_equal(source_dataset, original_dataset)
    pd.testing.assert_frame_equal(arima_predictions, original_predictions)
    assert result.columns.tolist() == DATASET_COLUMNS
    assert result[["ticker", "target_date"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2019-12-27")],
        ["AAPL", pd.Timestamp("2019-12-30")],
        ["AAPL", pd.Timestamp("2019-12-31")],
        ["JPM", pd.Timestamp("2019-12-27")],
        ["JPM", pd.Timestamp("2019-12-30")],
        ["JPM", pd.Timestamp("2019-12-31")],
    ]

    ordered = arima_predictions.sort_values(["ticker", "target_date"])
    current_prediction = ordered.loc[
        ordered["ticker"].eq("AAPL") & ordered["target_date"].eq("2019-12-27")
    ].iloc[0]
    next_prediction = ordered.loc[
        ordered["ticker"].eq("AAPL") & ordered["target_date"].eq("2019-12-30")
    ].iloc[0]
    residual_row = result.loc[
        result["ticker"].eq("AAPL") & result["target_date"].eq("2019-12-30")
    ].iloc[0]
    source_origin = source_dataset.loc[
        source_dataset["ticker"].eq("AAPL")
        & source_dataset["date"].eq(current_prediction["target_date"])
    ].iloc[0]

    assert residual_row["date"] == current_prediction["target_date"]
    assert residual_row["target_date"] == next_prediction["target_date"]
    assert residual_row["log_return"] == pytest.approx(
        current_prediction["actual_return"] - current_prediction["predicted_return"]
    )
    assert residual_row["target_return"] == pytest.approx(
        next_prediction["actual_return"] - next_prediction["predicted_return"]
    )
    assert residual_row["adj_close"] == source_origin["adj_close"]
    assert residual_row["split"] == "validation"


def test_validate_arima_residual_data_returns_isolated_tables(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Return sorted predictions and aligned residual dataset"""
    result = validate_arima_residual_data(source_dataset, arima_predictions)

    assert isinstance(result, ARIMAResidualData)
    assert result.arima_predictions[["ticker", "target_date"]].values.tolist() == sorted(
        arima_predictions[["ticker", "target_date"]].values.tolist()
    )
    assert result.residual_dataset["split"].tolist() == [
        "train",
        "validation",
        "validation",
        "train",
        "validation",
        "validation",
    ]
    assert result.arima_predictions is not arima_predictions


def test_weekend_transition_uses_trading_date_equality(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Accept Friday origins targeting the following Monday"""
    result = build_residual_dataset(source_dataset, arima_predictions)

    weekend_row = result.loc[result["ticker"].eq("AAPL") & result["date"].eq("2019-12-27")].iloc[0]
    assert weekend_row["target_date"] == pd.Timestamp("2019-12-30")


@pytest.mark.parametrize("removed_position", [1, -1])
def test_residual_data_rejects_missing_prediction_keys(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
    removed_position: int,
) -> None:
    """Reject missing interior and trailing forecast keys"""
    ordered = arima_predictions.sort_values(["ticker", "target_date"])
    aapl_indexes = ordered.loc[ordered["ticker"].eq("AAPL")].index
    malformed = arima_predictions.drop(aapl_indexes[removed_position])

    with pytest.raises(ValueError, match="missing prediction keys"):
        build_residual_dataset(source_dataset, malformed)


def test_residual_data_rejects_missing_ticker_predictions(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Reject omitted ticker forecast histories"""
    malformed = arima_predictions.loc[arima_predictions["ticker"].ne("JPM")]

    with pytest.raises(ValueError, match="missing prediction keys"):
        build_residual_dataset(source_dataset, malformed)


def test_residual_data_rejects_extra_prediction_keys(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Reject forecast keys absent from source data"""
    extra = arima_predictions.iloc[[0]].copy()
    extra["date"] = pd.Timestamp("2019-12-24")
    extra["target_date"] = pd.Timestamp("2019-12-25")
    malformed = pd.concat([arima_predictions, extra], ignore_index=True)

    with pytest.raises(ValueError, match="extra prediction keys"):
        build_residual_dataset(source_dataset, malformed)


def test_residual_data_rejects_duplicate_prediction_rows(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Reject duplicated ARIMA observation keys"""
    malformed = pd.concat([arima_predictions, arima_predictions.iloc[[0]]])

    with pytest.raises(ValueError, match="duplicate"):
        build_residual_dataset(source_dataset, malformed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.assign(model="naive_zero"), "model"),
        (lambda frame: frame.assign(ticker="TSLA"), "ticker"),
        (lambda frame: frame.assign(split="holdout"), "split"),
    ],
)
def test_residual_data_rejects_invalid_prediction_identity(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
    mutation: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    """Reject invalid model ticker and split identities"""
    with pytest.raises(ValueError, match=message):
        build_residual_dataset(source_dataset, mutation(arima_predictions))


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [("actual_return", np.nan), ("predicted_return", np.inf)],
)
def test_residual_data_rejects_nonfinite_prediction_values(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
    column: str,
    invalid_value: float,
) -> None:
    """Reject missing and infinite forecast numbers"""
    malformed = arima_predictions.copy()
    malformed.loc[malformed.index[0], column] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        build_residual_dataset(source_dataset, malformed)


def test_residual_data_rejects_mismatched_actual_return(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Reject forecasts with inconsistent realized returns"""
    malformed = arima_predictions.copy()
    malformed.loc[malformed.index[0], "actual_return"] += 0.1

    with pytest.raises(ValueError, match="actual_return"):
        build_residual_dataset(source_dataset, malformed)


@pytest.mark.parametrize("frame_name", ["dataset", "predictions"])
@pytest.mark.parametrize("column", ["date", "target_date"])
def test_residual_data_rejects_non_datetime_keys(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
    frame_name: str,
    column: str,
) -> None:
    """Reject string dates in either input table"""
    malformed_dataset = source_dataset.copy()
    malformed_predictions = arima_predictions.copy()
    target = malformed_dataset if frame_name == "dataset" else malformed_predictions
    target[column] = target[column].astype(str)

    with pytest.raises(ValueError, match=column):
        build_residual_dataset(malformed_dataset, malformed_predictions)


def test_residual_data_rejects_date_gap_with_context(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Report ticker and dates for chronology gaps"""
    malformed_dataset = source_dataset.copy()
    malformed_predictions = arima_predictions.copy()
    row = malformed_predictions.loc[
        malformed_predictions["ticker"].eq("AAPL")
        & malformed_predictions["target_date"].eq("2019-12-27")
    ].index[0]
    malformed_predictions.loc[row, "target_date"] = pd.Timestamp("2019-12-28")
    source_row = malformed_dataset.loc[
        malformed_dataset["ticker"].eq("AAPL") & malformed_dataset["target_date"].eq("2019-12-27")
    ].index[0]
    malformed_dataset.loc[source_row, "target_date"] = pd.Timestamp("2019-12-28")

    with pytest.raises(ValueError, match="AAPL.*2019-12-28.*2019-12-27"):
        build_residual_dataset(malformed_dataset, malformed_predictions)


def test_residual_data_excludes_source_test_rows(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Ignore test rows from full source snapshots"""
    result = build_residual_dataset(source_dataset, arima_predictions)

    assert set(result["split"]) == {"train", "validation"}
    test_target_dates = set(source_dataset.loc[source_dataset["split"].eq("test"), "target_date"])
    assert not set(result["target_date"]).intersection(test_target_dates)


def test_residual_data_rejects_test_split_prediction(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Reject final-test forecasts before residual calculation"""
    test_row = source_dataset.loc[source_dataset["split"].eq("test")].iloc[[0]].copy()
    test_prediction = test_row.loc[:, ["date", "target_date", "ticker", "split"]]
    test_prediction["model"] = "arima"
    test_prediction["actual_return"] = test_row["target_return"]
    test_prediction["predicted_return"] = 0.0
    malformed = pd.concat(
        [arima_predictions, test_prediction.loc[:, PREDICTION_COLUMNS]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="test split"):
        build_residual_dataset(source_dataset, malformed)


def test_residual_data_rejects_test_target_date(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Reject disguised forecasts targeting final-test dates"""
    malformed = arima_predictions.copy()
    test_target = source_dataset.loc[source_dataset["split"].eq("test"), "target_date"].iloc[0]
    malformed.loc[malformed.index[0], "target_date"] = test_target

    with pytest.raises(ValueError, match="test target date"):
        build_residual_dataset(source_dataset, malformed)


def test_residual_data_requires_prediction_columns(
    source_dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
) -> None:
    """Reject prediction tables missing required columns"""
    malformed = arima_predictions.drop(columns="predicted_return")

    with pytest.raises(ValueError, match="missing columns"):
        build_residual_dataset(source_dataset, malformed)
