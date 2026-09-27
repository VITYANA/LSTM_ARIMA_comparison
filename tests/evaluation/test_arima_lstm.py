"""Tests for strict additive ARIMA-LSTM forecast composition."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from evaluation.arima_lstm import (
    HYBRID_MODEL_NAME,
    RESIDUAL_MODEL_NAME,
    build_arima_lstm_predictions,
)
from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.lstm_selection import select_lstm_candidates
from evaluation.lstm_validation import CORE_TICKERS, LSTMRunKey, LSTMRunResult
from evaluation.metrics import calculate_metrics
from models.lstm import LSTMConfig

ALIGNMENT_COLUMNS = ["date", "target_date", "ticker", "split"]
BASE_CONFIG = LSTMConfig(5, 16, 0.0)


@pytest.fixture
def component_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unsorted aligned ARIMA and residual forecasts"""
    arima = pd.DataFrame(
        [
            ["2020-01-03", "2020-01-06", "JPM", "validation", "arima", -0.01, -0.002],
            ["2020-01-02", "2020-01-03", "AAPL", "validation", "arima", 0.01, 0.004],
            ["2020-01-03", "2020-01-06", "AAPL", "validation", "arima", -0.02, -0.005],
            ["2020-01-02", "2020-01-03", "JPM", "validation", "arima", 0.03, 0.01],
        ],
        columns=PREDICTION_COLUMNS,
    )
    arima["date"] = pd.to_datetime(arima["date"])
    arima["target_date"] = pd.to_datetime(arima["target_date"])

    residual = pd.DataFrame(
        [
            [
                "2020-01-02",
                "2020-01-03",
                "JPM",
                "validation",
                RESIDUAL_MODEL_NAME,
                0.02,
                0.003,
            ],
            [
                "2020-01-03",
                "2020-01-06",
                "AAPL",
                "validation",
                RESIDUAL_MODEL_NAME,
                -0.015,
                -0.002,
            ],
            [
                "2020-01-02",
                "2020-01-03",
                "AAPL",
                "validation",
                RESIDUAL_MODEL_NAME,
                0.006,
                0.001,
            ],
            [
                "2020-01-03",
                "2020-01-06",
                "JPM",
                "validation",
                RESIDUAL_MODEL_NAME,
                -0.008,
                -0.001,
            ],
        ],
        columns=PREDICTION_COLUMNS,
    )
    residual["date"] = pd.to_datetime(residual["date"])
    residual["target_date"] = pd.to_datetime(residual["target_date"])
    return arima, residual


def test_build_arima_lstm_predictions_adds_aligned_components(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Add residual forecasts while preserving raw returns"""
    arima, residual = component_predictions
    original_arima = arima.copy(deep=True)
    original_residual = residual.copy(deep=True)

    result = build_arima_lstm_predictions(arima, residual)

    pd.testing.assert_frame_equal(arima, original_arima)
    pd.testing.assert_frame_equal(residual, original_residual)
    assert result.columns.tolist() == list(PREDICTION_COLUMNS)
    assert result[["ticker", "target_date"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2020-01-03")],
        ["AAPL", pd.Timestamp("2020-01-06")],
        ["JPM", pd.Timestamp("2020-01-03")],
        ["JPM", pd.Timestamp("2020-01-06")],
    ]
    assert result["model"].eq(HYBRID_MODEL_NAME).all()
    assert result["actual_return"].tolist() == pytest.approx([0.01, -0.02, 0.03, -0.01])
    assert result["predicted_return"].tolist() == pytest.approx([0.005, -0.007, 0.013, -0.003])
    assert result["actual_return"].dtype == np.dtype("float64")
    assert result["predicted_return"].dtype == np.dtype("float64")
    assert np.isfinite(result[["actual_return", "predicted_return"]].to_numpy()).all()


def test_build_arima_lstm_predictions_accepts_custom_model_name(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Use a valid caller-selected hybrid identifier"""
    arima, residual = component_predictions

    result = build_arima_lstm_predictions(arima, residual, model_name="hybrid_candidate")

    assert result["model"].eq("hybrid_candidate").all()


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "duplicate_arima",
        "duplicate_residual",
        "ticker",
        "date",
        "target_date",
        "split",
    ],
)
def test_build_arima_lstm_predictions_rejects_key_misalignment(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
    case: str,
) -> None:
    """Reject missing extra duplicate and mismatched keys"""
    arima, residual = (frame.copy(deep=True) for frame in component_predictions)
    if case == "missing":
        residual = residual.iloc[1:].copy()
    elif case == "extra":
        extra = residual.iloc[[0]].copy()
        extra["date"] = pd.Timestamp("2020-01-06")
        extra["target_date"] = pd.Timestamp("2020-01-07")
        residual = pd.concat([residual, extra], ignore_index=True)
    elif case == "duplicate_arima":
        arima = pd.concat([arima, arima.iloc[[0]]], ignore_index=True)
    elif case == "duplicate_residual":
        residual = pd.concat([residual, residual.iloc[[0]]], ignore_index=True)
    elif case == "ticker":
        residual.loc[residual.index[0], "ticker"] = "SPY"
    elif case == "date":
        residual.loc[residual.index[0], "date"] = pd.Timestamp("2030-01-01")
    elif case == "target_date":
        residual.loc[residual.index[0], "target_date"] = pd.Timestamp("2030-01-02")
    else:
        residual.loc[residual.index[0], "split"] = "train"

    with pytest.raises(ValueError, match="align|duplicate|validation"):
        build_arima_lstm_predictions(arima, residual)


@pytest.mark.parametrize("table_name", ["arima", "residual"])
@pytest.mark.parametrize("split", ["train", "test"])
def test_build_arima_lstm_predictions_rejects_nonvalidation_rows(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
    table_name: str,
    split: str,
) -> None:
    """Reject train and final-test component rows"""
    arima, residual = (frame.copy(deep=True) for frame in component_predictions)
    target = arima if table_name == "arima" else residual
    target["split"] = split

    with pytest.raises(ValueError, match="validation"):
        build_arima_lstm_predictions(arima, residual)


@pytest.mark.parametrize(
    ("table_name", "model"),
    [("arima", "wrong_arima"), ("residual", "lstm")],
)
def test_build_arima_lstm_predictions_rejects_component_names(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
    table_name: str,
    model: str,
) -> None:
    """Reject unexpected component model identifiers"""
    arima, residual = (frame.copy(deep=True) for frame in component_predictions)
    target = arima if table_name == "arima" else residual
    target["model"] = model

    with pytest.raises(ValueError, match="model"):
        build_arima_lstm_predictions(arima, residual)


@pytest.mark.parametrize("table_name", ["arima", "residual"])
@pytest.mark.parametrize("column", ["actual_return", "predicted_return"])
@pytest.mark.parametrize("invalid_value", ["invalid", np.nan, np.inf])
def test_build_arima_lstm_predictions_rejects_invalid_numbers(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
    table_name: str,
    column: str,
    invalid_value: str | float,
) -> None:
    """Reject nonnumeric and nonfinite component values"""
    arima, residual = (frame.copy(deep=True) for frame in component_predictions)
    target = arima if table_name == "arima" else residual
    if isinstance(invalid_value, str):
        target[column] = target[column].astype(object)
    target.loc[target.index[0], column] = invalid_value

    with pytest.raises(ValueError, match="numeric|finite"):
        build_arima_lstm_predictions(arima, residual)


def test_build_arima_lstm_predictions_accepts_round_trip_tolerance(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Accept residual identity within explicit absolute tolerance"""
    arima, residual = (frame.copy(deep=True) for frame in component_predictions)
    residual.loc[residual.index[0], "actual_return"] += 5e-13

    result = build_arima_lstm_predictions(arima, residual)

    assert len(result) == len(arima)


def test_build_arima_lstm_predictions_rejects_residual_identity_mismatch(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Reject residual identity beyond numerical tolerance"""
    arima, residual = (frame.copy(deep=True) for frame in component_predictions)
    residual.loc[residual.index[0], "actual_return"] += 2e-12

    with pytest.raises(ValueError, match="residual identity"):
        build_arima_lstm_predictions(arima, residual)


@pytest.mark.parametrize("model_name", ["", "   ", 1])
def test_build_arima_lstm_predictions_rejects_invalid_output_name(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
    model_name: object,
) -> None:
    """Reject blank and nonstring hybrid identifiers"""
    arima, residual = component_predictions

    with pytest.raises(ValueError, match="model name"):
        build_arima_lstm_predictions(
            arima,
            residual,
            model_name=cast(str, model_name),
        )


@pytest.mark.parametrize("table_name", ["arima", "residual"])
@pytest.mark.parametrize("case", ["columns", "empty", "date", "target_date"])
def test_build_arima_lstm_predictions_rejects_malformed_tables(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
    table_name: str,
    case: str,
) -> None:
    """Reject malformed component table structures"""
    arima, residual = (frame.copy(deep=True) for frame in component_predictions)
    target = arima if table_name == "arima" else residual
    if case == "columns":
        target.drop(columns="actual_return", inplace=True)
    elif case == "empty":
        target.drop(index=target.index, inplace=True)
    else:
        target[case] = target[case].astype(str)

    with pytest.raises(ValueError, match="columns|empty|datetime"):
        build_arima_lstm_predictions(arima, residual)


def test_build_arima_lstm_predictions_rejects_nonfinite_sum(
    component_predictions: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Reject overflow created while adding finite components"""
    arima, residual = (frame.copy(deep=True) for frame in component_predictions)
    arima["actual_return"] = 1.5e308
    arima["predicted_return"] = 1.0e308
    residual["actual_return"] = 5.0e307
    residual["predicted_return"] = 1.0e308

    with pytest.raises(ValueError, match="finite"):
        build_arima_lstm_predictions(arima, residual)


def raw_validation_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aligned raw baseline ARIMA and LSTM tables"""
    naive_frames = []
    arima_frames = []
    lstm_frames = []
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    target_dates = pd.to_datetime(["2020-01-03", "2020-01-06"])
    for ticker in CORE_TICKERS:
        base = pd.DataFrame(
            {
                "date": dates,
                "target_date": target_dates,
                "ticker": ticker,
                "split": "validation",
                "model": "naive_zero",
                "actual_return": [0.01, -0.02],
                "predicted_return": [0.0, 0.0],
            },
            columns=PREDICTION_COLUMNS,
        )
        naive_frames.append(base)
        arima = base.copy()
        arima["model"] = "arima"
        arima["predicted_return"] = [0.002, -0.003]
        arima_frames.append(arima)
        lstm = base.copy()
        lstm["model"] = "lstm"
        lstm["predicted_return"] = [0.003, -0.004]
        lstm_frames.append(lstm)
    return (
        pd.concat(naive_frames, ignore_index=True),
        pd.concat(arima_frames, ignore_index=True),
        pd.concat(lstm_frames, ignore_index=True),
    )


def residual_seed_results(
    arima_predictions: pd.DataFrame,
) -> tuple[list[LSTMRunResult], pd.DataFrame]:
    """Build ten successful residual seeds per ticker"""
    results: list[LSTMRunResult] = []
    baseline_frames = []
    for ticker in CORE_TICKERS:
        arima = arima_predictions.loc[arima_predictions["ticker"].eq(ticker)].reset_index(drop=True)
        actual_residual = arima["actual_return"] - arima["predicted_return"]
        baseline = arima.copy()
        baseline["model"] = "naive_zero"
        baseline["actual_return"] = actual_residual
        baseline["predicted_return"] = 0.0
        baseline_frames.append(baseline)
        for seed in range(10):
            predictions = baseline.copy()
            predictions["model"] = f"residual_seed_{seed:02d}"
            predictions["predicted_return"] = [
                0.001 + seed / 100_000,
                -0.001 + seed / 100_000,
            ]
            results.append(
                LSTMRunResult(
                    key=LSTMRunKey(ticker, BASE_CONFIG, seed),
                    status="ok",
                    best_epoch=seed + 1,
                    parameter_count=73,
                    mae_bps=10.0 + seed,
                    rmse_bps=20.0 + seed,
                    predictions=predictions,
                    error=None,
                )
            )
    return results, pd.concat(baseline_frames, ignore_index=True)


def test_arima_lstm_predictions_integrate_with_selection_and_metrics() -> None:
    """Compare four aligned models through existing metrics"""
    naive, arima, ordinary_lstm = raw_validation_tables()
    results, residual_baseline = residual_seed_results(arima)
    selection = select_lstm_candidates(
        results,
        residual_baseline,
        selected_model=RESIDUAL_MODEL_NAME,
    )

    hybrid = build_arima_lstm_predictions(arima, selection.predictions)
    combined = pd.concat([naive, arima, ordinary_lstm, hybrid], ignore_index=True)
    expected_keys = (
        naive.loc[:, ALIGNMENT_COLUMNS].sort_values(ALIGNMENT_COLUMNS).reset_index(drop=True)
    )
    for model in ("naive_zero", "arima", "lstm", HYBRID_MODEL_NAME):
        model_keys = (
            combined.loc[combined["model"].eq(model), ALIGNMENT_COLUMNS]
            .sort_values(ALIGNMENT_COLUMNS)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(model_keys, expected_keys)

    metrics = calculate_metrics(combined)

    assert set(metrics["model"]) == {"naive_zero", "arima", "lstm", HYBRID_MODEL_NAME}
    assert set(metrics["ticker"]) == {*CORE_TICKERS, "ALL"}
    assert metrics.groupby("model").size().to_dict() == {
        "arima": 5,
        HYBRID_MODEL_NAME: 5,
        "lstm": 5,
        "naive_zero": 5,
    }
