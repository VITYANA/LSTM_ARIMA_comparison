"""Tests for LSTM seed aggregation and ticker selection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.lstm_selection import LSTMSelection, select_lstm_candidates
from evaluation.lstm_validation import CORE_TICKERS, LSTMRunKey, LSTMRunResult
from evaluation.metrics import calculate_metrics
from models.lstm import LSTMConfig

RUN_METRIC_COLUMNS = [
    "ticker",
    "window",
    "units",
    "dropout",
    "seed",
    "status",
    "best_epoch",
    "parameter_count",
    "mae_bps",
    "rmse_bps",
    "error",
]
CONFIGURATION_METRIC_COLUMNS = [
    "ticker",
    "window",
    "units",
    "dropout",
    "successful_seeds",
    "mean_mae_bps",
    "mean_rmse_bps",
    "std_mae_bps",
    "parameter_count",
]
CONFIGURATION_COLUMNS = [
    "ticker",
    "window",
    "units",
    "dropout",
    "successful_seeds",
    "mean_mae_bps",
    "mean_rmse_bps",
    "std_mae_bps",
    "parameter_count",
    "model",
]
BASE_CONFIG = LSTMConfig(5, 16, 0.0)
SECOND_CONFIG = LSTMConfig(21, 32, 0.2)


def prediction_rows(
    ticker: str,
    config: LSTMConfig,
    seed: int,
    predicted: list[float] | None = None,
) -> pd.DataFrame:
    """Build aligned predictions for one successful seed"""
    values = predicted or [seed / 1_000, seed / 1_000]
    model = (
        f"lstm_w{config.window}_u{config.units}_d{int(config.dropout * 100):02d}"
        f"_seed_{seed:02d}"
    )
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "target_date": pd.to_datetime(["2020-01-03", "2020-01-06"]),
            "ticker": ticker,
            "split": "validation",
            "model": model,
            "actual_return": [0.01, -0.02],
            "predicted_return": values,
        },
        columns=PREDICTION_COLUMNS,
    )


def successful_result(
    ticker: str,
    config: LSTMConfig,
    seed: int,
    *,
    mae_bps: float,
    rmse_bps: float,
    parameter_count: int,
) -> LSTMRunResult:
    """Build one successful terminal result"""
    return LSTMRunResult(
        key=LSTMRunKey(ticker, config, seed),
        status="ok",
        best_epoch=seed + 1,
        parameter_count=parameter_count,
        mae_bps=mae_bps,
        rmse_bps=rmse_bps,
        predictions=prediction_rows(ticker, config, seed),
        error=None,
    )


def config_results(
    ticker: str,
    config: LSTMConfig,
    *,
    mae_start: float,
    rmse_start: float,
    parameter_count: int,
) -> list[LSTMRunResult]:
    """Build ten successful results with explicit metrics"""
    return [
        successful_result(
            ticker,
            config,
            seed,
            mae_bps=mae_start + seed,
            rmse_bps=rmse_start + seed,
            parameter_count=parameter_count,
        )
        for seed in range(10)
    ]


def complete_results() -> list[LSTMRunResult]:
    """Build complete candidates with different AAPL quality"""
    results = config_results(
        "AAPL",
        BASE_CONFIG,
        mae_start=10.0,
        rmse_start=20.0,
        parameter_count=100,
    )
    results += config_results(
        "AAPL",
        SECOND_CONFIG,
        mae_start=30.0,
        rmse_start=40.0,
        parameter_count=200,
    )
    for position, ticker in enumerate(CORE_TICKERS[1:], start=1):
        results += config_results(
            ticker,
            BASE_CONFIG,
            mae_start=15.0 + position,
            rmse_start=25.0 + position,
            parameter_count=100,
        )
    return results


def baseline_predictions() -> pd.DataFrame:
    """Build aligned naive-zero validation predictions"""
    frames = []
    for ticker in CORE_TICKERS:
        frame = prediction_rows(ticker, BASE_CONFIG, 0, [0.0, 0.0])
        frame["model"] = "naive_zero"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_select_lstm_candidates_aggregates_and_selects_per_ticker() -> None:
    """Aggregate seed metrics and select independently per ticker"""
    results = complete_results()
    baseline = baseline_predictions()
    original_baseline = baseline.copy(deep=True)

    selection = select_lstm_candidates(results, baseline)

    assert isinstance(selection, LSTMSelection)
    pd.testing.assert_frame_equal(baseline, original_baseline)
    assert selection.run_metrics.columns.tolist() == RUN_METRIC_COLUMNS
    assert selection.configuration_metrics.columns.tolist() == CONFIGURATION_METRIC_COLUMNS
    assert selection.configurations.columns.tolist() == CONFIGURATION_COLUMNS
    aapl = selection.configuration_metrics.loc[
        selection.configuration_metrics["ticker"].eq("AAPL")
        & selection.configuration_metrics["window"].eq(5)
    ].iloc[0]
    expected = {
        "successful_seeds": 10,
        "mean_mae_bps": np.mean(np.arange(10.0, 20.0)),
        "mean_rmse_bps": np.mean(np.arange(20.0, 30.0)),
        "std_mae_bps": np.std(np.arange(10.0, 20.0), ddof=1),
        "parameter_count": 100,
    }
    for name, value in expected.items():
        assert aapl[name] == pytest.approx(value)
    assert selection.configurations[["ticker", "window"]].values.tolist() == [
        ["AAPL", 5],
        ["JPM", 5],
        ["SPY", 5],
        ["XOM", 5],
    ]


@pytest.mark.parametrize(
    ("case", "base_parameters", "second_parameters", "expected_window"),
    [
        ("mean_mae", 50, 100, 21),
        ("mean_rmse", 50, 100, 21),
        ("std_mae", 50, 100, 21),
        ("parameters", 50, 100, 5),
        ("window", 50, 50, 5),
    ],
)
def test_select_lstm_candidates_applies_deterministic_tie_breaks(
    case: str,
    base_parameters: int,
    second_parameters: int,
    expected_window: int,
) -> None:
    """Apply declared configuration ranking priorities"""
    results = complete_results()
    updated: list[LSTMRunResult] = []
    for result in results:
        if result.key.ticker != "AAPL":
            updated.append(result)
            continue
        parameter_count = base_parameters if result.key.config == BASE_CONFIG else second_parameters
        mae = 10.0
        rmse = 20.0
        if case == "mean_mae" and result.key.config == BASE_CONFIG:
            mae = 11.0
        elif case == "mean_rmse" and result.key.config == BASE_CONFIG:
            rmse = 21.0
        elif case == "std_mae" and result.key.config == BASE_CONFIG:
            mae += -1.0 if result.key.seed % 2 == 0 else 1.0
        updated.append(
            replace(
                result,
                mae_bps=mae,
                rmse_bps=rmse,
                parameter_count=parameter_count,
            )
        )

    selection = select_lstm_candidates(updated, baseline_predictions())

    selected = selection.configurations.loc[
        selection.configurations["ticker"].eq("AAPL"), "window"
    ].item()
    assert selected == expected_window


@pytest.mark.parametrize("case", ["missing", "failed"])
def test_select_lstm_candidates_excludes_incomplete_configurations(case: str) -> None:
    """Exclude candidates lacking ten successful unique seeds"""
    results = complete_results()
    target = next(
        result
        for result in results
        if result.key.ticker == "AAPL" and result.key.config == BASE_CONFIG and result.key.seed == 9
    )
    results.remove(target)
    if case == "failed":
        results.append(
            replace(
                target,
                status="failed",
                best_epoch=None,
                parameter_count=None,
                mae_bps=float("nan"),
                rmse_bps=float("nan"),
                predictions=pd.DataFrame(columns=PREDICTION_COLUMNS),
                error="failed",
            )
        )

    selection = select_lstm_candidates(results, baseline_predictions())

    selected = selection.configurations.loc[
        selection.configurations["ticker"].eq("AAPL"), "window"
    ].item()
    assert selected == 21


def test_select_lstm_candidates_rejects_duplicate_seed_results() -> None:
    """Reject duplicate ticker configuration and seed keys"""
    results = complete_results()
    results.append(results[0])

    with pytest.raises(ValueError, match="duplicate"):
        select_lstm_candidates(results, baseline_predictions())


def test_select_lstm_candidates_rejects_inconsistent_parameter_counts() -> None:
    """Reject one architecture reporting different parameter counts"""
    results = complete_results()
    results[0] = replace(results[0], parameter_count=999)

    with pytest.raises(ValueError, match="parameter"):
        select_lstm_candidates(results, baseline_predictions())


def test_select_lstm_candidates_requires_valid_config_for_every_ticker() -> None:
    """Raise when one ticker has no complete candidate"""
    results = [result for result in complete_results() if result.key.ticker != "XOM"]

    with pytest.raises(ValueError, match="XOM"):
        select_lstm_candidates(results, baseline_predictions())


def test_select_lstm_candidates_averages_selected_seed_predictions() -> None:
    """Average ten selected seeds into aligned forecasts"""
    selection = select_lstm_candidates(complete_results(), baseline_predictions())

    assert selection.predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert selection.predictions["model"].eq("lstm").all()
    assert selection.predictions["split"].eq("validation").all()
    assert selection.predictions.groupby("ticker").size().to_dict() == {
        ticker: 2 for ticker in CORE_TICKERS
    }
    assert selection.predictions["predicted_return"].tolist() == pytest.approx([0.0045] * 8)
    metrics = calculate_metrics(
        pd.concat([baseline_predictions(), selection.predictions], ignore_index=True)
    )
    assert set(metrics["model"]) == {"naive_zero", "lstm"}


@pytest.mark.parametrize(
    "case",
    ["missing_row", "target_date", "actual", "duplicate", "split", "baseline_missing"],
)
def test_select_lstm_candidates_rejects_misaligned_predictions(case: str) -> None:
    """Reject seed or baseline observation misalignment"""
    results = complete_results()
    baseline = baseline_predictions()
    target_index = next(
        index
        for index, result in enumerate(results)
        if result.key.ticker == "AAPL" and result.key.config == BASE_CONFIG
    )
    target = results[target_index]
    predictions = target.predictions.copy()
    if case == "missing_row":
        predictions = predictions.iloc[:1]
    elif case == "target_date":
        predictions.loc[0, "target_date"] = pd.Timestamp("2030-01-01")
    elif case == "actual":
        predictions.loc[0, "actual_return"] = 99.0
    elif case == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif case == "split":
        predictions["split"] = "train"
    else:
        baseline = baseline.iloc[1:].copy()
    if case != "baseline_missing":
        results[target_index] = replace(target, predictions=predictions)

    with pytest.raises(ValueError, match="align|duplicate|validation|keys"):
        select_lstm_candidates(results, baseline)


@pytest.mark.parametrize("case", ["columns", "empty", "ticker"])
def test_select_lstm_candidates_rejects_invalid_seed_tables(case: str) -> None:
    """Reject malformed successful seed prediction tables"""
    results = complete_results()
    target = results[0]
    predictions = target.predictions.copy()
    if case == "columns":
        predictions = predictions.drop(columns="actual_return")
    elif case == "empty":
        predictions = pd.DataFrame(columns=PREDICTION_COLUMNS)
    else:
        predictions["ticker"] = "JPM"
    results[0] = replace(target, predictions=predictions)

    with pytest.raises(ValueError, match="standard|empty|ticker"):
        select_lstm_candidates(results, baseline_predictions())


@pytest.mark.parametrize(
    "case",
    ["columns", "empty", "split", "model", "tickers", "duplicate"],
)
def test_select_lstm_candidates_rejects_invalid_baseline(case: str) -> None:
    """Reject malformed or incomplete baseline prediction tables"""
    baseline = baseline_predictions()
    if case == "columns":
        baseline = baseline.drop(columns="actual_return")
    elif case == "empty":
        baseline = pd.DataFrame(columns=PREDICTION_COLUMNS)
    elif case == "split":
        baseline["split"] = "train"
    elif case == "model":
        baseline["model"] = "wrong"
    elif case == "tickers":
        baseline = baseline.loc[baseline["ticker"].ne("XOM")]
    else:
        baseline = pd.concat([baseline, baseline.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="baseline|validation|ticker|duplicate"):
        select_lstm_candidates(complete_results(), baseline)


@pytest.mark.parametrize("results", [[], [object()]])
def test_select_lstm_candidates_rejects_invalid_result_collection(
    results: list[object],
) -> None:
    """Reject empty and incorrectly typed run collections"""
    with pytest.raises(ValueError, match="results"):
        select_lstm_candidates(
            results,  # type: ignore[arg-type]
            baseline_predictions(),
        )


@pytest.mark.parametrize("selected_model", ["", "   ", 1])
def test_select_lstm_candidates_rejects_invalid_selected_name(
    selected_model: object,
) -> None:
    """Reject blank and nonstring ensemble identifiers"""
    with pytest.raises(ValueError, match="model name"):
        select_lstm_candidates(
            complete_results(),
            baseline_predictions(),
            selected_model=selected_model,  # type: ignore[arg-type]
        )
