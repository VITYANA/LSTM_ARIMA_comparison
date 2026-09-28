"""Tests for final forecast-comparison statistical inference."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.final_models import FINAL_TEST_PROTOCOL, FinalTestProtocol
from evaluation.statistics import (
    BOOTSTRAP_COLUMNS,
    DM_COLUMNS,
    MODEL_COMPARISONS,
    _bootstrap_metric_differences,
    _dm_statistic_p_value,
    _holm_adjust,
    _moving_block_indexes,
    _newey_west_long_run_variance,
    calculate_bootstrap_intervals,
    calculate_dm_results,
)

CORE_TICKERS = ("AAPL", "JPM", "SPY", "XOM")


def inference_protocol(
    *,
    lag: int = 2,
    block: int = 3,
    repetitions: int = 100,
    seed: int = 7,
) -> FinalTestProtocol:
    """Return a compact valid inference protocol"""
    return replace(
        FINAL_TEST_PROTOCOL,
        dm_lag=lag,
        bootstrap_block=block,
        bootstrap_repetitions=repetitions,
        bootstrap_seed=seed,
        bootstrap_confidence=0.8,
    )


def final_predictions(observations: int = 25) -> pd.DataFrame:
    """Return aligned four-model forecasts on a shared calendar"""
    target_dates = pd.DatetimeIndex(
        [
            *pd.bdate_range("2020-01-02", periods=observations - 1),
            pd.Timestamp("2025-12-31"),
        ]
    )
    dates = target_dates - pd.offsets.BDay(1)
    frames: list[pd.DataFrame] = []
    factors = {"naive_zero": 0.0, "arima": 0.5, "lstm": 0.75, "arima_lstm": 0.9}
    base_actual = np.linspace(-0.03, 0.03, observations)
    for model, factor in factors.items():
        for ticker_position, ticker in enumerate(CORE_TICKERS, start=1):
            actual = base_actual * ticker_position
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "target_date": target_dates,
                        "ticker": ticker,
                        "split": "test",
                        "model": model,
                        "actual_return": actual,
                        "predicted_return": actual * factor,
                    }
                ).loc[:, PREDICTION_COLUMNS]
            )
    return pd.concat(frames, ignore_index=True)


def test_model_comparisons_alias_frozen_protocol_order() -> None:
    """Keep one authority for approved comparison order"""
    assert MODEL_COMPARISONS is FINAL_TEST_PROTOCOL.model_comparisons


def test_newey_west_dm_matches_independent_bartlett_calculation() -> None:
    """Match capped Bartlett variance and normal probability"""
    differential = np.array([1.0, -1.0, 2.0, -2.0, 1.0])
    centered = differential - differential.mean()
    lag = len(differential) - 1
    expected_variance = float(np.dot(centered, centered) / len(centered))
    for position in range(1, lag + 1):
        covariance = float(np.dot(centered[position:], centered[:-position]) / len(centered))
        expected_variance += 2.0 * (1.0 - position / (lag + 1)) * covariance

    variance = _newey_west_long_run_variance(differential, requested_lag=20)
    statistic, probability = _dm_statistic_p_value(
        float(differential.mean()),
        variance,
        len(differential),
    )
    expected_statistic = differential.mean() / math.sqrt(expected_variance / len(differential))

    assert variance == pytest.approx(expected_variance)
    assert statistic == pytest.approx(expected_statistic)
    assert probability == pytest.approx(math.erfc(abs(expected_statistic) / math.sqrt(2.0)))


@pytest.mark.parametrize(
    ("mean", "variance", "statistic", "probability"),
    [
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, np.inf, 0.0),
        (-1.0, 0.0, -np.inf, 0.0),
    ],
)
def test_dm_handles_zero_long_run_variance(
    mean: float,
    variance: float,
    statistic: float,
    probability: float,
) -> None:
    """Handle identical and deterministic loss differences"""
    assert _dm_statistic_p_value(mean, variance, 5) == (statistic, probability)


def test_dm_rejects_materially_negative_long_run_variance() -> None:
    """Reject invalid negative Newey West variance"""
    with pytest.raises(ValueError, match="variance"):
        _dm_statistic_p_value(0.1, -0.01, 5)


def test_holm_adjustment_is_monotone_in_original_order() -> None:
    """Match known Holm adjusted probabilities"""
    raw = np.array([0.01, 0.04, 0.03, 0.20, 0.50])

    adjusted = _holm_adjust(raw)

    assert adjusted == pytest.approx([0.05, 0.12, 0.12, 0.40, 0.50])


def test_calculate_dm_results_uses_date_level_pooled_differences() -> None:
    """Average tickers by date before pooled Newey West"""
    predictions = final_predictions()
    protocol = inference_protocol()

    result = calculate_dm_results(predictions, protocol)

    assert result.columns.tolist() == list(DM_COLUMNS)
    assert len(result) == 5 * 5 * 2
    aapl = result.loc[
        result["ticker"].eq("AAPL")
        & result["loss"].eq("absolute")
        & result["candidate"].eq("arima")
        & result["comparator"].eq("naive_zero")
    ].iloc[0]
    actual = np.linspace(-0.03, 0.03, 25)
    expected = np.abs(actual * 0.5) - np.abs(actual)
    assert aapl["mean_loss_difference"] == pytest.approx(expected.mean())

    pooled = result.loc[
        result["ticker"].eq("ALL")
        & result["loss"].eq("absolute")
        & result["candidate"].eq("arima")
        & result["comparator"].eq("naive_zero")
    ].iloc[0]
    date_differences = np.column_stack(
        [np.abs(actual * ticker * 0.5) - np.abs(actual * ticker) for ticker in range(1, 5)]
    ).mean(axis=1)
    variance = _newey_west_long_run_variance(date_differences, protocol.dm_lag)
    statistic, probability = _dm_statistic_p_value(
        float(date_differences.mean()),
        variance,
        len(date_differences),
    )
    assert pooled["observations"] == 25
    assert pooled["dm_statistic"] == pytest.approx(statistic)
    assert pooled["raw_p_value"] == pytest.approx(probability)

    for (_, _), family in result.groupby(["ticker", "loss"], sort=False):
        assert len(family) == 5
        assert family["holm_p_value"].to_numpy() == pytest.approx(
            _holm_adjust(family["raw_p_value"].to_numpy(dtype="float64"))
        )


def test_moving_block_indexes_are_consecutive_and_trimmed() -> None:
    """Build consecutive blocks trimmed to sample length"""
    indexes = _moving_block_indexes(
        observations=8,
        block_length=3,
        rng=np.random.default_rng(11),
    )

    assert len(indexes) == 8
    assert np.diff(indexes[:3]).tolist() == [1, 1]
    assert np.diff(indexes[3:6]).tolist() == [1, 1]
    assert np.diff(indexes[6:]).tolist() == [1]


def test_bootstrap_metric_differences_resample_common_date_rows() -> None:
    """Apply identical sampled dates across ticker columns"""
    candidate = np.arange(1, 25, dtype="float64").reshape(6, 4) / 100
    comparator = candidate + 0.01
    rng = np.random.default_rng(3)

    mae, rmse = _bootstrap_metric_differences(
        candidate,
        comparator,
        repetitions=4,
        block_length=3,
        rng=rng,
    )

    manual_rng = np.random.default_rng(3)
    expected_mae: list[float] = []
    expected_rmse: list[float] = []
    for _ in range(4):
        indexes = _moving_block_indexes(6, 3, manual_rng)
        candidate_sample = candidate[indexes].reshape(-1)
        comparator_sample = comparator[indexes].reshape(-1)
        expected_mae.append(
            float(np.mean(np.abs(candidate_sample)) - np.mean(np.abs(comparator_sample)))
        )
        expected_rmse.append(
            float(
                np.sqrt(np.mean(np.square(candidate_sample)))
                - np.sqrt(np.mean(np.square(comparator_sample)))
            )
        )
    assert mae == pytest.approx(expected_mae)
    assert rmse == pytest.approx(expected_rmse)


def test_calculate_bootstrap_intervals_is_reproducible() -> None:
    """Return fixed seed estimates and percentile intervals"""
    predictions = final_predictions()
    protocol = inference_protocol(repetitions=50)

    first = calculate_bootstrap_intervals(predictions, protocol)
    second = calculate_bootstrap_intervals(predictions, protocol)

    assert first.columns.tolist() == list(BOOTSTRAP_COLUMNS)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert len(first) == 5 * 5 * 2
    row = first.loc[
        first["ticker"].eq("AAPL")
        & first["metric"].eq("mae")
        & first["candidate"].eq("arima")
        & first["comparator"].eq("naive_zero")
    ].iloc[0]
    actual = np.linspace(-0.03, 0.03, 25)
    assert row["estimate"] == pytest.approx(np.mean(np.abs(actual * 0.5)) - np.mean(np.abs(actual)))
    assert row["repetitions"] == 50
    assert row["block_length"] == 3
    assert row["seed"] == 7


@pytest.mark.parametrize(
    "case",
    ["calendar", "model", "nonfinite", "short", "protocol", "repetitions", "seed"],
)
def test_statistics_reject_invalid_inputs(case: str) -> None:
    """Reject malformed forecasts calendars and inference parameters"""
    predictions = final_predictions()
    protocol: object = inference_protocol()
    if case == "calendar":
        interior_date = predictions["target_date"].drop_duplicates().iloc[5]
        selected = predictions["ticker"].eq("XOM") & predictions["target_date"].eq(interior_date)
        predictions.loc[selected, "date"] -= pd.Timedelta(days=1)
    elif case == "model":
        predictions.loc[predictions["model"].eq("arima"), "model"] = "wrong"
    elif case == "nonfinite":
        predictions.loc[0, "predicted_return"] = np.inf
    elif case == "short":
        protocol = inference_protocol(lag=29, block=30)
    elif case == "protocol":
        protocol = object()
    elif case == "repetitions":
        object.__setattr__(protocol, "bootstrap_repetitions", 0)
    elif case == "seed":
        object.__setattr__(protocol, "bootstrap_seed", -1)

    if case not in {"short", "repetitions", "seed"}:
        with pytest.raises(ValueError):
            calculate_dm_results(predictions, protocol)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        calculate_bootstrap_intervals(predictions, protocol)  # type: ignore[arg-type]
