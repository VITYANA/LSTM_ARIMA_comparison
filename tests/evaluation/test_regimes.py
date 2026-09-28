"""Tests for leakage-safe final volatility regimes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.final_models import FINAL_TEST_PROTOCOL, FinalTestProtocol
from evaluation.regimes import (
    REGIME_ASSIGNMENT_COLUMNS,
    REGIME_THRESHOLD_COLUMNS,
    VolatilityRegimes,
    build_volatility_regimes,
    calculate_regime_return_metrics,
)

CORE_TICKERS = ("AAPL", "JPM", "SPY", "XOM")
MODELS = ("naive_zero", "arima", "lstm", "arima_lstm")


def regime_protocol(window: int = 3) -> FinalTestProtocol:
    """Return protocol with a compact volatility window"""
    return replace(FINAL_TEST_PROTOCOL, volatility_window=window, annualization_days=4)


def regime_dataset() -> pd.DataFrame:
    """Return chronological histories spanning every final split"""
    schedule = (
        ("2015-12-24", "2015-12-28", "train"),
        ("2015-12-28", "2015-12-29", "train"),
        ("2015-12-29", "2015-12-30", "train"),
        ("2015-12-30", "2015-12-31", "train"),
        ("2015-12-31", "2016-01-04", "validation"),
        ("2019-12-27", "2019-12-30", "validation"),
        ("2019-12-30", "2019-12-31", "validation"),
        ("2019-12-31", "2020-01-02", "test"),
        ("2025-12-30", "2025-12-31", "test"),
    )
    values = np.array([0.00, 0.01, 0.03, 0.02, 0.05, 0.01, 0.04, 0.02, 0.08])
    rows: list[dict[str, object]] = []
    for ticker_position, ticker in enumerate(CORE_TICKERS, start=1):
        for position, (date, target_date, split) in enumerate(schedule):
            value = float(values[position] * ticker_position)
            rows.append(
                {
                    "date": date,
                    "target_date": target_date,
                    "ticker": ticker,
                    "adj_close": 100.0 + position,
                    "log_return": value,
                    "target_return": value + 0.001,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def regime_predictions() -> pd.DataFrame:
    """Return aligned forecasts with three regimes per ticker"""
    target_dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2025-12-31"])
    dates = pd.to_datetime(["2019-12-31", "2020-01-02", "2025-12-30"])
    actual = np.array([0.01, -0.02, 0.03])
    frames: list[pd.DataFrame] = []
    factors = {"naive_zero": 0.0, "arima": 0.5, "lstm": 0.75, "arima_lstm": 1.0}
    for model, factor in factors.items():
        for ticker in CORE_TICKERS:
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


def regime_assignments() -> pd.DataFrame:
    """Return one deterministic regime for every observation"""
    baseline = regime_predictions().loc[lambda frame: frame["model"].eq("naive_zero")]
    assignments = baseline.loc[:, ["date", "target_date", "ticker", "split"]].copy()
    assignments["volatility"] = np.tile([0.1, 0.2, 0.3], len(CORE_TICKERS))
    assignments["regime"] = np.tile(["low", "medium", "high"], len(CORE_TICKERS))
    return assignments.loc[:, REGIME_ASSIGNMENT_COLUMNS].reset_index(drop=True)


def test_build_volatility_regimes_uses_training_terciles_and_origin_history() -> None:
    """Fit thresholds before test and classify origin volatility"""
    dataset = regime_dataset()
    protocol = regime_protocol()

    result = build_volatility_regimes(dataset, protocol)

    assert isinstance(result, VolatilityRegimes)
    assert result.thresholds.columns.tolist() == list(REGIME_THRESHOLD_COLUMNS)
    assert result.assignments.columns.tolist() == list(REGIME_ASSIGNMENT_COLUMNS)
    aapl = dataset.loc[dataset["ticker"].eq("AAPL")].copy()
    rolling = aapl["log_return"].rolling(3).std(ddof=1) * np.sqrt(4)
    training_volatility = rolling[aapl["split"].isin(["train", "validation"])].dropna()
    threshold = result.thresholds.loc[result.thresholds["ticker"].eq("AAPL")].iloc[0]
    assert threshold["lower_threshold"] == pytest.approx(training_volatility.quantile(1 / 3))
    assert threshold["upper_threshold"] == pytest.approx(training_volatility.quantile(2 / 3))
    first_test = result.assignments.loc[result.assignments["ticker"].eq("AAPL")].iloc[0]
    assert first_test["volatility"] == pytest.approx(rolling.iloc[-2])


def test_future_return_does_not_change_earlier_regime_assignment() -> None:
    """Prevent future returns from changing earlier regimes"""
    dataset = regime_dataset()
    baseline = build_volatility_regimes(dataset, regime_protocol()).assignments
    changed = dataset.copy(deep=True)
    future = changed["ticker"].eq("AAPL") & changed["target_date"].eq("2025-12-31")
    changed.loc[future, "log_return"] = 100.0

    updated = build_volatility_regimes(changed, regime_protocol()).assignments

    first_target = pd.Timestamp("2020-01-02")
    baseline_first = baseline.loc[
        baseline["ticker"].eq("AAPL") & baseline["target_date"].eq(first_target)
    ].iloc[0]
    updated_first = updated.loc[
        updated["ticker"].eq("AAPL") & updated["target_date"].eq(first_target)
    ].iloc[0]
    pd.testing.assert_series_equal(baseline_first, updated_first)


def test_coincident_thresholds_assign_ties_to_low() -> None:
    """Resolve coincident threshold ties deterministically"""
    dataset = regime_dataset()
    for ticker in CORE_TICKERS:
        positions = dataset.index[dataset["ticker"].eq(ticker)]
        dataset.loc[positions, "log_return"] = np.arange(len(positions), dtype="float64")

    result = build_volatility_regimes(dataset, regime_protocol(window=2))

    assert np.allclose(
        result.thresholds["lower_threshold"],
        result.thresholds["upper_threshold"],
    )
    first_test = result.assignments.groupby("ticker", sort=True).head(1)
    assert first_test["regime"].eq("low").all()


def test_build_volatility_regimes_rejects_insufficient_history() -> None:
    """Reject histories shorter than volatility window"""
    with pytest.raises(ValueError, match="history"):
        build_volatility_regimes(regime_dataset(), regime_protocol(window=20))


def test_calculate_regime_return_metrics_groups_ticker_and_pooled_rows() -> None:
    """Calculate aligned return metrics within each regime"""
    result = calculate_regime_return_metrics(
        regime_predictions(),
        regime_assignments(),
    )

    assert set(result["regime"]) == {"low", "medium", "high"}
    low_aapl = result.loc[
        result["regime"].eq("low") & result["ticker"].eq("AAPL") & result["model"].eq("arima")
    ].iloc[0]
    assert low_aapl["mae_bps"] == pytest.approx(50.0)
    assert low_aapl["rel_mae"] == pytest.approx(0.5)
    assert not result.loc[result["ticker"].eq("ALL")].empty


def test_calculate_regime_return_metrics_allows_unobserved_regimes() -> None:
    """Skip regimes absent from final observations"""
    assignments = regime_assignments()
    assignments["regime"] = "low"

    result = calculate_regime_return_metrics(regime_predictions(), assignments)

    assert set(result["regime"]) == {"low"}


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "empty",
        "date-type",
        "missing-date",
        "split",
        "nonfinite-volatility",
        "missing",
        "duplicate",
        "regime",
    ],
)
def test_calculate_regime_return_metrics_rejects_invalid_assignments(case: str) -> None:
    """Reject incomplete duplicate or unknown regime assignments"""
    assignments = regime_assignments()
    if case == "schema":
        assignments = assignments.drop(columns="volatility")
    elif case == "empty":
        assignments = assignments.iloc[0:0]
    elif case == "date-type":
        assignments["date"] = assignments["date"].astype(str)
    elif case == "missing-date":
        assignments.loc[0, "date"] = pd.NaT
    elif case == "split":
        assignments["split"] = "validation"
    elif case == "nonfinite-volatility":
        assignments.loc[0, "volatility"] = np.inf
    elif case == "missing":
        assignments = assignments.iloc[1:].reset_index(drop=True)
    elif case == "duplicate":
        assignments = pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True)
    else:
        assignments.loc[0, "regime"] = "extreme"

    with pytest.raises(ValueError):
        calculate_regime_return_metrics(regime_predictions(), assignments)
