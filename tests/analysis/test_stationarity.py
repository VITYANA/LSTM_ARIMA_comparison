"""Tests for time-series stationarity diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import (  # type: ignore[import-untyped]
    InterpolationWarning,
)

import analysis.stationarity as stationarity_module
from analysis.stationarity import (
    StationarityResult,
    acf_pacf_table,
    build_stationarity_report,
    run_adf,
    run_kpss,
)


def stationary_series(size: int = 400) -> pd.Series:
    """Return a deterministic stationary AR process"""
    rng = np.random.default_rng(42)
    values = np.zeros(size)
    noise = rng.normal(size=size)
    for index in range(1, size):
        values[index] = 0.35 * values[index - 1] + noise[index]
    return pd.Series(values)


def random_walk_series(size: int = 400) -> pd.Series:
    """Return a deterministic nonstationary random walk"""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(size=size).cumsum())


def test_acf_pacf_table_returns_aligned_requested_lags() -> None:
    """Return aligned ACF and PACF coefficients"""
    result = acf_pacf_table(stationary_series(), nlags=10)

    assert result.columns.tolist() == ["lag", "acf", "pacf"]
    assert result["lag"].tolist() == list(range(11))
    assert result.loc[0, "acf"] == pytest.approx(1.0)
    assert result.loc[0, "pacf"] == pytest.approx(1.0)
    assert result.loc[1, "acf"] == pytest.approx(result.loc[1, "pacf"])


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(dtype=float),
        pd.Series(np.arange(29, dtype=float)),
        pd.Series([1.0] * 40),
        pd.Series([1.0] * 29 + [np.inf]),
    ],
    ids=["empty", "short", "constant", "infinite"],
)
def test_acf_pacf_table_rejects_invalid_series(series: pd.Series) -> None:
    """Reject empty short constant and infinite series"""
    with pytest.raises(ValueError):
        acf_pacf_table(series, nlags=5)


@pytest.mark.parametrize("nlags", [0, -1, 50])
def test_acf_pacf_table_rejects_invalid_lags(nlags: int) -> None:
    """Reject nonpositive or excessive lag counts"""
    with pytest.raises(ValueError, match="nlags"):
        acf_pacf_table(stationary_series(100), nlags=nlags)


def test_acf_pacf_table_does_not_mutate_source() -> None:
    """Preserve source values and index ordering"""
    series = stationary_series()
    original = series.copy(deep=True)

    acf_pacf_table(series, nlags=10)

    pd.testing.assert_series_equal(series, original)


@pytest.mark.parametrize(
    ("series", "expected_stationary"),
    [
        (stationary_series(), True),
        (random_walk_series(), False),
    ],
    ids=["stationary", "random-walk"],
)
def test_run_adf_identifies_unit_root_status(
    series: pd.Series,
    expected_stationary: bool,
) -> None:
    """Classify stationary and unit-root series with ADF"""
    result = run_adf(series)

    assert result.test == "ADF"
    assert result.stationary_at_5pct is expected_stationary
    assert (result.p_value < 0.05) is expected_stationary
    assert result.observations > 0
    assert result.critical_1pct < result.critical_5pct < result.critical_10pct


@pytest.mark.parametrize(
    ("series", "expected_stationary"),
    [
        (stationary_series(), True),
        (random_walk_series(), False),
    ],
    ids=["stationary", "random-walk"],
)
def test_run_kpss_identifies_stationarity_status(
    series: pd.Series,
    expected_stationary: bool,
) -> None:
    """Classify stationary and nonstationary series with KPSS"""
    with pytest.warns(InterpolationWarning):
        result = run_kpss(series)

    assert result.test == "KPSS"
    assert result.stationary_at_5pct is expected_stationary
    assert (result.p_value >= 0.05) is expected_stationary
    assert result.observations == len(series)
    assert result.critical_1pct > result.critical_5pct > result.critical_10pct


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(dtype=float),
        pd.Series(np.arange(29, dtype=float)),
        pd.Series([1.0] * 40),
        pd.Series([1.0] * 29 + [np.inf]),
    ],
    ids=["empty", "short", "constant", "infinite"],
)
def test_stationarity_tests_reject_invalid_series(series: pd.Series) -> None:
    """Reject unsuitable series before statistical tests"""
    with pytest.raises(ValueError):
        run_adf(series)
    with pytest.raises(ValueError):
        run_kpss(series)


def test_build_stationarity_report_runs_both_tests_by_ticker_and_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build ordered diagnostics with protocol regressions"""
    frame = pd.DataFrame(
        {
            "ticker": ["BBB"] * 30 + ["AAA"] * 30,
            "log_price": np.arange(60, dtype=float),
            "log_return": stationary_series(60),
        }
    )
    original = frame.copy(deep=True)
    calls: list[tuple[str, str, int]] = []

    def fake_result(
        test: str,
        series: pd.Series,
        regression: str,
    ) -> StationarityResult:
        calls.append((test, regression, len(series)))
        return StationarityResult(
            test=test,  # type: ignore[arg-type]
            statistic=1.0,
            p_value=0.04,
            lags=2,
            observations=len(series),
            critical_1pct=1.0,
            critical_5pct=2.0,
            critical_10pct=3.0,
            stationary_at_5pct=True,
        )

    monkeypatch.setattr(
        stationarity_module,
        "run_adf",
        lambda series, regression: fake_result("ADF", series, regression),
    )
    monkeypatch.setattr(
        stationarity_module,
        "run_kpss",
        lambda series, regression: fake_result("KPSS", series, regression),
    )

    result = build_stationarity_report(frame)

    assert result.columns.tolist() == [
        "ticker",
        "series",
        "regression",
        "test",
        "statistic",
        "p_value",
        "lags",
        "observations",
        "critical_1pct",
        "critical_5pct",
        "critical_10pct",
        "stationary_at_5pct",
    ]
    assert result[["ticker", "series", "test"]].values.tolist() == [
        [ticker, series, test]
        for ticker in ["AAA", "BBB"]
        for series in ["log_price", "log_return"]
        for test in ["ADF", "KPSS"]
    ]
    assert calls == [
        (test, regression, 30)
        for _ticker in ["AAA", "BBB"]
        for regression in ["ct", "c"]
        for test in ["ADF", "KPSS"]
    ]
    pd.testing.assert_frame_equal(frame, original)
