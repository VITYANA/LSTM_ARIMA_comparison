"""Autocorrelation and stationarity diagnostics for financial series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import (  # type: ignore[import-untyped]
    acf,
    adfuller,
    kpss,
    pacf,
)

MIN_STATIONARITY_OBSERVATIONS = 30
Regression = Literal["c", "ct"]


@dataclass(frozen=True)
class StationarityResult:
    """Store a normalized result of a stationarity test."""

    test: Literal["ADF", "KPSS"]
    statistic: float
    p_value: float
    lags: int
    observations: int
    critical_1pct: float
    critical_5pct: float
    critical_10pct: float
    stationary_at_5pct: bool


def _clean_series(series: pd.Series) -> pd.Series:
    """Return finite observations and reject unsuitable inputs."""
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(clean) < MIN_STATIONARITY_OBSERVATIONS:
        raise ValueError(
            f"series must contain at least {MIN_STATIONARITY_OBSERVATIONS} finite observations"
        )
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("series must contain only finite observations")
    if clean.nunique() < 2:
        raise ValueError("series must not be constant")
    return clean


def acf_pacf_table(series: pd.Series, nlags: int = 20) -> pd.DataFrame:
    """Calculate ACF and PACF values for identical lag positions."""
    clean = _clean_series(series)
    if nlags < 1 or nlags >= len(clean) // 2:
        raise ValueError("nlags must be positive and less than half the sample size")
    return pd.DataFrame(
        {
            "lag": range(nlags + 1),
            "acf": acf(clean, nlags=nlags, fft=True),
            "pacf": pacf(clean, nlags=nlags, method="ywm"),
        }
    )


def run_adf(series: pd.Series, regression: Regression = "c") -> StationarityResult:
    """Run the Augmented Dickey-Fuller unit-root test."""
    clean = _clean_series(series)
    statistic, p_value, lags, observations, critical, _ = adfuller(
        clean,
        regression=regression,
        autolag="AIC",
        result_object=False,
    )
    return StationarityResult(
        test="ADF",
        statistic=float(statistic),
        p_value=float(p_value),
        lags=int(lags),
        observations=int(observations),
        critical_1pct=float(critical["1%"]),
        critical_5pct=float(critical["5%"]),
        critical_10pct=float(critical["10%"]),
        stationary_at_5pct=bool(p_value < 0.05),
    )


def run_kpss(series: pd.Series, regression: Regression = "c") -> StationarityResult:
    """Run the KPSS stationarity test."""
    clean = _clean_series(series)
    statistic, p_value, lags, critical = kpss(
        clean,
        regression=regression,
        nlags="auto",
        result_object=False,
    )
    return StationarityResult(
        test="KPSS",
        statistic=float(statistic),
        p_value=float(p_value),
        lags=int(lags),
        observations=len(clean),
        critical_1pct=float(critical["1%"]),
        critical_5pct=float(critical["5%"]),
        critical_10pct=float(critical["10%"]),
        stationary_at_5pct=bool(p_value >= 0.05),
    )


def build_stationarity_report(
    frame: pd.DataFrame,
    series_columns: tuple[str, ...] = ("log_price", "log_return"),
) -> pd.DataFrame:
    """Run ADF and KPSS by ticker using protocol regression specifications."""
    records: list[dict[str, object]] = []
    for ticker, group in frame.groupby("ticker", sort=True):
        for column in series_columns:
            regression: Regression = "ct" if column == "log_price" else "c"
            results = (
                run_adf(group[column], regression=regression),
                run_kpss(group[column], regression=regression),
            )
            for result in results:
                record: dict[str, object] = asdict(result)
                record.update(
                    {
                        "ticker": str(ticker),
                        "series": column,
                        "regression": regression,
                    }
                )
                records.append(record)

    columns = [
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
    return pd.DataFrame.from_records(records, columns=columns)
