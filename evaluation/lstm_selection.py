"""Aggregate LSTM seeds and select one ensemble per ticker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.contracts import (
    PREDICTION_COLUMNS,
    require_finite_numeric,
)
from evaluation.lstm_validation import CORE_TICKERS, LSTMRunResult
from evaluation.metrics import calculate_metrics
from models.interfaces import validate_model_name

EXPECTED_SEEDS = frozenset(range(10))
BASELINE_MODEL = "naive_zero"
RUN_METRIC_COLUMNS = (
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
)
CONFIGURATION_METRIC_COLUMNS = (
    "ticker",
    "window",
    "units",
    "dropout",
    "successful_seeds",
    "mean_mae_bps",
    "mean_rmse_bps",
    "std_mae_bps",
    "parameter_count",
)
CONFIGURATION_COLUMNS = (*CONFIGURATION_METRIC_COLUMNS, "model")
CONFIGURATION_KEY = ("ticker", "window", "units", "dropout")
OBSERVATION_COLUMNS = (
    "date",
    "target_date",
    "ticker",
    "split",
    "actual_return",
)


@dataclass(frozen=True)
class LSTMSelection:
    """Contain seed metrics, selected configurations and ensembles."""

    run_metrics: pd.DataFrame
    configuration_metrics: pd.DataFrame
    configurations: pd.DataFrame
    predictions: pd.DataFrame


def _run_metric_table(results: tuple[LSTMRunResult, ...]) -> pd.DataFrame:
    """Return one deterministic row per terminal seed run."""
    rows = [
        {
            "ticker": result.key.ticker,
            "window": result.key.config.window,
            "units": result.key.config.units,
            "dropout": result.key.config.dropout,
            "seed": result.key.seed,
            "status": result.status,
            "best_epoch": result.best_epoch,
            "parameter_count": result.parameter_count,
            "mae_bps": result.mae_bps,
            "rmse_bps": result.rmse_bps,
            "error": result.error,
        }
        for result in results
    ]
    frame = pd.DataFrame(rows, columns=RUN_METRIC_COLUMNS)
    if frame.duplicated([*CONFIGURATION_KEY, "seed"]).any():
        raise ValueError("results contain duplicate configuration seed keys")
    return frame.sort_values([*CONFIGURATION_KEY, "seed"]).reset_index(drop=True)


def _configuration_metric_table(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful seed metrics for every observed configuration."""
    rows: list[dict[str, object]] = []
    for keys, group in run_metrics.groupby(list(CONFIGURATION_KEY), sort=True):
        successful = group.loc[group["status"].eq("ok")]
        parameter_counts = successful["parameter_count"].dropna().unique()
        if len(parameter_counts) > 1:
            raise ValueError("successful seeds have inconsistent parameter counts")
        rows.append(
            {
                **dict(zip(CONFIGURATION_KEY, keys, strict=True)),
                "successful_seeds": len(successful),
                "mean_mae_bps": successful["mae_bps"].mean(),
                "mean_rmse_bps": successful["rmse_bps"].mean(),
                "std_mae_bps": successful["mae_bps"].std(ddof=1),
                "parameter_count": parameter_counts[0] if len(parameter_counts) == 1 else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=CONFIGURATION_METRIC_COLUMNS)


def _eligible_configuration_keys(
    results: tuple[LSTMRunResult, ...],
) -> set[tuple[object, ...]]:
    """Return configurations with exactly ten successful expected seeds."""
    groups: dict[tuple[object, ...], list[LSTMRunResult]] = {}
    for result in results:
        key = (
            result.key.ticker,
            result.key.config.window,
            result.key.config.units,
            result.key.config.dropout,
        )
        groups.setdefault(key, []).append(result)
    return {
        key
        for key, runs in groups.items()
        if {run.key.seed for run in runs} == EXPECTED_SEEDS
        and all(run.status == "ok" for run in runs)
    }


def _validate_result_predictions(result: LSTMRunResult) -> pd.DataFrame:
    """Return validated and ordered predictions for one successful seed."""
    predictions = result.predictions
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError("seed predictions must use standard columns")
    if predictions.empty:
        raise ValueError("successful seed predictions must not be empty")
    if not predictions["split"].eq("validation").all():
        raise ValueError("seed predictions must contain validation rows only")
    if not predictions["ticker"].eq(result.key.ticker).all():
        raise ValueError("seed prediction ticker does not match run key")
    if predictions.duplicated(["date", "target_date", "ticker", "split", "model"]).any():
        raise ValueError("seed predictions contain duplicate observation keys")
    require_finite_numeric(predictions["actual_return"], "actual_return")
    require_finite_numeric(predictions["predicted_return"], "predicted_return")
    return predictions.sort_values(["ticker", "target_date", "date"]).reset_index(drop=True)


def _configuration_predictions(runs: list[LSTMRunResult]) -> list[pd.DataFrame]:
    """Return ten seed tables after exact within-configuration alignment."""
    tables = [_validate_result_predictions(result) for result in runs]
    expected = tables[0].loc[:, OBSERVATION_COLUMNS]
    if any(not table.loc[:, OBSERVATION_COLUMNS].equals(expected) for table in tables[1:]):
        raise ValueError("seed prediction observations do not align")
    return tables


def _validate_baseline(baseline: pd.DataFrame) -> pd.DataFrame:
    """Return an isolated aligned naive-zero validation table."""
    if baseline.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError("baseline predictions must use standard columns")
    if baseline.empty:
        raise ValueError("baseline predictions must not be empty")
    if not baseline["split"].eq("validation").all():
        raise ValueError("baseline predictions must contain validation rows only")
    if not baseline["model"].eq(BASELINE_MODEL).all():
        raise ValueError(f"baseline predictions must use {BASELINE_MODEL!r}")
    if set(baseline["ticker"]) != set(CORE_TICKERS):
        raise ValueError("baseline ticker set must match core tickers")
    if baseline.duplicated(["date", "target_date", "ticker", "split", "model"]).any():
        raise ValueError("baseline predictions contain duplicate observation keys")
    require_finite_numeric(baseline["actual_return"], "actual_return")
    require_finite_numeric(baseline["predicted_return"], "predicted_return")
    return baseline.copy(deep=True)


def select_lstm_candidates(
    results: Sequence[LSTMRunResult],
    baseline_predictions: pd.DataFrame,
    selected_model: str = "lstm",
) -> LSTMSelection:
    """Select complete ticker candidates and average their seed forecasts."""
    selected_name = validate_model_name(selected_model)
    result_values = tuple(results)
    if not result_values or not all(isinstance(result, LSTMRunResult) for result in result_values):
        raise ValueError("results must contain LSTMRunResult values")
    baseline = _validate_baseline(baseline_predictions)
    run_metrics = _run_metric_table(result_values)
    configuration_metrics = _configuration_metric_table(run_metrics)
    eligible_keys = _eligible_configuration_keys(result_values)
    eligibility = configuration_metrics.apply(
        lambda row: tuple(row[column] for column in CONFIGURATION_KEY) in eligible_keys,
        axis=1,
    )
    eligible = configuration_metrics.loc[eligibility].copy()

    selected_rows: list[pd.Series] = []
    for ticker in CORE_TICKERS:
        ticker_candidates = eligible.loc[eligible["ticker"].eq(ticker)]
        if ticker_candidates.empty:
            raise ValueError(f"ticker {ticker!r} has no complete successful configuration")
        winner = ticker_candidates.sort_values(
            [
                "mean_mae_bps",
                "mean_rmse_bps",
                "std_mae_bps",
                "parameter_count",
                "window",
                "units",
                "dropout",
            ]
        ).iloc[0]
        selected_rows.append(winner)

    configurations = pd.DataFrame(selected_rows).loc[:, CONFIGURATION_METRIC_COLUMNS]
    configurations["model"] = selected_name
    configurations = configurations.loc[:, CONFIGURATION_COLUMNS].reset_index(drop=True)

    ensemble_frames: list[pd.DataFrame] = []
    for selected in configurations.itertuples(index=False):
        runs = [
            result
            for result in result_values
            if result.key.ticker == selected.ticker
            and result.key.config.window == selected.window
            and result.key.config.units == selected.units
            and result.key.config.dropout == selected.dropout
        ]
        tables = _configuration_predictions(runs)
        combined = pd.concat(tables, ignore_index=True)
        ensemble = (
            combined.groupby(list(OBSERVATION_COLUMNS), as_index=False, sort=True)[
                "predicted_return"
            ]
            .mean()
            .assign(model=selected_name)
        )
        ensemble_frames.append(ensemble.loc[:, PREDICTION_COLUMNS])

    predictions = pd.concat(ensemble_frames, ignore_index=True).sort_values(
        ["ticker", "target_date", "date"]
    )
    predictions = predictions.reset_index(drop=True)
    calculate_metrics(
        pd.concat([baseline, predictions], ignore_index=True),
        baseline_model=BASELINE_MODEL,
    )
    return LSTMSelection(
        run_metrics=run_metrics,
        configuration_metrics=configuration_metrics.reset_index(drop=True),
        configurations=configurations,
        predictions=predictions,
    )
