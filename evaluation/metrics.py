"""Calculate relative and pooled forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.contracts import (
    BPS_FACTOR,
    EVALUATION_SPLITS,
    PREDICTION_COLUMNS,
    require_finite_numeric,
)

METRIC_COLUMNS = (
    "split",
    "ticker",
    "model",
    "observations",
    "mae_bps",
    "rmse_bps",
    "rel_mae",
    "oos_r2",
    "directional_accuracy",
)
ALIGNMENT_COLUMNS = ("split", "ticker", "target_date")
UNIQUE_KEY_COLUMNS = (*ALIGNMENT_COLUMNS, "model")


def _validate_predictions(predictions: pd.DataFrame, baseline_model: str) -> None:
    """Validate prediction-table structure and scalar values."""
    missing_columns = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"predictions missing columns: {sorted(missing_columns)}")
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    if not predictions["split"].isin(EVALUATION_SPLITS).all():
        raise ValueError("predictions contain invalid split values")
    if predictions.duplicated(list(UNIQUE_KEY_COLUMNS)).any():
        raise ValueError("predictions contain duplicate model observation keys")
    require_finite_numeric(predictions["actual_return"], "actual_return")
    require_finite_numeric(predictions["predicted_return"], "predicted_return")
    if not predictions["model"].eq(baseline_model).any():
        raise ValueError(f"baseline model {baseline_model!r} is absent")


def _sorted_model_rows(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return one model ordered by alignment columns."""
    rows = predictions.loc[predictions["model"].eq(model)].copy()
    return rows.sort_values(list(ALIGNMENT_COLUMNS)).reset_index(drop=True)


def _validate_model_alignment(predictions: pd.DataFrame, baseline_model: str) -> None:
    """Require every candidate to match baseline observations exactly."""
    baseline = _sorted_model_rows(predictions, baseline_model)
    baseline_keys = baseline.loc[:, ALIGNMENT_COLUMNS]

    for model in predictions["model"].unique():
        if model == baseline_model:
            continue
        candidate = _sorted_model_rows(predictions, model)
        candidate_keys = candidate.loc[:, ALIGNMENT_COLUMNS]
        if not candidate_keys.equals(baseline_keys):
            raise ValueError(f"model {model!r} keys do not match baseline keys")
        if not np.array_equal(
            candidate["actual_return"].to_numpy(),
            baseline["actual_return"].to_numpy(),
        ):
            raise ValueError(f"model {model!r} has inconsistent actual returns")


def _metric_row(
    observations: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    split: str,
    ticker: str,
    model: str,
) -> dict[str, object]:
    """Calculate one ticker-level or pooled metric row."""
    actual = observations["actual_return"].to_numpy(dtype="float64")
    predicted = observations["predicted_return"].to_numpy(dtype="float64")
    errors = actual - predicted

    baseline_actual = baseline["actual_return"].to_numpy(dtype="float64")
    baseline_predicted = baseline["predicted_return"].to_numpy(dtype="float64")
    baseline_errors = baseline_actual - baseline_predicted
    baseline_mae = float(np.mean(np.abs(baseline_errors)))
    baseline_mse = float(np.mean(np.square(baseline_errors)))
    if baseline_mae == 0.0:
        raise ValueError("relative metrics are undefined when baseline error is zero")

    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(np.square(errors)))
    return {
        "split": split,
        "ticker": ticker,
        "model": model,
        "observations": len(observations),
        "mae_bps": mae * BPS_FACTOR,
        "rmse_bps": float(np.sqrt(mse)) * BPS_FACTOR,
        "rel_mae": mae / baseline_mae,
        "oos_r2": 1.0 - mse / baseline_mse,
        "directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
    }


def calculate_metrics(
    predictions: pd.DataFrame,
    baseline_model: str = "naive_zero",
) -> pd.DataFrame:
    """Calculate ticker and pooled metrics relative to one baseline."""
    _validate_predictions(predictions, baseline_model)
    _validate_model_alignment(predictions, baseline_model)

    rows: list[dict[str, object]] = []
    for split in sorted(predictions["split"].unique()):
        split_rows = predictions.loc[predictions["split"].eq(split)]
        baseline_rows = split_rows.loc[split_rows["model"].eq(baseline_model)]
        for model in sorted(split_rows["model"].unique()):
            model_rows = split_rows.loc[split_rows["model"].eq(model)]
            tickers = sorted(model_rows["ticker"].unique())
            for ticker in [*tickers, "ALL"]:
                if ticker == "ALL":
                    observations = model_rows
                    baseline = baseline_rows
                else:
                    observations = model_rows.loc[model_rows["ticker"].eq(ticker)]
                    baseline = baseline_rows.loc[baseline_rows["ticker"].eq(ticker)]
                rows.append(
                    _metric_row(
                        observations,
                        baseline,
                        split=split,
                        ticker=ticker,
                        model=model,
                    )
                )

    return pd.DataFrame(rows, columns=METRIC_COLUMNS)
