"""Calculate metrics for the independent final-test forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import pandas as pd

from evaluation.contracts import BPS_FACTOR, PREDICTION_COLUMNS, require_finite_numeric
from evaluation.final_backtest import align_final_predictions
from evaluation.final_models import (
    FINAL_KEY_COLUMNS,
    FINAL_SORT_COLUMNS,
    FINAL_TEST_PROTOCOL,
    FinalSeedResult,
    average_final_seed_predictions,
    validate_final_dataset,
)

FINAL_MODELS = ("naive_zero", "arima", "lstm", "arima_lstm")
RETURN_METRIC_COLUMNS = (
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
SEED_METRIC_COLUMNS = (
    "ticker",
    "family",
    "seed",
    "observations",
    "mae_bps",
    "rmse_bps",
)
PRICE_PREDICTION_COLUMNS = (
    *PREDICTION_COLUMNS,
    "adj_close",
    "actual_price",
    "predicted_price",
)
PRICE_METRIC_COLUMNS = (
    "ticker",
    "model",
    "observations",
    "price_mae",
    "price_rmse",
    "smape",
    "mase",
)


class PriceMetricRow(TypedDict):
    """Describe one ticker or pooled price-metric row."""

    ticker: str
    model: str
    observations: int
    price_mae: float
    price_rmse: float
    smape: float
    mase: float


def _validate_final_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one aligned final-only prediction table."""
    if "model" not in predictions:
        raise ValueError("predictions must contain a model column")
    if set(predictions["model"]) != set(FINAL_MODELS):
        raise ValueError("final predictions must contain exactly four frozen models")
    tables = {
        model: predictions.loc[predictions["model"].eq(model)].reset_index(drop=True)
        for model in FINAL_MODELS
    }
    return align_final_predictions(
        tables["naive_zero"],
        tables["arima"],
        tables["lstm"],
        tables["arima_lstm"],
    )


def _return_metric_row(
    observations: pd.DataFrame,
    baseline: pd.DataFrame,
    ticker: str,
    model: str,
) -> dict[str, object]:
    """Calculate one final return-metric row."""
    actual = observations["actual_return"].to_numpy(dtype="float64")
    predicted = observations["predicted_return"].to_numpy(dtype="float64")
    errors = actual - predicted
    baseline_errors = baseline["actual_return"].to_numpy(dtype="float64") - baseline[
        "predicted_return"
    ].to_numpy(dtype="float64")
    baseline_mae = float(np.mean(np.abs(baseline_errors)))
    baseline_mse = float(np.mean(np.square(baseline_errors)))
    if baseline_mae == 0.0:
        raise ValueError("relative metrics are undefined when baseline error is zero")
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(np.square(errors)))
    return {
        "split": "test",
        "ticker": ticker,
        "model": model,
        "observations": len(observations),
        "mae_bps": mae * BPS_FACTOR,
        "rmse_bps": float(np.sqrt(mse)) * BPS_FACTOR,
        "rel_mae": mae / baseline_mae,
        "oos_r2": 1.0 - mse / baseline_mse,
        "directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
    }


def _return_metrics_from_aligned(
    predictions: pd.DataFrame,
    baseline_model: str,
) -> pd.DataFrame:
    """Calculate return metrics from an already aligned table."""
    if baseline_model not in FINAL_MODELS:
        raise ValueError("baseline_model must be one of the frozen final models")
    rows: list[dict[str, object]] = []
    baseline_rows = predictions.loc[predictions["model"].eq(baseline_model)]
    for model in FINAL_MODELS:
        model_rows = predictions.loc[predictions["model"].eq(model)]
        for ticker in [*sorted(model_rows["ticker"].unique()), "ALL"]:
            if ticker == "ALL":
                observations = model_rows
                baseline = baseline_rows
            else:
                observations = model_rows.loc[model_rows["ticker"].eq(ticker)]
                baseline = baseline_rows.loc[baseline_rows["ticker"].eq(ticker)]
            rows.append(_return_metric_row(observations, baseline, ticker, model))
    return pd.DataFrame(rows, columns=RETURN_METRIC_COLUMNS)


def _calculate_return_metrics(
    predictions: pd.DataFrame,
    baseline_model: str,
) -> pd.DataFrame:
    """Validate final predictions and calculate aligned return metrics."""
    return _return_metrics_from_aligned(
        _validate_final_predictions(predictions),
        baseline_model,
    )


def calculate_final_return_metrics(
    predictions: pd.DataFrame,
    baseline_model: str = "naive_zero",
) -> pd.DataFrame:
    """Calculate ticker and pooled metrics on aligned final-test rows."""
    return _calculate_return_metrics(predictions, baseline_model)


def _validated_seed_groups(
    results: Sequence[FinalSeedResult],
    family: str,
) -> dict[str, tuple[FinalSeedResult, ...]]:
    """Return complete aligned seed groups for every frozen ticker."""
    values = tuple(results)
    expected_tickers = {spec.ticker for spec in FINAL_TEST_PROTOCOL.ticker_specs}
    if not all(isinstance(result, FinalSeedResult) for result in values):
        raise ValueError("seed metrics require FinalSeedResult values")
    groups = {
        ticker: tuple(result for result in values if result.ticker == ticker)
        for ticker in expected_tickers
    }
    if set(result.ticker for result in values) != expected_tickers:
        raise ValueError("seed results must contain every protocol ticker")
    specs = {spec.ticker: spec for spec in FINAL_TEST_PROTOCOL.ticker_specs}
    for ticker, group in groups.items():
        average_final_seed_predictions(group, family)
        expected_config = (
            specs[ticker].lstm_config if family == "lstm" else specs[ticker].residual_config
        )
        if any(result.config != expected_config for result in group):
            raise ValueError("seed config does not match the frozen ticker configuration")
    return groups


def calculate_seed_return_metrics(
    lstm_results: Sequence[FinalSeedResult],
    hybrid_results: Sequence[FinalSeedResult],
) -> pd.DataFrame:
    """Calculate MAE and RMSE for every selected LSTM and hybrid seed."""
    lstm_groups = _validated_seed_groups(lstm_results, "lstm")
    hybrid_groups = _validated_seed_groups(hybrid_results, "arima_lstm")
    rows: list[dict[str, object]] = []
    for family, groups in (("lstm", lstm_groups), ("arima_lstm", hybrid_groups)):
        for ticker in sorted(groups):
            ordinary = {result.seed: result for result in lstm_groups[ticker]}
            for result in sorted(groups[ticker], key=lambda value: value.seed):
                if family == "arima_lstm":
                    reference = ordinary[result.seed].predictions
                    if not result.predictions.loc[:, FINAL_KEY_COLUMNS].equals(
                        reference.loc[:, FINAL_KEY_COLUMNS]
                    ) or not result.predictions["actual_return"].equals(reference["actual_return"]):
                        raise ValueError("ordinary and hybrid seed observations do not align")
                errors = result.predictions["actual_return"].to_numpy(
                    dtype="float64"
                ) - result.predictions["predicted_return"].to_numpy(dtype="float64")
                rows.append(
                    {
                        "ticker": ticker,
                        "family": family,
                        "seed": result.seed,
                        "observations": len(result.predictions),
                        "mae_bps": float(np.mean(np.abs(errors))) * BPS_FACTOR,
                        "rmse_bps": float(np.sqrt(np.mean(np.square(errors)))) * BPS_FACTOR,
                    }
                )
    return pd.DataFrame(rows, columns=SEED_METRIC_COLUMNS)


def build_price_predictions(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct actual and forecast one-day adjusted prices."""
    validated_dataset = validate_final_dataset(dataset)
    validated_predictions = _validate_final_predictions(predictions)
    source = validated_dataset.loc[:, [*FINAL_KEY_COLUMNS, "adj_close", "target_return"]]
    merged = validated_predictions.merge(
        source,
        on=list(FINAL_KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    if merged[["adj_close", "target_return"]].isna().any().any():
        raise ValueError("predictions do not match source price observations")
    require_finite_numeric(merged["adj_close"], "adj_close")
    if not merged["adj_close"].gt(0.0).all():
        raise ValueError("adj_close must be strictly positive")
    if not np.allclose(
        merged["actual_return"].to_numpy(dtype="float64"),
        merged["target_return"].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("actual returns do not match the source dataset")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        merged["actual_price"] = merged["adj_close"] * np.exp(merged["actual_return"])
        merged["predicted_price"] = merged["adj_close"] * np.exp(merged["predicted_return"])
    for column in ("actual_price", "predicted_price"):
        require_finite_numeric(merged[column], column)
    return (
        merged.loc[:, PRICE_PREDICTION_COLUMNS]
        .sort_values(["model", *FINAL_SORT_COLUMNS], kind="mergesort")
        .reset_index(drop=True)
    )


def _mase_scales(dataset: pd.DataFrame) -> dict[str, float]:
    """Return train-and-validation naive price-error scales."""
    training = dataset.loc[dataset["split"].isin({"train", "validation"})].copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        training["target_price"] = training["adj_close"] * np.exp(training["target_return"])
    require_finite_numeric(training["target_price"], "training target_price")
    scales = (
        (training["target_price"] - training["adj_close"]).abs().groupby(training["ticker"]).mean()
    )
    if not np.isfinite(scales.to_numpy()).all() or not scales.gt(0.0).all():
        raise ValueError("MASE scale must be finite and strictly positive")
    return {str(ticker): float(scale) for ticker, scale in scales.items()}


def _price_metric_row(
    observations: pd.DataFrame,
    ticker: str,
    model: str,
    scale: float,
) -> PriceMetricRow:
    """Calculate one ticker-level price metric row."""
    actual = observations["actual_price"].to_numpy(dtype="float64")
    predicted = observations["predicted_price"].to_numpy(dtype="float64")
    absolute = np.abs(actual - predicted)
    denominator = np.abs(actual) + np.abs(predicted)
    smape_values = np.divide(
        2.0 * absolute,
        denominator,
        out=np.zeros_like(absolute),
        where=denominator != 0.0,
    )
    return {
        "ticker": ticker,
        "model": model,
        "observations": len(observations),
        "price_mae": float(np.mean(absolute)),
        "price_rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
        "smape": float(np.mean(smape_values)),
        "mase": float(np.mean(absolute)) / scale,
    }


def calculate_final_price_metrics(
    price_predictions: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate ticker price metrics and pooled scale-free macro averages."""
    if price_predictions.columns.tolist() != list(PRICE_PREDICTION_COLUMNS):
        raise ValueError("price_predictions must use the standard price schema")
    prediction_rows = _validate_final_predictions(price_predictions.loc[:, PREDICTION_COLUMNS])
    validated = prediction_rows.merge(
        price_predictions.loc[
            :, [*FINAL_KEY_COLUMNS, "model", "adj_close", "actual_price", "predicted_price"]
        ],
        on=[*FINAL_KEY_COLUMNS, "model"],
        how="left",
        validate="one_to_one",
    )
    for column in ("adj_close", "actual_price", "predicted_price"):
        require_finite_numeric(validated[column], column)
    if not validated["adj_close"].gt(0.0).all():
        raise ValueError("adj_close must be strictly positive")
    if not validated[["actual_price", "predicted_price"]].ge(0.0).all().all():
        raise ValueError("reconstructed prices must be nonnegative")
    scales = _mase_scales(validate_final_dataset(dataset))

    rows: list[PriceMetricRow] = []
    for model in FINAL_MODELS:
        model_rows = validated.loc[validated["model"].eq(model)]
        ticker_rows: list[PriceMetricRow] = []
        for ticker in sorted(model_rows["ticker"].unique()):
            observations = model_rows.loc[model_rows["ticker"].eq(ticker)]
            ticker_rows.append(_price_metric_row(observations, ticker, model, scales[ticker]))
        rows.extend(ticker_rows)
        rows.append(
            {
                "ticker": "ALL",
                "model": model,
                "observations": sum(row["observations"] for row in ticker_rows),
                "price_mae": np.nan,
                "price_rmse": np.nan,
                "smape": float(np.mean([row["smape"] for row in ticker_rows])),
                "mase": float(np.mean([row["mase"] for row in ticker_rows])),
            }
        )
    return pd.DataFrame(rows, columns=PRICE_METRIC_COLUMNS)
