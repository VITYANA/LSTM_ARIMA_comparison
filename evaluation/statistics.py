"""Run pre-specified inference for independent final-test forecasts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from evaluation.final_metrics import FINAL_MODELS, _validate_final_predictions
from evaluation.final_models import FINAL_TEST_PROTOCOL, FinalTestProtocol

MODEL_COMPARISONS = FINAL_TEST_PROTOCOL.model_comparisons
DM_COLUMNS = (
    "ticker",
    "loss",
    "candidate",
    "comparator",
    "observations",
    "mean_loss_difference",
    "dm_statistic",
    "raw_p_value",
    "holm_p_value",
    "reject_null",
)
BOOTSTRAP_COLUMNS = (
    "ticker",
    "metric",
    "candidate",
    "comparator",
    "observations",
    "estimate",
    "ci_lower",
    "ci_upper",
    "repetitions",
    "block_length",
    "seed",
)
LOSSES = ("absolute", "squared")
METRICS = ("mae", "rmse")
NEGATIVE_VARIANCE_TOLERANCE = 1e-12


def _validated_error_matrices(
    predictions: pd.DataFrame,
    protocol: FinalTestProtocol,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Return model error matrices with shared dates as rows."""
    validated = _validate_final_predictions(predictions)
    tickers = tuple(spec.ticker for spec in protocol.ticker_specs)
    reference_calendar: pd.DataFrame | None = None
    matrices: dict[str, np.ndarray] = {}

    for model in FINAL_MODELS:
        columns: list[np.ndarray] = []
        for ticker in tickers:
            rows = (
                validated.loc[validated["model"].eq(model) & validated["ticker"].eq(ticker)]
                .sort_values(["target_date", "date"], kind="mergesort")
                .reset_index(drop=True)
            )
            calendar = rows.loc[:, ["date", "target_date"]]
            if reference_calendar is None:
                reference_calendar = calendar
            elif not calendar.equals(reference_calendar):
                raise ValueError("final tickers must share one complete test calendar")
            columns.append(
                rows["actual_return"].to_numpy(dtype="float64")
                - rows["predicted_return"].to_numpy(dtype="float64")
            )
        matrices[model] = np.column_stack(columns)
    return tickers, matrices


def _newey_west_long_run_variance(
    differential: np.ndarray,
    requested_lag: int,
) -> float:
    """Estimate long-run variance with capped Bartlett weights."""
    observations = len(differential)
    lag = min(requested_lag, observations - 1)
    centered = differential - differential.mean()
    variance = float(np.dot(centered, centered) / observations)
    for position in range(1, lag + 1):
        covariance = float(np.dot(centered[position:], centered[:-position]) / observations)
        variance += 2.0 * (1.0 - position / (lag + 1)) * covariance
    return variance


def _dm_statistic_p_value(
    mean_difference: float,
    long_run_variance: float,
    observations: int,
) -> tuple[float, float]:
    """Return a DM statistic and two-sided normal p-value."""
    if long_run_variance < -NEGATIVE_VARIANCE_TOLERANCE:
        raise ValueError("long-run variance must not be negative")
    if long_run_variance <= 0.0:
        if mean_difference == 0.0:
            return 0.0, 1.0
        return math.copysign(math.inf, mean_difference), 0.0
    statistic = mean_difference / math.sqrt(long_run_variance / observations)
    probability = math.erfc(abs(statistic) / math.sqrt(2.0))
    return statistic, probability


def _holm_adjust(probabilities: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in their original order."""
    order = np.argsort(probabilities, kind="stable")
    ordered = probabilities[order]
    adjusted_ordered = np.maximum.accumulate(
        np.minimum(ordered * np.arange(len(ordered), 0, -1), 1.0)
    )
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def _loss_matrices(error_matrices: dict[str, np.ndarray], loss: str) -> dict[str, np.ndarray]:
    """Transform forecast errors into absolute or squared losses."""
    if loss == "absolute":
        return {model: np.abs(errors) for model, errors in error_matrices.items()}
    return {model: np.square(errors) for model, errors in error_matrices.items()}


def calculate_dm_results(
    predictions: pd.DataFrame,
    protocol: FinalTestProtocol = FINAL_TEST_PROTOCOL,
) -> pd.DataFrame:
    """Calculate ticker and pooled Diebold-Mariano comparisons."""
    if not isinstance(protocol, FinalTestProtocol):
        raise ValueError("protocol must be a FinalTestProtocol")
    tickers, errors = _validated_error_matrices(predictions, protocol)
    rows: list[dict[str, object]] = []

    for ticker_position, ticker in [*enumerate(tickers), (None, "ALL")]:
        for loss in LOSSES:
            losses = _loss_matrices(errors, loss)
            for candidate, comparator in MODEL_COMPARISONS:
                differences = losses[candidate] - losses[comparator]
                if ticker_position is None:
                    differential = differences.mean(axis=1)
                else:
                    differential = differences[:, ticker_position]
                mean_difference = float(differential.mean())
                variance = _newey_west_long_run_variance(differential, protocol.dm_lag)
                statistic, probability = _dm_statistic_p_value(
                    mean_difference,
                    variance,
                    len(differential),
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "loss": loss,
                        "candidate": candidate,
                        "comparator": comparator,
                        "observations": len(differential),
                        "mean_loss_difference": mean_difference,
                        "dm_statistic": statistic,
                        "raw_p_value": probability,
                    }
                )

    result = pd.DataFrame(rows)
    result["holm_p_value"] = np.nan
    for indexes in result.groupby(["ticker", "loss"], sort=False).groups.values():
        positions = list(indexes)
        result.loc[positions, "holm_p_value"] = _holm_adjust(
            result.loc[positions, "raw_p_value"].to_numpy(dtype="float64")
        )
    result["reject_null"] = result["holm_p_value"].le(protocol.holm_alpha)
    return result.loc[:, DM_COLUMNS]


def _moving_block_indexes(
    observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample consecutive moving blocks and trim to the source length."""
    block_count = math.ceil(observations / block_length)
    starts = rng.integers(0, observations - block_length + 1, size=block_count)
    indexes = np.concatenate(
        [np.arange(start, start + block_length, dtype="int64") for start in starts]
    )
    return indexes[:observations]


def _metric_differences(
    candidate_errors: np.ndarray,
    comparator_errors: np.ndarray,
) -> tuple[float, float]:
    """Return candidate-minus-comparator MAE and RMSE."""
    candidate = candidate_errors.reshape(-1)
    comparator = comparator_errors.reshape(-1)
    mae = float(np.mean(np.abs(candidate)) - np.mean(np.abs(comparator)))
    rmse = float(np.sqrt(np.mean(np.square(candidate))) - np.sqrt(np.mean(np.square(comparator))))
    return mae, rmse


def _bootstrap_metric_differences(
    candidate_errors: np.ndarray,
    comparator_errors: np.ndarray,
    repetitions: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap fresh metric differences from shared date blocks."""
    mae = np.empty(repetitions, dtype="float64")
    rmse = np.empty(repetitions, dtype="float64")
    for repetition in range(repetitions):
        indexes = _moving_block_indexes(len(candidate_errors), block_length, rng)
        mae[repetition], rmse[repetition] = _metric_differences(
            candidate_errors[indexes],
            comparator_errors[indexes],
        )
    return mae, rmse


def calculate_bootstrap_intervals(
    predictions: pd.DataFrame,
    protocol: FinalTestProtocol = FINAL_TEST_PROTOCOL,
) -> pd.DataFrame:
    """Calculate moving-block percentile intervals for MAE and RMSE differences."""
    if not isinstance(protocol, FinalTestProtocol):
        raise ValueError("protocol must be a FinalTestProtocol")
    if type(protocol.bootstrap_repetitions) is not int or protocol.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be a positive integer")
    if type(protocol.bootstrap_seed) is not int or protocol.bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    tickers, errors = _validated_error_matrices(predictions, protocol)
    observations = len(next(iter(errors.values())))
    if observations < protocol.bootstrap_block:
        raise ValueError("test calendar must contain at least one bootstrap block")

    lower_percentile = (1.0 - protocol.bootstrap_confidence) * 50.0
    upper_percentile = 100.0 - lower_percentile
    rng = np.random.default_rng(protocol.bootstrap_seed)
    rows: list[dict[str, object]] = []
    for ticker_position, ticker in [*enumerate(tickers), (None, "ALL")]:
        for candidate, comparator in MODEL_COMPARISONS:
            if ticker_position is None:
                candidate_errors = errors[candidate]
                comparator_errors = errors[comparator]
            else:
                candidate_errors = errors[candidate][:, [ticker_position]]
                comparator_errors = errors[comparator][:, [ticker_position]]
            estimates = _metric_differences(candidate_errors, comparator_errors)
            distributions = _bootstrap_metric_differences(
                candidate_errors,
                comparator_errors,
                protocol.bootstrap_repetitions,
                protocol.bootstrap_block,
                rng,
            )
            for metric, estimate, distribution in zip(
                METRICS,
                estimates,
                distributions,
                strict=True,
            ):
                rows.append(
                    {
                        "ticker": ticker,
                        "metric": metric,
                        "candidate": candidate,
                        "comparator": comparator,
                        "observations": candidate_errors.size,
                        "estimate": estimate,
                        "ci_lower": float(np.percentile(distribution, lower_percentile)),
                        "ci_upper": float(np.percentile(distribution, upper_percentile)),
                        "repetitions": protocol.bootstrap_repetitions,
                        "block_length": protocol.bootstrap_block,
                        "seed": protocol.bootstrap_seed,
                    }
                )
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)
