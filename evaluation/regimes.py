"""Build frozen volatility regimes for independent final evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.contracts import PREDICTION_COLUMNS, require_finite_numeric
from evaluation.final_metrics import (
    RETURN_METRIC_COLUMNS,
    _return_metrics_from_aligned,
    _validate_final_predictions,
)
from evaluation.final_models import (
    FINAL_KEY_COLUMNS,
    FINAL_SORT_COLUMNS,
    FINAL_TEST_PROTOCOL,
    FinalTestProtocol,
    validate_final_dataset,
)

REGIMES = ("low", "medium", "high")
REGIME_THRESHOLD_COLUMNS = ("ticker", "lower_threshold", "upper_threshold")
REGIME_ASSIGNMENT_COLUMNS = (
    *FINAL_KEY_COLUMNS,
    "volatility",
    "regime",
)
REGIME_METRIC_COLUMNS = ("regime", *RETURN_METRIC_COLUMNS)


@dataclass(frozen=True)
class VolatilityRegimes:
    """Contain frozen ticker thresholds and final-test assignments."""

    thresholds: pd.DataFrame
    assignments: pd.DataFrame


def _classify_regime(volatility: pd.Series, lower: float, upper: float) -> np.ndarray:
    """Assign deterministic low, medium and high labels."""
    return np.select(
        [volatility.le(lower), volatility.le(upper)],
        ["low", "medium"],
        default="high",
    )


def build_volatility_regimes(
    dataset: pd.DataFrame,
    protocol: FinalTestProtocol = FINAL_TEST_PROTOCOL,
) -> VolatilityRegimes:
    """Fit train-validation volatility terciles and classify final test rows."""
    validated = validate_final_dataset(dataset, protocol)
    threshold_rows: list[dict[str, object]] = []
    assignment_frames: list[pd.DataFrame] = []
    annualization = float(np.sqrt(protocol.annualization_days))
    for spec in protocol.ticker_specs:
        rows = validated.loc[validated["ticker"].eq(spec.ticker)].sort_values(
            ["date", "target_date"],
            kind="mergesort",
        )
        volatility = (
            rows["log_return"]
            .rolling(protocol.volatility_window, min_periods=protocol.volatility_window)
            .std(ddof=1)
            * annualization
        )
        training = volatility.loc[rows["split"].isin({"train", "validation"})].dropna()
        test_mask = rows["split"].eq("test")
        test_volatility = volatility.loc[test_mask]
        if (
            training.empty
            or not np.isfinite(training.to_numpy()).all()
            or not np.isfinite(test_volatility.to_numpy()).all()
        ):
            raise ValueError(f"insufficient volatility history for ticker {spec.ticker!r}")
        lower = float(training.quantile(1 / 3))
        upper = float(training.quantile(2 / 3))
        threshold_rows.append(
            {
                "ticker": spec.ticker,
                "lower_threshold": lower,
                "upper_threshold": upper,
            }
        )
        assignments = rows.loc[test_mask, FINAL_KEY_COLUMNS].copy()
        assignments["volatility"] = test_volatility.to_numpy(dtype="float64")
        assignments["regime"] = _classify_regime(
            assignments["volatility"],
            lower,
            upper,
        )
        assignment_frames.append(assignments.loc[:, REGIME_ASSIGNMENT_COLUMNS])

    thresholds = pd.DataFrame(threshold_rows, columns=REGIME_THRESHOLD_COLUMNS)
    assignments = (
        pd.concat(assignment_frames, ignore_index=True)
        .sort_values(list(FINAL_SORT_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )
    return VolatilityRegimes(thresholds=thresholds, assignments=assignments)


def _validate_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical complete regime assignment table."""
    if assignments.columns.tolist() != list(REGIME_ASSIGNMENT_COLUMNS):
        raise ValueError("assignments must use the standard regime schema")
    if assignments.empty:
        raise ValueError("assignments must not be empty")
    for column in ("date", "target_date"):
        if not pd.api.types.is_datetime64_any_dtype(assignments[column]):
            raise ValueError(f"assignment {column} must be datetime")
        if assignments[column].isna().any():
            raise ValueError("assignment dates must be finite")
    if not assignments["split"].eq("test").all():
        raise ValueError("regime assignments must contain test rows only")
    if assignments.duplicated(list(FINAL_KEY_COLUMNS)).any():
        raise ValueError("regime assignments contain duplicate observation keys")
    require_finite_numeric(assignments["volatility"], "volatility")
    if not assignments["regime"].isin(REGIMES).all():
        raise ValueError("regime assignments contain unknown labels")
    return (
        assignments.loc[:, REGIME_ASSIGNMENT_COLUMNS]
        .sort_values(list(FINAL_SORT_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )


def calculate_regime_return_metrics(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    baseline_model: str = "naive_zero",
) -> pd.DataFrame:
    """Calculate ticker and pooled final return metrics within each regime."""
    validated_predictions = _validate_final_predictions(predictions)
    validated_assignments = _validate_assignments(assignments)
    baseline = (
        validated_predictions.loc[
            validated_predictions["model"].eq("naive_zero"), FINAL_KEY_COLUMNS
        ]
        .sort_values(list(FINAL_SORT_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )
    if not validated_assignments.loc[:, FINAL_KEY_COLUMNS].equals(baseline):
        raise ValueError("regime assignments do not match final prediction keys")
    merged = validated_predictions.merge(
        validated_assignments.loc[:, [*FINAL_KEY_COLUMNS, "regime"]],
        on=list(FINAL_KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    rows: list[pd.DataFrame] = []
    for regime in REGIMES:
        regime_predictions = merged.loc[
            merged["regime"].eq(regime), PREDICTION_COLUMNS
        ].reset_index(drop=True)
        if regime_predictions.empty:
            continue
        metrics = _return_metrics_from_aligned(
            regime_predictions,
            baseline_model,
        )
        metrics.insert(0, "regime", regime)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True).loc[:, REGIME_METRIC_COLUMNS]
