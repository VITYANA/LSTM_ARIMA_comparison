"""Tests for forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import calculate_metrics

PREDICTION_COLUMNS = [
    "date",
    "target_date",
    "ticker",
    "split",
    "model",
    "actual_return",
    "predicted_return",
]
METRIC_COLUMNS = [
    "split",
    "ticker",
    "model",
    "observations",
    "mae_bps",
    "rmse_bps",
    "rel_mae",
    "oos_r2",
    "directional_accuracy",
]


def make_predictions(rows: list[list[object]]) -> pd.DataFrame:
    """Build a prediction table from compact rows"""
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame


@pytest.fixture
def paired_predictions() -> pd.DataFrame:
    """Return aligned baseline and candidate predictions"""
    return make_predictions(
        [
            ["2020-01-01", "2020-01-02", "AAA", "validation", "naive_zero", 0.01, 0.0],
            ["2020-01-02", "2020-01-03", "AAA", "validation", "naive_zero", -0.02, 0.0],
            ["2020-01-01", "2020-01-02", "AAA", "validation", "candidate", 0.01, 0.005],
            ["2020-01-02", "2020-01-03", "AAA", "validation", "candidate", -0.02, -0.01],
        ]
    )


def select_metric(
    metrics: pd.DataFrame,
    *,
    split: str,
    ticker: str,
    model: str,
) -> pd.Series:
    """Select one uniquely identified metric row"""
    selected = metrics.loc[
        metrics["split"].eq(split) & metrics["ticker"].eq(ticker) & metrics["model"].eq(model)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_calculate_metrics_matches_hand_calculated_values(
    paired_predictions: pd.DataFrame,
) -> None:
    """Match hand-calculated baseline and candidate metrics"""
    result = calculate_metrics(paired_predictions)

    assert result.columns.tolist() == METRIC_COLUMNS
    baseline = select_metric(
        result,
        split="validation",
        ticker="AAA",
        model="naive_zero",
    )
    # Baseline errors are 0.01 and -0.02
    expected_baseline = {
        "mae_bps": 150.0,
        "rmse_bps": 158.11388300841895,
        "rel_mae": 1.0,
        "oos_r2": 0.0,
        "directional_accuracy": 0.0,
    }
    assert baseline["observations"] == 2
    assert baseline[list(expected_baseline)].to_dict() == pytest.approx(expected_baseline)

    candidate = select_metric(
        result,
        split="validation",
        ticker="AAA",
        model="candidate",
    )
    # Candidate errors are 0.005 and -0.01
    expected_candidate = {
        "mae_bps": 75.0,
        "rmse_bps": 79.05694150420948,
        "rel_mae": 0.5,
        "oos_r2": 0.75,
        "directional_accuracy": 1.0,
    }
    assert candidate["observations"] == 2
    assert candidate[list(expected_candidate)].to_dict() == pytest.approx(expected_candidate)


def test_calculate_metrics_builds_weighted_pooled_rows() -> None:
    """Calculate pooled metrics directly from observations"""
    predictions = make_predictions(
        [
            ["2019-01-01", "2019-01-02", "AAA", "train", "naive_zero", 0.01, 0.0],
            ["2019-01-01", "2019-01-02", "BBB", "train", "naive_zero", 0.03, 0.0],
            ["2019-01-02", "2019-01-03", "BBB", "train", "naive_zero", -0.03, 0.0],
            ["2019-01-03", "2019-01-06", "BBB", "train", "naive_zero", 0.03, 0.0],
            [
                "2020-01-01",
                "2020-01-02",
                "AAA",
                "validation",
                "naive_zero",
                -0.02,
                0.0,
            ],
        ]
    )

    result = calculate_metrics(predictions)

    assert result[["split", "model", "ticker"]].values.tolist() == [
        ["train", "naive_zero", "AAA"],
        ["train", "naive_zero", "BBB"],
        ["train", "naive_zero", "ALL"],
        ["validation", "naive_zero", "AAA"],
        ["validation", "naive_zero", "ALL"],
    ]
    train_tickers = result.loc[
        result["split"].eq("train") & result["ticker"].ne("ALL"),
        "mae_bps",
    ]
    train_all = select_metric(
        result,
        split="train",
        ticker="ALL",
        model="naive_zero",
    )
    validation_all = select_metric(
        result,
        split="validation",
        ticker="ALL",
        model="naive_zero",
    )
    assert train_tickers.mean() == pytest.approx(200.0)
    assert train_all["observations"] == 4
    assert train_all["mae_bps"] == pytest.approx(250.0)
    assert validation_all["observations"] == 1
    assert validation_all["mae_bps"] == pytest.approx(200.0)


def test_calculate_metrics_accepts_custom_baseline_name(
    paired_predictions: pd.DataFrame,
) -> None:
    """Use caller-selected model as comparison baseline"""
    predictions = paired_predictions.copy()
    predictions["model"] = predictions["model"].replace({"naive_zero": "reference"})

    result = calculate_metrics(predictions, baseline_model="reference")

    reference = select_metric(
        result,
        split="validation",
        ticker="AAA",
        model="reference",
    )
    assert reference["rel_mae"] == pytest.approx(1.0)
    assert reference["oos_r2"] == pytest.approx(0.0)


def test_calculate_metrics_rejects_empty_predictions() -> None:
    """Reject empty prediction tables before evaluation"""
    empty = pd.DataFrame(columns=PREDICTION_COLUMNS)

    with pytest.raises(ValueError, match="must not be empty"):
        calculate_metrics(empty)


@pytest.mark.parametrize("missing_column", PREDICTION_COLUMNS)
def test_calculate_metrics_rejects_missing_columns(
    paired_predictions: pd.DataFrame,
    missing_column: str,
) -> None:
    """Reject prediction tables missing required columns"""
    invalid = paired_predictions.drop(columns=missing_column)

    with pytest.raises(ValueError, match=rf"missing columns.*{missing_column}"):
        calculate_metrics(invalid)


def test_calculate_metrics_rejects_duplicate_keys(
    paired_predictions: pd.DataFrame,
) -> None:
    """Reject duplicate model observation keys"""
    duplicate = pd.concat(
        [paired_predictions, paired_predictions.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        calculate_metrics(duplicate)


@pytest.mark.parametrize("invalid_split", ["test", "holdout", None])
def test_calculate_metrics_rejects_invalid_splits(
    paired_predictions: pd.DataFrame,
    invalid_split: str | None,
) -> None:
    """Reject test, unknown and missing splits"""
    invalid = paired_predictions.copy()
    invalid.loc[0, "split"] = invalid_split

    with pytest.raises(ValueError, match="split"):
        calculate_metrics(invalid)


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("actual_return", "invalid", "numeric"),
        ("predicted_return", "invalid", "numeric"),
        ("actual_return", np.nan, "finite"),
        ("predicted_return", np.nan, "finite"),
        ("actual_return", np.inf, "finite"),
        ("predicted_return", -np.inf, "finite"),
    ],
)
def test_calculate_metrics_rejects_invalid_returns(
    paired_predictions: pd.DataFrame,
    column: str,
    invalid_value: str | float,
    message: str,
) -> None:
    """Reject nonnumeric and nonfinite return values"""
    invalid = paired_predictions.copy()
    if isinstance(invalid_value, str):
        invalid[column] = invalid[column].astype(object)
    invalid.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match=message):
        calculate_metrics(invalid)


def test_calculate_metrics_requires_present_baseline(
    paired_predictions: pd.DataFrame,
) -> None:
    """Reject tables without selected baseline model"""
    candidate_only = paired_predictions.loc[paired_predictions["model"].eq("candidate")].copy()

    with pytest.raises(ValueError, match="baseline"):
        calculate_metrics(candidate_only)


def test_calculate_metrics_rejects_inconsistent_actual_returns(
    paired_predictions: pd.DataFrame,
) -> None:
    """Reject model rows with inconsistent actual returns"""
    invalid = paired_predictions.copy()
    candidate_index = invalid.index[invalid["model"].eq("candidate")][0]
    invalid.loc[candidate_index, "actual_return"] = 0.02

    with pytest.raises(ValueError, match="actual returns"):
        calculate_metrics(invalid)


@pytest.mark.parametrize("mismatch", ["missing", "different"])
def test_calculate_metrics_requires_exact_candidate_keys(
    paired_predictions: pd.DataFrame,
    mismatch: str,
) -> None:
    """Require candidates to match all baseline observations"""
    invalid = paired_predictions.copy()
    candidate_indexes = invalid.index[invalid["model"].eq("candidate")]
    if mismatch == "missing":
        invalid = invalid.drop(index=candidate_indexes[-1])
    else:
        invalid.loc[candidate_indexes[-1], "target_date"] = pd.Timestamp("2020-01-04")

    with pytest.raises(ValueError, match="keys"):
        calculate_metrics(invalid)


def test_calculate_metrics_rejects_perfect_baseline(
    paired_predictions: pd.DataFrame,
) -> None:
    """Reject zero baseline error denominators"""
    invalid = paired_predictions.copy()
    baseline_rows = invalid["model"].eq("naive_zero")
    invalid.loc[baseline_rows, "predicted_return"] = invalid.loc[
        baseline_rows,
        "actual_return",
    ]

    with pytest.raises(ValueError, match="baseline error"):
        calculate_metrics(invalid)
