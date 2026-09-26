"""Tests for validation-based ARIMA candidate selection."""

from __future__ import annotations

import pandas as pd
import pytest

from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.metrics import calculate_metrics
from evaluation.selection import ARIMASelection, select_arima_candidates

SHORTLIST_COLUMNS = ["ticker", "p", "d", "q", "bic", "bic_rank", "model"]
CANDIDATE_METRIC_COLUMNS = [
    "split",
    "ticker",
    "model",
    "observations",
    "mae_bps",
    "rmse_bps",
    "rel_mae",
    "oos_r2",
    "directional_accuracy",
    "p",
    "d",
    "q",
    "bic",
    "bic_rank",
]
ORDER_COLUMNS = ["ticker", "p", "d", "q", "bic", "bic_rank", "model"]


def make_predictions(rows: list[list[object]]) -> pd.DataFrame:
    """Build prediction tables from compact rows"""
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame


def aligned_model_rows(
    ticker: str,
    actual: list[float],
    by_model: dict[str, list[float]],
) -> list[list[object]]:
    """Build aligned rows for one ticker and many models"""
    rows: list[list[object]] = []
    dates = pd.date_range("2020-01-01", periods=len(actual), freq="D")
    for model, predicted in by_model.items():
        for date, actual_return, predicted_return in zip(
            dates,
            actual,
            predicted,
            strict=True,
        ):
            rows.append(
                [
                    date,
                    date + pd.Timedelta(days=1),
                    ticker,
                    "validation",
                    model,
                    actual_return,
                    predicted_return,
                ]
            )
    return rows


def make_shortlist(
    ticker_orders: dict[str, list[tuple[int, int, int, float]]],
) -> pd.DataFrame:
    """Build ranked shortlist metadata for each ticker"""
    rows: list[list[object]] = []
    for ticker, orders in ticker_orders.items():
        for rank, (p, d, q, bic) in enumerate(orders, start=1):
            rows.append([ticker, p, d, q, bic, rank, f"arima_bic_{rank}"])
    return pd.DataFrame(rows, columns=SHORTLIST_COLUMNS)


@pytest.fixture
def selection_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned predictions with different ticker winners"""
    rows = aligned_model_rows(
        "AAA",
        [0.02, -0.02],
        {
            "naive_zero": [0.0, 0.0],
            "arima_bic_1": [0.01, -0.01],
            "arima_bic_2": [0.02, -0.02],
            "arima_bic_3": [-0.01, 0.01],
        },
    )
    rows += aligned_model_rows(
        "BBB",
        [0.03, -0.01],
        {
            "naive_zero": [0.0, 0.0],
            "arima_bic_1": [0.025, -0.005],
            "arima_bic_2": [0.01, 0.01],
            "arima_bic_3": [-0.02, 0.02],
        },
    )
    shortlist = make_shortlist(
        {
            "AAA": [(1, 0, 0, 10.0), (0, 0, 1, 11.0), (1, 0, 1, 12.0)],
            "BBB": [(2, 0, 0, 20.0), (0, 0, 2, 21.0), (2, 0, 1, 22.0)],
        }
    )
    return make_predictions(rows), shortlist


def test_select_arima_candidates_builds_unified_predictions(
    selection_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Select ticker winners and build unified predictions"""
    predictions, shortlist = selection_inputs
    original_predictions = predictions.copy(deep=True)
    original_shortlist = shortlist.copy(deep=True)

    selected = select_arima_candidates(predictions, shortlist)

    assert isinstance(selected, ARIMASelection)
    pd.testing.assert_frame_equal(predictions, original_predictions)
    pd.testing.assert_frame_equal(shortlist, original_shortlist)
    assert selected.candidate_metrics.columns.tolist() == CANDIDATE_METRIC_COLUMNS
    assert selected.orders.columns.tolist() == ORDER_COLUMNS
    assert selected.orders[["ticker", "bic_rank"]].values.tolist() == [
        ["AAA", 2],
        ["BBB", 1],
    ]
    assert selected.predictions["model"].eq("arima").all()
    assert selected.predictions["split"].eq("validation").all()
    assert selected.predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert selected.predictions[["ticker", "predicted_return"]].values.tolist() == [
        ["AAA", 0.02],
        ["AAA", -0.02],
        ["BBB", 0.025],
        ["BBB", -0.005],
    ]

    baseline = predictions.loc[predictions["model"].eq("naive_zero")]
    metrics = calculate_metrics(pd.concat([baseline, selected.predictions]))
    assert set(metrics["ticker"]) == {"AAA", "BBB", "ALL"}
    assert set(metrics["model"]) == {"naive_zero", "arima"}


@pytest.mark.parametrize(
    ("candidate_one", "candidate_two", "orders", "expected_rank"),
    [
        pytest.param(
            [0.03, 0.01],
            [0.02, 0.02],
            [(1, 0, 0, 1.0), (0, 0, 1, 1.0), (1, 0, 1, 2.0)],
            2,
            id="rmse",
        ),
        pytest.param(
            [0.02, 0.02],
            [0.02, 0.02],
            [(0, 2, 0, 1.0), (0, 0, 1, 1.0), (1, 0, 2, 2.0)],
            2,
            id="complexity",
        ),
        pytest.param(
            [0.02, 0.02],
            [0.02, 0.02],
            [(1, 0, 0, 2.0), (0, 0, 1, 1.0), (1, 0, 1, 3.0)],
            2,
            id="bic",
        ),
        pytest.param(
            [0.02, 0.02],
            [0.02, 0.02],
            [(1, 0, 0, 1.0), (0, 0, 1, 1.0), (1, 0, 1, 2.0)],
            2,
            id="p",
        ),
        pytest.param(
            [0.02, 0.02],
            [0.02, 0.02],
            [(0, 1, 0, 1.0), (0, 0, 1, 1.0), (1, 0, 1, 2.0)],
            1,
            id="q",
        ),
    ],
)
def test_select_arima_candidates_applies_deterministic_tie_breaks(
    candidate_one: list[float],
    candidate_two: list[float],
    orders: list[tuple[int, int, int, float]],
    expected_rank: int,
) -> None:
    """Apply validation selection keys in declared priority"""
    rows = aligned_model_rows(
        "AAA",
        [0.03, 0.03],
        {
            "naive_zero": [0.0, 0.0],
            "arima_bic_1": candidate_one,
            "arima_bic_2": candidate_two,
            "arima_bic_3": [0.0, 0.0],
        },
    )

    result = select_arima_candidates(
        make_predictions(rows),
        make_shortlist({"AAA": orders}),
    )

    assert result.orders.loc[0, "bic_rank"] == expected_rank


def test_select_arima_candidates_rejects_empty_predictions(
    selection_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Reject empty prediction tables before selection"""
    _, shortlist = selection_inputs
    empty = pd.DataFrame(columns=PREDICTION_COLUMNS)

    with pytest.raises(ValueError, match="predictions.*empty"):
        select_arima_candidates(empty, shortlist)


def test_select_arima_candidates_rejects_empty_shortlist(
    selection_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Reject empty shortlist tables before selection"""
    predictions, _ = selection_inputs
    empty = pd.DataFrame(columns=SHORTLIST_COLUMNS)

    with pytest.raises(ValueError, match="shortlist.*empty"):
        select_arima_candidates(predictions, empty)


@pytest.mark.parametrize(
    ("input_name", "missing_column"),
    [("predictions", "predicted_return"), ("shortlist", "bic")],
)
def test_select_arima_candidates_rejects_missing_columns(
    selection_inputs: tuple[pd.DataFrame, pd.DataFrame],
    input_name: str,
    missing_column: str,
) -> None:
    """Reject inputs missing required selection columns"""
    predictions, shortlist = selection_inputs
    if input_name == "predictions":
        predictions = predictions.drop(columns=missing_column)
    else:
        shortlist = shortlist.drop(columns=missing_column)

    with pytest.raises(ValueError, match=rf"{input_name}.*{missing_column}"):
        select_arima_candidates(predictions, shortlist)


def test_select_arima_candidates_requires_validation_rows(
    selection_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Reject non-validation prediction rows"""
    predictions, shortlist = selection_inputs
    predictions = predictions.assign(split="train")

    with pytest.raises(ValueError, match="validation"):
        select_arima_candidates(predictions, shortlist)


@pytest.mark.parametrize("case", ["absent_baseline", "missing_candidate", "extra_candidate"])
def test_select_arima_candidates_requires_exact_model_set(
    selection_inputs: tuple[pd.DataFrame, pd.DataFrame],
    case: str,
) -> None:
    """Require baseline and exact shortlisted candidate models"""
    predictions, shortlist = selection_inputs
    if case == "absent_baseline":
        predictions = predictions.loc[predictions["model"].ne("naive_zero")]
    elif case == "missing_candidate":
        predictions = predictions.loc[predictions["model"].ne("arima_bic_3")]
    else:
        extra = predictions.loc[predictions["model"].eq("arima_bic_3")].copy()
        extra["model"] = "arima_bic_4"
        predictions = pd.concat([predictions, extra], ignore_index=True)

    with pytest.raises(ValueError, match="models|baseline"):
        select_arima_candidates(predictions, shortlist)


def test_select_arima_candidates_requires_aligned_candidate_keys(
    selection_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Require candidates aligned with baseline observations"""
    predictions, shortlist = selection_inputs
    invalid = predictions.copy()
    index = invalid.index[invalid["model"].eq("arima_bic_1")][0]
    invalid.loc[index, "target_date"] = pd.Timestamp("2030-01-01")

    with pytest.raises(ValueError, match="keys"):
        select_arima_candidates(invalid, shortlist)


@pytest.mark.parametrize(
    "case",
    ["duplicate", "missing_ticker", "missing_rank", "model_name"],
)
def test_select_arima_candidates_rejects_invalid_shortlist_mapping(
    selection_inputs: tuple[pd.DataFrame, pd.DataFrame],
    case: str,
) -> None:
    """Reject duplicate incomplete or inconsistent shortlist mappings"""
    predictions, shortlist = selection_inputs
    if case == "duplicate":
        shortlist = pd.concat([shortlist, shortlist.iloc[[0]]], ignore_index=True)
    elif case == "missing_ticker":
        shortlist = shortlist.loc[shortlist["ticker"].eq("AAA")]
    elif case == "missing_rank":
        shortlist = shortlist.drop(index=shortlist.index[0])
    else:
        shortlist = shortlist.copy()
        shortlist.loc[0, "model"] = "wrong_name"

    with pytest.raises(ValueError, match="duplicate|ticker|rank|model"):
        select_arima_candidates(predictions, shortlist)


@pytest.mark.parametrize("selected_model", ["", "   "])
def test_select_arima_candidates_rejects_blank_selected_name(
    selection_inputs: tuple[pd.DataFrame, pd.DataFrame],
    selected_model: str,
) -> None:
    """Reject blank unified selected model names"""
    predictions, shortlist = selection_inputs

    with pytest.raises(ValueError, match="model name"):
        select_arima_candidates(predictions, shortlist, selected_model=selected_model)
