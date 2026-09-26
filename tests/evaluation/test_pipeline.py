"""Tests for the model-agnostic prediction pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pytest

from evaluation.pipeline import build_predictions
from models.interfaces import ForecastModel
from models.naive import NaiveModel

DATASET_COLUMNS = [
    "date",
    "target_date",
    "ticker",
    "adj_close",
    "log_return",
    "target_return",
    "split",
]
PREDICTION_COLUMNS = [
    "date",
    "target_date",
    "ticker",
    "split",
    "model",
    "actual_return",
    "predicted_return",
]
PredictionFactory = Callable[[pd.DataFrame], object]


@pytest.fixture
def model_dataset() -> pd.DataFrame:
    """Return shuffled rows across all research splits"""
    frame = pd.DataFrame(
        [
            ["2016-01-05", "2016-01-06", "BBB", 51.0, 0.02, -0.01, "validation"],
            ["2010-01-04", "2010-01-05", "AAA", 100.0, 0.01, 0.02, "train"],
            ["2020-01-02", "2020-01-03", "AAA", 120.0, -0.01, 0.03, "test"],
            ["2016-01-04", "2016-01-05", "BBB", 50.0, -0.02, 0.02, "validation"],
            ["2010-01-05", "2010-01-06", "BBB", 80.0, 0.03, -0.04, "train"],
        ],
        columns=DATASET_COLUMNS,
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame


class ConstantModel:
    """Return one constant through structural model typing."""

    @property
    def name(self) -> str:
        """Return the custom model identifier"""
        return "constant"

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Return constant predictions aligned to observations"""
        return pd.Series(
            0.25,
            index=observations.index,
            dtype="float64",
            name="predicted_return",
        )


class SpyModel:
    """Capture observations received from the pipeline."""

    def __init__(self) -> None:
        self.received: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        """Return the spy model identifier"""
        return "spy"

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Capture observations and return aligned zeros"""
        self.received = observations.copy(deep=True)
        return pd.Series(
            0.0,
            index=observations.index,
            dtype="float64",
            name="predicted_return",
        )


class InvalidNameModel:
    """Expose an invalid runtime model name."""

    def __init__(self, name: object) -> None:
        self.name = name

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Return otherwise valid aligned predictions"""
        return pd.Series(
            0.0,
            index=observations.index,
            dtype="float64",
            name="predicted_return",
        )


class InvalidPredictionModel:
    """Return a supplied invalid prediction object."""

    def __init__(self, factory: PredictionFactory) -> None:
        self.factory = factory

    @property
    def name(self) -> str:
        """Return the invalid-output model identifier"""
        return "invalid_output"

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Return the configured runtime prediction object"""
        return cast(pd.Series, self.factory(observations))


def aligned_predictions(
    observations: pd.DataFrame,
    values: object = 0.0,
    *,
    name: str | None = "predicted_return",
) -> pd.Series:
    """Build aligned prediction series for validation cases"""
    return pd.Series(values, index=observations.index, name=name)


def test_build_predictions_creates_sorted_contract(
    model_dataset: pd.DataFrame,
) -> None:
    """Build deterministic baseline rows without source mutation"""
    original = model_dataset.copy(deep=True)

    result = build_predictions(model_dataset, NaiveModel())

    pd.testing.assert_frame_equal(model_dataset, original)
    assert result.columns.tolist() == PREDICTION_COLUMNS
    assert result[["split", "ticker", "target_date"]].values.tolist() == [
        ["train", "AAA", pd.Timestamp("2010-01-05")],
        ["train", "BBB", pd.Timestamp("2010-01-06")],
        ["validation", "BBB", pd.Timestamp("2016-01-05")],
        ["validation", "BBB", pd.Timestamp("2016-01-06")],
    ]
    assert result["model"].tolist() == ["naive_zero"] * 4
    assert result["actual_return"].tolist() == [0.02, -0.04, 0.02, -0.01]
    assert result["predicted_return"].tolist() == [0.0] * 4
    assert "test" not in result["split"].values


def test_build_predictions_accepts_structural_custom_model(
    model_dataset: pd.DataFrame,
) -> None:
    """Accept models without explicit protocol inheritance"""
    model: ForecastModel = ConstantModel()

    result = build_predictions(model_dataset, model)

    assert result["model"].eq("constant").all()
    assert result["predicted_return"].eq(0.25).all()


def test_build_predictions_hides_target_and_final_test_from_model(
    model_dataset: pd.DataFrame,
) -> None:
    """Hide targets and final test from model input"""
    model = SpyModel()

    build_predictions(model_dataset, model)

    assert model.received is not None
    assert "target_return" not in model.received.columns
    assert set(model.received["split"]) == {"train", "validation"}
    assert model.received["target_date"].max() == pd.Timestamp("2016-01-06")


@pytest.mark.parametrize("missing_column", DATASET_COLUMNS)
def test_build_predictions_rejects_missing_columns(
    model_dataset: pd.DataFrame,
    missing_column: str,
) -> None:
    """Reject datasets missing required model columns"""
    invalid = model_dataset.drop(columns=missing_column)

    with pytest.raises(ValueError, match=rf"missing columns.*{missing_column}"):
        build_predictions(invalid, NaiveModel())


def test_build_predictions_rejects_empty_dataset() -> None:
    """Reject empty model datasets before prediction"""
    empty = pd.DataFrame(columns=DATASET_COLUMNS)

    with pytest.raises(ValueError, match="must not be empty"):
        build_predictions(empty, NaiveModel())


def test_build_predictions_rejects_only_test_rows(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject datasets without evaluable research rows"""
    test_only = model_dataset.loc[model_dataset["split"].eq("test")].copy()

    with pytest.raises(ValueError, match="train or validation"):
        build_predictions(test_only, NaiveModel())


@pytest.mark.parametrize("invalid_split", ["holdout", None])
def test_build_predictions_rejects_unknown_splits(
    model_dataset: pd.DataFrame,
    invalid_split: str | None,
) -> None:
    """Reject unknown or missing split labels"""
    invalid = model_dataset.copy()
    invalid.loc[0, "split"] = invalid_split

    with pytest.raises(ValueError, match="unknown split"):
        build_predictions(invalid, NaiveModel())


def test_build_predictions_rejects_duplicate_keys(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject duplicate forecast observation keys"""
    duplicate = pd.concat([model_dataset, model_dataset.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        build_predictions(duplicate, NaiveModel())


@pytest.mark.parametrize(
    ("invalid_return", "message"),
    [
        ("not-a-number", "numeric"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        (-np.inf, "finite"),
    ],
)
def test_build_predictions_rejects_invalid_targets(
    model_dataset: pd.DataFrame,
    invalid_return: str | float,
    message: str,
) -> None:
    """Reject nonnumeric and nonfinite target returns"""
    invalid = model_dataset.copy()
    if isinstance(invalid_return, str):
        invalid["target_return"] = invalid["target_return"].astype(object)
    invalid.loc[0, "target_return"] = invalid_return

    with pytest.raises(ValueError, match=message):
        build_predictions(invalid, NaiveModel())


@pytest.mark.parametrize("invalid_name", [None, 123, "", "   "])
def test_build_predictions_rejects_invalid_model_names(
    model_dataset: pd.DataFrame,
    invalid_name: object,
) -> None:
    """Reject empty and non-string model names"""
    model = cast(ForecastModel, InvalidNameModel(invalid_name))

    with pytest.raises(ValueError, match="model name"):
        build_predictions(model_dataset, model)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda rows: [0.0] * len(rows), "Series"),
        (lambda rows: aligned_predictions(rows, name="forecast"), "name"),
        (
            lambda rows: pd.Series(
                0.0,
                index=rows.index[:-1],
                name="predicted_return",
            ),
            "length",
        ),
        (
            lambda rows: pd.Series(
                0.0,
                index=pd.RangeIndex(len(rows)),
                name="predicted_return",
            ),
            "index",
        ),
        (lambda rows: aligned_predictions(rows, "invalid"), "numeric"),
        (lambda rows: aligned_predictions(rows, np.nan), "finite"),
        (lambda rows: aligned_predictions(rows, np.inf), "finite"),
    ],
)
def test_build_predictions_rejects_invalid_model_outputs(
    model_dataset: pd.DataFrame,
    factory: PredictionFactory,
    message: str,
) -> None:
    """Reject malformed or nonfinite prediction outputs"""
    model = InvalidPredictionModel(factory)

    with pytest.raises(ValueError, match=message):
        build_predictions(model_dataset, model)
