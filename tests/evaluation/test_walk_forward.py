"""Tests for leakage-safe walk-forward validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pytest

from evaluation.walk_forward import build_walk_forward_predictions
from models.interfaces import ForecastModel

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
    """Return shuffled two-ticker chronological research rows"""
    frame = pd.DataFrame(
        [
            ["2016-01-05", "2016-01-06", "BBB", 52.0, -0.04, 0.05, "validation"],
            ["2010-01-04", "2010-01-05", "AAA", 100.0, 0.01, 0.02, "train"],
            ["2015-12-31", "2016-01-01", "BBB", 49.0, 0.90, 0.91, "test"],
            ["2016-01-04", "2016-01-05", "AAA", 101.0, 0.02, 0.03, "validation"],
            ["2010-01-04", "2010-01-05", "BBB", 50.0, -0.01, -0.02, "train"],
            ["2016-01-05", "2016-01-06", "AAA", 102.0, 0.03, 0.04, "validation"],
            ["2016-01-04", "2016-01-05", "BBB", 51.0, -0.03, 0.04, "validation"],
            ["2015-12-31", "2016-01-01", "AAA", 99.0, -0.90, -0.91, "test"],
        ],
        columns=DATASET_COLUMNS,
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame


class RecordingModel:
    """Record one observation passed to a fitted model."""

    def __init__(self, observations: list[pd.DataFrame]) -> None:
        self._observations = observations

    @property
    def name(self) -> str:
        """Return the stable recording model name"""
        return "recording"

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Record model input and return an aligned prediction"""
        self._observations.append(observations.copy(deep=True))
        return pd.Series(
            0.25,
            index=observations.index,
            dtype="float64",
            name="predicted_return",
        )


class RecordingFitter:
    """Record every ticker-specific expanding history."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, pd.DataFrame]] = []
        self.observations: list[pd.DataFrame] = []

    def __call__(self, ticker: str, history: pd.DataFrame) -> ForecastModel:
        """Record history and return a recording model"""
        self.calls.append((ticker, history.copy(deep=True)))
        return RecordingModel(self.observations)


class RuntimeModel:
    """Expose controlled runtime names and predictions."""

    def __init__(self, name: object, factory: PredictionFactory) -> None:
        self.name = name
        self._factory = factory

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Return the configured prediction object"""
        return cast(pd.Series, self._factory(observations))


def aligned_predictions(
    observations: pd.DataFrame,
    values: object = 0.0,
    *,
    name: str | None = "predicted_return",
) -> pd.Series:
    """Build aligned predictions for validation cases"""
    return pd.Series(values, index=observations.index, name=name)


def test_build_walk_forward_predictions_uses_expanding_history(
    model_dataset: pd.DataFrame,
) -> None:
    """Build sorted validation predictions from expanding histories"""
    original = model_dataset.copy(deep=True)
    fitter = RecordingFitter()

    result = build_walk_forward_predictions(model_dataset, fitter)

    pd.testing.assert_frame_equal(model_dataset, original)
    assert result.columns.tolist() == PREDICTION_COLUMNS
    assert result[["ticker", "target_date"]].values.tolist() == [
        ["AAA", pd.Timestamp("2016-01-05")],
        ["AAA", pd.Timestamp("2016-01-06")],
        ["BBB", pd.Timestamp("2016-01-05")],
        ["BBB", pd.Timestamp("2016-01-06")],
    ]
    assert result["split"].eq("validation").all()
    assert result["model"].eq("recording").all()
    assert result["actual_return"].tolist() == [0.03, 0.04, 0.04, 0.05]
    assert result["predicted_return"].eq(0.25).all()
    assert len(fitter.calls) == 4
    assert [len(history) for _, history in fitter.calls] == [2, 3, 2, 3]

    for (ticker, history), observation in zip(
        fitter.calls,
        fitter.observations,
        strict=True,
    ):
        origin_date = observation["date"].iloc[0]
        assert "target_return" not in history.columns
        assert "target_return" not in observation.columns
        assert history["ticker"].eq(ticker).all()
        assert history["date"].max() <= origin_date
        assert "test" not in history["split"].values


@pytest.mark.parametrize("missing_column", DATASET_COLUMNS)
def test_build_walk_forward_predictions_rejects_missing_columns(
    model_dataset: pd.DataFrame,
    missing_column: str,
) -> None:
    """Reject datasets missing required model columns"""
    invalid = model_dataset.drop(columns=missing_column)

    with pytest.raises(ValueError, match=rf"missing columns.*{missing_column}"):
        build_walk_forward_predictions(invalid, RecordingFitter())


def test_build_walk_forward_predictions_rejects_empty_dataset() -> None:
    """Reject empty datasets before walk-forward evaluation"""
    empty = pd.DataFrame(columns=DATASET_COLUMNS)

    with pytest.raises(ValueError, match="must not be empty"):
        build_walk_forward_predictions(empty, RecordingFitter())


def test_build_walk_forward_predictions_requires_validation_rows(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject datasets without validation observations"""
    train_and_test = model_dataset.loc[model_dataset["split"].ne("validation")]

    with pytest.raises(ValueError, match="validation"):
        build_walk_forward_predictions(train_and_test, RecordingFitter())


def test_build_walk_forward_predictions_rejects_unknown_splits(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject unknown dataset split labels"""
    invalid = model_dataset.copy()
    invalid.loc[0, "split"] = "holdout"

    with pytest.raises(ValueError, match="unknown split"):
        build_walk_forward_predictions(invalid, RecordingFitter())


def test_build_walk_forward_predictions_rejects_duplicate_keys(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject duplicate forecast observation keys"""
    duplicate = pd.concat([model_dataset, model_dataset.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        build_walk_forward_predictions(duplicate, RecordingFitter())


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("log_return", "invalid", "numeric"),
        ("log_return", np.nan, "finite"),
        ("log_return", np.inf, "finite"),
        ("target_return", "invalid", "numeric"),
        ("target_return", np.nan, "finite"),
        ("target_return", np.inf, "finite"),
    ],
)
def test_build_walk_forward_predictions_rejects_invalid_returns(
    model_dataset: pd.DataFrame,
    column: str,
    invalid_value: str | float,
    message: str,
) -> None:
    """Reject nonnumeric and nonfinite return values"""
    invalid = model_dataset.copy()
    if isinstance(invalid_value, str):
        invalid[column] = invalid[column].astype(object)
    invalid.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match=message):
        build_walk_forward_predictions(invalid, RecordingFitter())


@pytest.mark.parametrize("invalid_name", [None, 123, "", "   "])
def test_build_walk_forward_predictions_rejects_invalid_model_names(
    model_dataset: pd.DataFrame,
    invalid_name: object,
) -> None:
    """Reject blank and non-string fitted model names"""

    def fitter(_ticker: str, _history: pd.DataFrame) -> ForecastModel:
        return cast(
            ForecastModel,
            RuntimeModel(invalid_name, lambda rows: aligned_predictions(rows)),
        )

    with pytest.raises(ValueError, match="model name"):
        build_walk_forward_predictions(model_dataset, fitter)


def test_build_walk_forward_predictions_rejects_changing_model_names(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject model names changing between validation dates"""
    calls = 0

    def fitter(_ticker: str, _history: pd.DataFrame) -> ForecastModel:
        nonlocal calls
        calls += 1
        name = "stable" if calls == 1 else "changed"
        return cast(ForecastModel, RuntimeModel(name, aligned_predictions))

    with pytest.raises(ValueError, match="model name.*change"):
        build_walk_forward_predictions(model_dataset, fitter)


def test_build_walk_forward_predictions_rejects_name_change_during_prediction(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject model names changing during prediction"""

    class ChangingNameModel:
        """Change the model name after producing a prediction."""

        def __init__(self) -> None:
            self.predicted = False

        @property
        def name(self) -> str:
            """Return a name dependent on prediction state"""
            return "changed" if self.predicted else "stable"

        def predict(self, observations: pd.DataFrame) -> pd.Series:
            """Change state and return aligned predictions"""
            self.predicted = True
            return aligned_predictions(observations)

    def fitter(_ticker: str, _history: pd.DataFrame) -> ForecastModel:
        return ChangingNameModel()

    with pytest.raises(ValueError, match="model name.*change"):
        build_walk_forward_predictions(model_dataset, fitter)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda rows: [0.0] * len(rows), "Series"),
        (lambda rows: aligned_predictions(rows, name="forecast"), "name"),
        (
            lambda rows: pd.Series(
                [0.0, 0.0],
                index=pd.RangeIndex(2),
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
def test_build_walk_forward_predictions_rejects_invalid_outputs(
    model_dataset: pd.DataFrame,
    factory: PredictionFactory,
    message: str,
) -> None:
    """Reject malformed or nonfinite one-step predictions"""
    one_validation = model_dataset.loc[
        model_dataset["split"].ne("validation")
        | (model_dataset["ticker"].eq("AAA") & model_dataset["date"].eq(pd.Timestamp("2016-01-04")))
    ]

    def fitter(_ticker: str, _history: pd.DataFrame) -> ForecastModel:
        return cast(ForecastModel, RuntimeModel("invalid_output", factory))

    with pytest.raises(ValueError, match=message):
        build_walk_forward_predictions(one_validation, fitter)


def test_build_walk_forward_predictions_adds_failure_context(
    model_dataset: pd.DataFrame,
) -> None:
    """Add ticker and origin date to fitting failures"""
    calls: list[tuple[str, pd.Timestamp]] = []

    def fitter(ticker: str, history: pd.DataFrame) -> ForecastModel:
        origin_date = cast(pd.Timestamp, history["date"].max())
        calls.append((ticker, origin_date))
        if len(calls) == 2:
            raise ValueError("ARIMA order (2, 0, 1) did not converge")
        return cast(ForecastModel, RuntimeModel("arima", aligned_predictions))

    result: pd.DataFrame | None = None
    with pytest.raises(ValueError, match=r"AAA.*2016-01-05.*\(2, 0, 1\)"):
        result = build_walk_forward_predictions(model_dataset, fitter)

    assert result is None
    assert calls == [
        ("AAA", pd.Timestamp("2016-01-04")),
        ("AAA", pd.Timestamp("2016-01-05")),
    ]
