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


@pytest.fixture
def history_boundary_dataset() -> pd.DataFrame:
    """Return 253 train targets with consecutive business dates"""
    dates = pd.bdate_range("2010-01-04", periods=254)
    frame = pd.DataFrame(
        {
            "date": dates[:-1],
            "target_date": dates[1:],
            "ticker": "AAA",
            "adj_close": np.arange(100.0, 353.0),
            "log_return": np.arange(253, dtype="float64") / 10_000.0,
            "target_return": np.arange(1, 254, dtype="float64") / 10_000.0,
            "split": "train",
        }
    )
    return frame.loc[:, DATASET_COLUMNS]


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


def test_build_walk_forward_predictions_reports_completed_observations(
    model_dataset: pd.DataFrame,
) -> None:
    """Report progress after every completed validation forecast"""
    progress: list[tuple[int, int]] = []

    build_walk_forward_predictions(
        model_dataset,
        RecordingFitter(),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_walk_forward_defaults_to_existing_validation_behavior(
    model_dataset: pd.DataFrame,
) -> None:
    """Keep default scheduling equivalent to explicit validation"""
    default = build_walk_forward_predictions(model_dataset, RecordingFitter())
    explicit = build_walk_forward_predictions(
        model_dataset,
        RecordingFitter(),
        target_splits=("validation",),
    )

    pd.testing.assert_frame_equal(default, explicit)


def test_walk_forward_supports_train_and_validation_targets(
    model_dataset: pd.DataFrame,
) -> None:
    """Forecast requested evaluation splits without test leakage"""
    original = model_dataset.copy(deep=True)
    fitter = RecordingFitter()

    result = build_walk_forward_predictions(
        model_dataset,
        fitter,
        target_splits=("train", "validation"),
    )

    pd.testing.assert_frame_equal(model_dataset, original)
    assert result[["ticker", "split", "target_date"]].values.tolist() == [
        ["AAA", "train", pd.Timestamp("2010-01-05")],
        ["AAA", "validation", pd.Timestamp("2016-01-05")],
        ["AAA", "validation", pd.Timestamp("2016-01-06")],
        ["BBB", "train", pd.Timestamp("2010-01-05")],
        ["BBB", "validation", pd.Timestamp("2016-01-05")],
        ["BBB", "validation", pd.Timestamp("2016-01-06")],
    ]
    assert "test" not in result["split"].values
    assert all("test" not in history["split"].values for _, history in fitter.calls)


def test_walk_forward_handles_duplicate_dataframe_indexes(
    model_dataset: pd.DataFrame,
) -> None:
    """Schedule each observation once despite repeated index labels"""
    repeated_index = model_dataset.copy()
    repeated_index.index = [0, 1, 2, 0, 3, 0, 0, 4]

    result = build_walk_forward_predictions(repeated_index, RecordingFitter())

    assert result[["ticker", "target_date"]].values.tolist() == [
        ["AAA", pd.Timestamp("2016-01-05")],
        ["AAA", pd.Timestamp("2016-01-06")],
        ["BBB", pd.Timestamp("2016-01-05")],
        ["BBB", pd.Timestamp("2016-01-06")],
    ]


def test_walk_forward_uses_exact_minimum_history_boundary(
    history_boundary_dataset: pd.DataFrame,
) -> None:
    """Start forecasting when exactly 252 returns are available"""
    fitter = RecordingFitter()

    result = build_walk_forward_predictions(
        history_boundary_dataset,
        fitter,
        target_splits=("train",),
        min_history=252,
    )

    assert result["date"].tolist() == [
        history_boundary_dataset.iloc[251]["date"],
        history_boundary_dataset.iloc[252]["date"],
    ]
    assert [len(history) for _, history in fitter.calls] == [252, 253]
    for (_, history), observation in zip(
        fitter.calls,
        fitter.observations,
        strict=True,
    ):
        assert history["date"].max() <= observation["date"].iloc[0]


def test_walk_forward_progress_counts_only_eligible_targets(
    history_boundary_dataset: pd.DataFrame,
) -> None:
    """Report progress against history-eligible forecast targets"""
    progress: list[tuple[int, int]] = []

    build_walk_forward_predictions(
        history_boundary_dataset,
        RecordingFitter(),
        target_splits=("train",),
        min_history=252,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert progress == [(1, 2), (2, 2)]


@pytest.mark.parametrize(
    "target_splits",
    [
        (),
        ("validation", "validation"),
        ("test",),
        ("validation", 1),
        "validation",
    ],
)
def test_walk_forward_rejects_invalid_target_splits(
    model_dataset: pd.DataFrame,
    target_splits: object,
) -> None:
    """Reject empty duplicate unsupported and malformed schedules"""
    with pytest.raises(ValueError, match="target_splits"):
        build_walk_forward_predictions(
            model_dataset,
            RecordingFitter(),
            target_splits=cast(tuple[str, ...], target_splits),
        )


@pytest.mark.parametrize("min_history", [True, False, 0, -1, 1.5, "2", None])
def test_walk_forward_rejects_invalid_minimum_history(
    model_dataset: pd.DataFrame,
    min_history: object,
) -> None:
    """Reject minimum history values outside positive integers"""
    with pytest.raises(ValueError, match="min_history"):
        build_walk_forward_predictions(
            model_dataset,
            RecordingFitter(),
            min_history=cast(int, min_history),
        )


def test_walk_forward_rejects_noncallable_progress(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject progress values that cannot receive updates"""
    with pytest.raises(ValueError, match="progress"):
        build_walk_forward_predictions(
            model_dataset,
            RecordingFitter(),
            progress=cast(Callable[[int, int], None], 1),
        )


def test_walk_forward_rejects_schedule_without_eligible_targets(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject schedules fully removed by history requirements"""
    with pytest.raises(ValueError, match="eligible"):
        build_walk_forward_predictions(
            model_dataset,
            RecordingFitter(),
            target_splits=("validation",),
            min_history=100,
        )


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
