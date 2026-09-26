"""Tests for the naive zero-return predictor."""

from __future__ import annotations

import pandas as pd
import pytest

from models.interfaces import ForecastModel
from models.naive import NaiveModel


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(3),
        pd.Index([7, 2, 11], name="observation"),
    ],
)
def test_naive_model_preserves_index_and_returns_float_zeros(
    index: pd.Index,
) -> None:
    """Preserve arbitrary indexes and return float zeros"""
    observations = pd.DataFrame(
        {"target_return": [0.15, -0.20, 0.03]},
        index=index,
    )
    original = observations.copy(deep=True)

    result = NaiveModel().predict(observations)

    pd.testing.assert_frame_equal(observations, original)
    assert result.index.equals(index)
    assert result.name == "predicted_return"
    assert result.dtype == "float64"
    assert result.tolist() == [0.0, 0.0, 0.0]


def test_naive_model_accepts_empty_frame() -> None:
    """Return a typed empty prediction series"""
    observations = pd.DataFrame(index=pd.Index([], name="observation"))

    result = NaiveModel().predict(observations)

    assert result.empty
    assert result.index.equals(observations.index)
    assert result.name == "predicted_return"
    assert result.dtype == "float64"


def test_naive_model_exposes_stable_name() -> None:
    """Expose baseline name through forecast protocol"""
    model: ForecastModel = NaiveModel()

    assert model.name == "naive_zero"
