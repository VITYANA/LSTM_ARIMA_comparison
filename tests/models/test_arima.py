"""Tests for fitted ARIMA forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import pytest

import models.arima as arima_module
from models.arima import ARIMAFitter, ARIMAModel, ARIMAOrder
from models.interfaces import ForecastModel


@dataclass
class FakeResult:
    """Provide controlled forecasts and convergence metadata."""

    values: list[object]
    converged: bool = True

    @property
    def mle_retvals(self) -> dict[str, bool]:
        """Expose optimizer convergence metadata"""
        return {"converged": self.converged}

    def forecast(self, steps: int) -> np.ndarray:
        """Return the requested controlled forecast values"""
        return np.asarray(self.values[:steps])


@pytest.fixture
def history() -> pd.DataFrame:
    """Return finite stationary-looking model history"""
    return pd.DataFrame({"log_return": [0.01, -0.02, 0.015, -0.005, 0.012]})


@pytest.mark.parametrize(
    ("index", "values"),
    [
        (pd.Index([], name="observation"), []),
        (pd.Index([7, 2, 11], name="observation"), [0.01, -0.02, 0.03]),
    ],
)
def test_arima_model_returns_aligned_float_forecasts(
    index: pd.Index,
    values: list[float],
) -> None:
    """Return aligned finite forecasts without mutating observations"""
    observations = pd.DataFrame({"log_return": [0.0] * len(index)}, index=index)
    original = observations.copy(deep=True)
    model: ForecastModel = ARIMAModel(FakeResult(cast(list[object], values)))

    result = model.predict(observations)

    pd.testing.assert_frame_equal(observations, original)
    assert model.name == "arima"
    assert result.name == "predicted_return"
    assert result.index.equals(index)
    assert result.dtype == "float64"
    assert result.tolist() == values


@pytest.mark.parametrize("invalid_name", [None, 7, "", "   "])
def test_arima_model_rejects_invalid_names(invalid_name: object) -> None:
    """Reject empty and non-string model identifiers"""
    with pytest.raises(ValueError, match="model name"):
        ARIMAModel(FakeResult([0.0]), name=cast(str, invalid_name))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.0], "length"),
        (["invalid", "values"], "numeric"),
        ([np.nan, 0.0], "finite"),
        ([np.inf, 0.0], "finite"),
    ],
)
def test_arima_model_rejects_invalid_forecasts(
    values: list[object],
    message: str,
) -> None:
    """Reject malformed forecasts from fitted results"""
    model = ARIMAModel(FakeResult(values))
    observations = pd.DataFrame(index=pd.Index([3, 5]))

    with pytest.raises(ValueError, match=message):
        model.predict(observations)


@pytest.mark.parametrize(
    "invalid_order",
    [
        (1, 0),
        (1, 0, 1, 2),
        (True, 0, 1),
        (1.5, 0, 1),
        (-1, 0, 1),
        (1, -1, 1),
        (1, 0, -1),
        (1, 1, 1),
    ],
)
def test_arima_model_fit_rejects_invalid_orders(
    history: pd.DataFrame,
    invalid_order: tuple[object, ...],
) -> None:
    """Reject malformed or differenced ARIMA orders"""
    with pytest.raises(ValueError, match="order"):
        ARIMAModel.fit(history, cast(ARIMAOrder, invalid_order))


@pytest.mark.parametrize(
    ("invalid_history", "message"),
    [
        (pd.DataFrame(), "log_return"),
        (pd.DataFrame({"log_return": []}), "empty"),
        (pd.DataFrame({"log_return": ["invalid"]}), "numeric"),
        (pd.DataFrame({"log_return": [np.nan]}), "finite"),
        (pd.DataFrame({"log_return": [np.inf]}), "finite"),
        (pd.DataFrame({"log_return": [-np.inf]}), "finite"),
    ],
)
def test_arima_model_fit_rejects_invalid_history(
    invalid_history: pd.DataFrame,
    message: str,
) -> None:
    """Reject missing empty or invalid return histories"""
    with pytest.raises(ValueError, match=message):
        ARIMAModel.fit(invalid_history, (1, 0, 0))


def test_arima_model_fit_rejects_non_converged_result(
    monkeypatch: pytest.MonkeyPatch,
    history: pd.DataFrame,
) -> None:
    """Reject optimizer results that did not converge"""

    class NonConvergedARIMA:
        """Return a controlled non-converged fit."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fit(self) -> FakeResult:
            """Return non-converged optimizer metadata"""
            return FakeResult([0.0], converged=False)

    monkeypatch.setattr(arima_module, "StatsmodelsARIMA", NonConvergedARIMA)

    with pytest.raises(ValueError, match=r"\(1, 0, 2\).*converge"):
        ARIMAModel.fit(history, (1, 0, 2))


def test_arima_model_fit_wraps_failure_with_order(
    monkeypatch: pytest.MonkeyPatch,
    history: pd.DataFrame,
) -> None:
    """Add attempted order to fitting exceptions"""

    class FailingARIMA:
        """Raise a controlled fitting failure."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fit(self) -> FakeResult:
            """Raise the controlled optimizer error"""
            raise RuntimeError("optimizer failed")

    monkeypatch.setattr(arima_module, "StatsmodelsARIMA", FailingARIMA)

    with pytest.raises(ValueError, match=r"\(2, 0, 1\).*optimizer failed"):
        ARIMAModel.fit(history, (2, 0, 1))


def test_arima_fitter_selects_ticker_order_and_copies_mapping(
    monkeypatch: pytest.MonkeyPatch,
    history: pd.DataFrame,
) -> None:
    """Use fixed ticker order from isolated mapping"""
    received: list[tuple[ARIMAOrder, str]] = []

    def fake_fit(
        cls: type[ARIMAModel],
        observations: pd.DataFrame,
        order: ARIMAOrder,
        *,
        name: str = "arima",
    ) -> ARIMAModel:
        pd.testing.assert_frame_equal(observations, history)
        received.append((order, name))
        return cls(FakeResult([0.0]), name=name)

    monkeypatch.setattr(ARIMAModel, "fit", classmethod(fake_fit))
    orders: dict[str, ARIMAOrder] = {"AAA": (1, 0, 2)}
    fitter = ARIMAFitter(orders, model_name="arima_bic_1")
    orders["AAA"] = (5, 0, 5)

    model = fitter("AAA", history)

    assert model.name == "arima_bic_1"
    assert received == [((1, 0, 2), "arima_bic_1")]


def test_arima_fitter_rejects_unknown_ticker(history: pd.DataFrame) -> None:
    """Reject tickers without a configured order"""
    fitter = ARIMAFitter({"AAA": (1, 0, 0)})

    with pytest.raises(ValueError, match="BBB.*order"):
        fitter("BBB", history)


def test_arima_model_fit_produces_finite_real_forecast() -> None:
    """Fit statsmodels and produce finite forecast"""
    random = np.random.default_rng(42)
    innovations = random.normal(scale=0.01, size=120)
    returns = np.empty_like(innovations)
    returns[0] = innovations[0]
    for position in range(1, len(returns)):
        returns[position] = 0.4 * returns[position - 1] + innovations[position]
    history = pd.DataFrame({"log_return": returns})
    observations = pd.DataFrame(index=pd.Index([120], name="observation"))

    model = ARIMAModel.fit(history, (1, 0, 0))
    result = model.predict(observations)

    assert result.index.equals(observations.index)
    assert np.isfinite(result.iloc[0])
