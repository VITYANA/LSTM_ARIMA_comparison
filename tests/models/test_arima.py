"""Tests for fitted ARIMA forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import cast

import numpy as np
import pandas as pd
import pytest

import models.arima as arima_module
from models.arima import (
    ARIMAFitter,
    ARIMAModel,
    ARIMAOrder,
    build_arima_fitters,
    search_arima_orders,
    shortlist_arima_orders,
)
from models.interfaces import ForecastModel


@dataclass
class FakeResult:
    """Provide controlled forecasts and convergence metadata."""

    values: list[object]
    converged: bool = True
    aic: float = 1.0
    bic: float = 2.0

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


@pytest.fixture
def model_dataset() -> pd.DataFrame:
    """Return two-ticker train validation and test rows"""
    rows: list[dict[str, object]] = []
    split_values = {
        "AAA": {"train": [0.01, 0.02], "validation": [0.31], "test": [0.41]},
        "BBB": {"train": [-0.01, -0.02], "validation": [-0.31], "test": [-0.41]},
    }
    day = 1
    for ticker, by_split in split_values.items():
        for split, values in by_split.items():
            for value in values:
                rows.append(
                    {
                        "date": pd.Timestamp(2020, 1, day),
                        "target_date": pd.Timestamp(2020, 1, day + 1),
                        "ticker": ticker,
                        "adj_close": 100.0 + day,
                        "log_return": value,
                        "target_return": 10.0 + day,
                        "split": split,
                    }
                )
                day += 1
    return pd.DataFrame(rows)


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
    ],
)
def test_arima_model_fit_rejects_invalid_orders(
    history: pd.DataFrame,
    invalid_order: tuple[object, ...],
) -> None:
    """Reject malformed or differenced ARIMA orders"""
    with pytest.raises(ValueError, match="order"):
        ARIMAModel.fit(history, cast(ARIMAOrder, invalid_order))


def test_arima_model_fit_accepts_nonnegative_differencing(
    monkeypatch: pytest.MonkeyPatch,
    history: pd.DataFrame,
) -> None:
    """Accept configurable nonnegative differencing orders"""
    received: list[ARIMAOrder] = []

    class DifferencedARIMA:
        """Capture a valid differenced model order."""

        def __init__(
            self,
            _values: np.ndarray,
            *,
            order: ARIMAOrder,
            trend: str,
        ) -> None:
            assert trend == "n"
            received.append(order)

        def fit(self) -> FakeResult:
            """Return a converged fitted result"""
            return FakeResult([0.0])

    monkeypatch.setattr(arima_module, "StatsmodelsARIMA", DifferencedARIMA)

    ARIMAModel.fit(history, (1, 1, 0))

    assert received == [(1, 1, 0)]


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
    fit_calls = 0

    class NonConvergedARIMA:
        """Return a controlled non-converged fit."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fit(self, **_kwargs: object) -> FakeResult:
            """Return non-converged optimizer metadata"""
            nonlocal fit_calls
            fit_calls += 1
            return FakeResult([0.0], converged=False)

    monkeypatch.setattr(arima_module, "StatsmodelsARIMA", NonConvergedARIMA)

    with pytest.raises(ValueError, match=r"\(1, 0, 2\).*converge"):
        ARIMAModel.fit(history, (1, 0, 2))
    assert fit_calls == 2


def test_arima_model_fit_retries_non_convergence_with_powell(
    monkeypatch: pytest.MonkeyPatch,
    history: pd.DataFrame,
) -> None:
    """Retry transient optimizer non-convergence with Powell"""
    fit_options: list[dict[str, str] | None] = []

    class RetryARIMA:
        """Converge only when fitted with the fallback optimizer."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fit(
            self,
            *,
            method_kwargs: dict[str, str] | None = None,
        ) -> FakeResult:
            """Record optimizer settings and return controlled results"""
            fit_options.append(method_kwargs)
            return FakeResult([0.025], converged=method_kwargs == {"method": "powell"})

    monkeypatch.setattr(arima_module, "StatsmodelsARIMA", RetryARIMA)

    model = ARIMAModel.fit(history, (0, 0, 0))
    result = model.predict(pd.DataFrame(index=[7]))

    assert result.tolist() == [0.025]
    assert fit_options == [None, {"method": "powell"}]


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


def test_search_arima_orders_uses_complete_grid_and_train_only(
    monkeypatch: pytest.MonkeyPatch,
    model_dataset: pd.DataFrame,
) -> None:
    """Search complete grid using isolated ticker train returns"""
    received: list[tuple[tuple[float, ...], ARIMAOrder, str]] = []

    class SearchARIMA:
        """Capture candidate inputs and return valid criteria."""

        def __init__(
            self,
            values: np.ndarray,
            *,
            order: ARIMAOrder,
            trend: str,
        ) -> None:
            received.append((tuple(values.tolist()), order, trend))
            self._order = order

        def fit(self) -> FakeResult:
            """Return deterministic finite candidate criteria"""
            p, _, q = self._order
            return FakeResult([], aic=float(p + q), bic=float(10 * p + q))

    monkeypatch.setattr(arima_module, "StatsmodelsARIMA", SearchARIMA)
    original = model_dataset.copy(deep=True)

    result = search_arima_orders(model_dataset)

    pd.testing.assert_frame_equal(model_dataset, original)
    assert len(result) == 72
    assert result.columns.tolist() == [
        "ticker",
        "p",
        "d",
        "q",
        "aic",
        "bic",
        "converged",
        "status",
    ]
    for ticker in ("AAA", "BBB"):
        ticker_rows = result.loc[result["ticker"].eq(ticker)]
        assert set(zip(ticker_rows["p"], ticker_rows["q"], strict=True)) == set(
            product(range(6), repeat=2)
        )
    assert result["d"].eq(0).all()
    assert {entry[2] for entry in received} == {"c"}
    supplied_histories = [entry[0] for entry in received]
    assert supplied_histories.count((0.01, 0.02)) == 36
    assert supplied_histories.count((-0.01, -0.02)) == 36


def test_search_arima_orders_supports_configurable_d_values(
    monkeypatch: pytest.MonkeyPatch,
    model_dataset: pd.DataFrame,
) -> None:
    """Search every explicitly configured differencing order"""
    received: list[tuple[ARIMAOrder, str]] = []

    class DifferencingSearchARIMA:
        """Capture candidate orders across differencing values."""

        def __init__(
            self,
            _values: np.ndarray,
            *,
            order: ARIMAOrder,
            trend: str,
        ) -> None:
            received.append((order, trend))

        def fit(self) -> FakeResult:
            """Return valid finite selection criteria"""
            return FakeResult([])

    monkeypatch.setattr(arima_module, "StatsmodelsARIMA", DifferencingSearchARIMA)
    one_ticker = model_dataset.loc[model_dataset["ticker"].eq("AAA")]

    result = search_arima_orders(
        one_ticker,
        p_values=(0,),
        d_values=(0, 1),
        q_values=(0,),
    )

    assert result["d"].tolist() == [0, 1]
    assert received == [((0, 0, 0), "c"), ((0, 1, 0), "n")]


@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_converged"),
    [
        ("ok", "ok", True),
        ("exception", "fit_failed", False),
        ("not_converged", "not_converged", False),
        ("nonfinite_aic", "nonfinite_aic", True),
        ("nonfinite_bic", "nonfinite_bic", True),
    ],
)
def test_search_arima_orders_records_candidate_status(
    monkeypatch: pytest.MonkeyPatch,
    model_dataset: pd.DataFrame,
    outcome: str,
    expected_status: str,
    expected_converged: bool,
) -> None:
    """Record each candidate fitting outcome without aborting"""

    class StatusARIMA:
        """Return the requested candidate fitting outcome."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fit(self) -> FakeResult:
            """Return or raise the configured fitting outcome"""
            if outcome == "exception":
                raise RuntimeError("controlled failure")
            return FakeResult(
                [],
                converged=outcome != "not_converged",
                aic=np.inf if outcome == "nonfinite_aic" else 1.0,
                bic=np.nan if outcome == "nonfinite_bic" else 2.0,
            )

    monkeypatch.setattr(arima_module, "StatsmodelsARIMA", StatusARIMA)
    one_ticker = model_dataset.loc[model_dataset["ticker"].eq("AAA")]

    result = search_arima_orders(one_ticker, p_values=(1,), q_values=(2,))

    assert len(result) == 1
    assert result.loc[0, "status"] == expected_status
    assert bool(result.loc[0, "converged"]) is expected_converged


def test_search_arima_orders_propagates_dataset_contract_failures(
    model_dataset: pd.DataFrame,
) -> None:
    """Reject invalid datasets before fitting any candidates"""
    invalid = model_dataset.drop(columns="target_return")

    with pytest.raises(ValueError, match="target_return"):
        search_arima_orders(invalid, p_values=(0,), q_values=(0,))


def test_shortlist_arima_orders_ranks_only_valid_candidates() -> None:
    """Rank valid candidates deterministically by raw BIC"""
    search_results = pd.DataFrame(
        [
            ("BBB", 2, 0, 0, 5.0, 4.0, True, "ok"),
            ("AAA", 2, 0, 0, 5.0, 3.0, True, "ok"),
            ("AAA", 1, 0, 1, 5.0, 1.0, True, "ok"),
            ("BBB", 0, 0, 2, 5.0, 2.0, True, "ok"),
            ("AAA", 0, 0, 2, 5.0, 1.0, True, "ok"),
            ("BBB", 1, 0, 1, 5.0, 3.0, True, "ok"),
            ("AAA", 5, 0, 5, 0.0, 0.0, False, "not_converged"),
            ("BBB", 5, 0, 5, 0.0, 0.0, False, "fit_failed"),
        ],
        columns=["ticker", "p", "d", "q", "aic", "bic", "converged", "status"],
    )

    result = shortlist_arima_orders(search_results, size=3)

    assert result["ticker"].tolist() == ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"]
    assert result["bic_rank"].tolist() == [1, 2, 3, 1, 2, 3]
    assert result["model"].tolist() == [
        "arima_bic_1",
        "arima_bic_2",
        "arima_bic_3",
        "arima_bic_1",
        "arima_bic_2",
        "arima_bic_3",
    ]
    assert list(zip(result["p"], result["q"], strict=True)) == [
        (0, 2),
        (1, 1),
        (2, 0),
        (0, 2),
        (1, 1),
        (2, 0),
    ]
    assert result["status"].eq("ok").all()


def test_shortlist_arima_orders_rejects_insufficient_candidates() -> None:
    """Reject tickers lacking requested valid candidate count"""
    search_results = pd.DataFrame(
        [("AAA", 0, 0, 0, 1.0, 1.0, True, "ok")],
        columns=["ticker", "p", "d", "q", "aic", "bic", "converged", "status"],
    )

    with pytest.raises(ValueError, match="AAA.*3"):
        shortlist_arima_orders(search_results, size=3)


@pytest.mark.parametrize("invalid_column", ["aic", "bic"])
def test_shortlist_arima_orders_rejects_nonnumeric_criteria(
    invalid_column: str,
) -> None:
    """Reject nonnumeric model selection criteria"""
    search_results = pd.DataFrame(
        [
            ("AAA", 0, 0, 0, 1.0, 1.0, True, "ok"),
            ("AAA", 1, 0, 0, 2.0, 2.0, True, "ok"),
            ("AAA", 2, 0, 0, 3.0, 3.0, True, "ok"),
        ],
        columns=["ticker", "p", "d", "q", "aic", "bic", "converged", "status"],
    ).astype({"aic": "object", "bic": "object"})
    search_results.loc[0, invalid_column] = "invalid"

    with pytest.raises(ValueError, match="criteria.*numeric"):
        shortlist_arima_orders(search_results)


@pytest.mark.parametrize("invalid_size", [0, -1, 1.5, True])
def test_shortlist_arima_orders_rejects_invalid_size(invalid_size: object) -> None:
    """Reject non-positive or non-integer shortlist sizes"""
    search_results = pd.DataFrame(
        columns=["ticker", "p", "d", "q", "aic", "bic", "converged", "status"]
    )

    with pytest.raises(ValueError, match="size"):
        shortlist_arima_orders(search_results, size=cast(int, invalid_size))


def test_build_arima_fitters_assembles_ranked_ticker_orders(
    monkeypatch: pytest.MonkeyPatch,
    history: pd.DataFrame,
) -> None:
    """Build three fitters from ticker BIC ranks"""
    shortlist = pd.DataFrame(
        [
            ("AAA", 1, 0, 0, 1, "arima_bic_1"),
            ("AAA", 2, 0, 0, 2, "arima_bic_2"),
            ("AAA", 3, 0, 0, 3, "arima_bic_3"),
            ("BBB", 0, 0, 1, 1, "arima_bic_1"),
            ("BBB", 0, 0, 2, 2, "arima_bic_2"),
            ("BBB", 0, 0, 3, 3, "arima_bic_3"),
        ],
        columns=["ticker", "p", "d", "q", "bic_rank", "model"],
    )
    received: list[tuple[ARIMAOrder, str]] = []

    def fake_fit(
        cls: type[ARIMAModel],
        _history: pd.DataFrame,
        order: ARIMAOrder,
        *,
        name: str = "arima",
    ) -> ARIMAModel:
        received.append((order, name))
        return cls(FakeResult([0.0]), name=name)

    monkeypatch.setattr(ARIMAModel, "fit", classmethod(fake_fit))

    fitters = build_arima_fitters(shortlist)
    for model_name, fitter in fitters.items():
        fitter("AAA", history)
        fitter("BBB", history)
        assert model_name.startswith("arima_bic_")

    assert list(fitters) == ["arima_bic_1", "arima_bic_2", "arima_bic_3"]
    assert received == [
        ((1, 0, 0), "arima_bic_1"),
        ((0, 0, 1), "arima_bic_1"),
        ((2, 0, 0), "arima_bic_2"),
        ((0, 0, 2), "arima_bic_2"),
        ((3, 0, 0), "arima_bic_3"),
        ((0, 0, 3), "arima_bic_3"),
    ]


@pytest.mark.parametrize(
    ("rows", "columns", "message"),
    [
        ([], ["ticker", "p", "d", "q", "bic_rank"], "model"),
        (
            [("AAA", 1, 0, 0, 1, "arima_bic_1")],
            ["ticker", "p", "d", "q", "bic_rank", "model"],
            "ranks",
        ),
        (
            [
                ("AAA", 1, 0, 0, 1, "arima_bic_1"),
                ("AAA", 2, 0, 0, 1, "arima_bic_1"),
            ],
            ["ticker", "p", "d", "q", "bic_rank", "model"],
            "duplicate",
        ),
        (
            [
                ("AAA", 1, 0, 0, 1, "arima_bic_1"),
                ("AAA", 2, 0, 0, 2, "arima_bic_2"),
                ("AAA", 3, 0, 0, 3, "arima_bic_3"),
                ("BBB", 0, 0, 1, 1, "arima_bic_1"),
                ("BBB", 0, 0, 2, 2, "arima_bic_2"),
            ],
            ["ticker", "p", "d", "q", "bic_rank", "model"],
            "every.*rank",
        ),
        (
            [
                ("AAA", 1, 0, 0, 1, "arima_bic_1"),
                ("AAA", 2, 0, 0, 2, "wrong_name"),
                ("AAA", 3, 0, 0, 3, "arima_bic_3"),
            ],
            ["ticker", "p", "d", "q", "bic_rank", "model"],
            "model name",
        ),
    ],
)
def test_build_arima_fitters_rejects_malformed_shortlists(
    rows: list[tuple[object, ...]],
    columns: list[str],
    message: str,
) -> None:
    """Reject malformed duplicate or incomplete shortlist mappings"""
    shortlist = pd.DataFrame(rows, columns=columns)

    with pytest.raises(ValueError, match=message):
        build_arima_fitters(shortlist)
