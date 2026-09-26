"""Fitted ARIMA forecasting models and fitters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, Self, cast

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA  # type: ignore[import-untyped]

from evaluation.contracts import validate_model_dataset

ARIMAOrder = tuple[int, int, int]
PREDICTION_NAME = "predicted_return"
RETURN_COLUMN = "log_return"
SEARCH_COLUMNS = (
    "ticker",
    "p",
    "d",
    "q",
    "aic",
    "bic",
    "converged",
    "status",
)
SHORTLIST_COLUMNS = (*SEARCH_COLUMNS, "bic_rank", "model")


class _ARIMAResult(Protocol):
    """Minimal fitted-result surface required for forecasting."""

    @property
    def mle_retvals(self) -> Mapping[str, object]:
        """Return optimizer metadata."""
        ...

    def forecast(self, steps: int) -> object:
        """Return forecasts for the requested horizon."""
        ...


class _ARIMACandidateResult(Protocol):
    """Minimal fitted-result surface required for model selection."""

    @property
    def aic(self) -> float:
        """Return the Akaike information criterion."""
        ...

    @property
    def bic(self) -> float:
        """Return the Bayesian information criterion."""
        ...

    @property
    def mle_retvals(self) -> Mapping[str, object]:
        """Return optimizer metadata."""
        ...


def _validate_model_name(name: object) -> str:
    """Return a non-empty model identifier."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name must be a non-empty string")
    return name


def _validate_order(order: object) -> ARIMAOrder:
    """Return a valid nonnegative ARIMA order."""
    if not isinstance(order, tuple) or len(order) != 3:
        raise ValueError("ARIMA order must contain exactly three integers")
    if any(type(value) is not int for value in order):
        raise ValueError("ARIMA order values must be integers")
    p, d, q = cast(ARIMAOrder, order)
    if p < 0 or d < 0 or q < 0:
        raise ValueError("ARIMA order values must be non-negative")
    return p, d, q


def _trend_for_order(order: ARIMAOrder) -> str:
    """Return a statsmodels-compatible deterministic trend."""
    return "c" if order[1] == 0 else "n"


def _return_values(history: pd.DataFrame) -> np.ndarray:
    """Extract a finite numeric return history."""
    if RETURN_COLUMN not in history.columns:
        raise ValueError(f"history missing column {RETURN_COLUMN!r}")
    if history.empty:
        raise ValueError("history must not be empty")
    returns = history[RETURN_COLUMN]
    if returns.dtype.kind not in "iuf":
        raise ValueError(f"{RETURN_COLUMN} must be numeric")
    values = returns.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError(f"{RETURN_COLUMN} must contain only finite values")
    return values


class ARIMAModel:
    """Adapt one fitted statsmodels result to ForecastModel."""

    def __init__(self, fitted_result: _ARIMAResult, *, name: str = "arima") -> None:
        self._fitted_result = fitted_result
        self._name = _validate_model_name(name)

    @classmethod
    def fit(
        cls,
        history: pd.DataFrame,
        order: ARIMAOrder,
        *,
        name: str = "arima",
    ) -> Self:
        """Fit and return one stationary ARIMA model."""
        validated_order = _validate_order(order)
        validated_name = _validate_model_name(name)
        values = _return_values(history)

        try:
            fitted = StatsmodelsARIMA(
                values,
                order=validated_order,
                trend=_trend_for_order(validated_order),
            ).fit()
        except Exception as error:
            raise ValueError(f"ARIMA order {validated_order} fit failed: {error}") from error

        result = cast(_ARIMAResult, fitted)
        if result.mle_retvals.get("converged") is not True:
            raise ValueError(f"ARIMA order {validated_order} did not converge")
        return cls(result, name=validated_name)

    @property
    def name(self) -> str:
        """Return the stable model identifier."""
        return self._name

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Forecast returns aligned with observation rows."""
        if len(observations) == 0:
            return pd.Series(
                index=observations.index,
                dtype="float64",
                name=PREDICTION_NAME,
            )

        forecast = np.asarray(self._fitted_result.forecast(steps=len(observations)))
        values = forecast.reshape(-1)
        if len(values) != len(observations):
            raise ValueError("forecast length must match observations")
        if values.dtype.kind not in "iuf":
            raise ValueError("forecast values must be numeric")
        numeric = values.astype("float64")
        if not np.isfinite(numeric).all():
            raise ValueError("forecast values must contain only finite values")
        return pd.Series(
            numeric,
            index=observations.index,
            dtype="float64",
            name=PREDICTION_NAME,
        )


class ARIMAFitter:
    """Fit ticker-specific fixed ARIMA orders."""

    def __init__(
        self,
        orders: Mapping[str, ARIMAOrder],
        model_name: str = "arima",
    ) -> None:
        self._orders = {ticker: _validate_order(order) for ticker, order in orders.items()}
        self._model_name = _validate_model_name(model_name)

    def __call__(self, ticker: str, history: pd.DataFrame) -> ARIMAModel:
        """Fit the configured order for one ticker."""
        try:
            order = self._orders[ticker]
        except KeyError as error:
            raise ValueError(f"ticker {ticker!r} has no configured ARIMA order") from error
        return ARIMAModel.fit(history, order, name=self._model_name)


def _fit_candidate(values: np.ndarray, order: ARIMAOrder) -> dict[str, object]:
    """Fit one search candidate and describe its outcome."""
    try:
        fitted = StatsmodelsARIMA(
            values,
            order=order,
            trend=_trend_for_order(order),
        ).fit()
        result = cast(_ARIMACandidateResult, fitted)
        aic = float(result.aic)
        bic = float(result.bic)
        converged = result.mle_retvals.get("converged") is True
    except Exception:
        return {
            "aic": np.nan,
            "bic": np.nan,
            "converged": False,
            "status": "fit_failed",
        }

    if not converged:
        status = "not_converged"
    elif not np.isfinite(aic):
        status = "nonfinite_aic"
    elif not np.isfinite(bic):
        status = "nonfinite_bic"
    else:
        status = "ok"
    return {
        "aic": aic,
        "bic": bic,
        "converged": converged,
        "status": status,
    }


def search_arima_orders(
    dataset: pd.DataFrame,
    p_values: Sequence[int] = tuple(range(6)),
    d_values: Sequence[int] = tuple(range(1)),
    q_values: Sequence[int] = tuple(range(6)),
) -> pd.DataFrame:
    """Evaluate stationary ARIMA orders on each ticker's train returns."""
    validate_model_dataset(dataset)
    rows: list[dict[str, object]] = []

    for ticker in sorted(dataset["ticker"].unique()):
        train = dataset.loc[dataset["ticker"].eq(ticker) & dataset["split"].eq("train")]
        values = _return_values(train)
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = _validate_order((p, d, q))
                    rows.append(
                        {
                            "ticker": ticker,
                            "p": p,
                            "d": d,
                            "q": q,
                            **_fit_candidate(values, order),
                        }
                    )

    result = pd.DataFrame(rows, columns=SEARCH_COLUMNS)
    return result.sort_values(["ticker", "p", "d", "q"]).reset_index(drop=True)


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    label: str,
) -> None:
    """Require all columns used by an ARIMA table."""
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def shortlist_arima_orders(
    search_results: pd.DataFrame,
    size: int = 3,
) -> pd.DataFrame:
    """Return each ticker's best valid candidates ranked by BIC."""
    if type(size) is not int or size <= 0:
        raise ValueError("shortlist size must be a positive integer")
    _require_columns(search_results, SEARCH_COLUMNS, "search results")

    valid = search_results.loc[
        search_results["status"].eq("ok") & search_results["converged"].eq(True)
    ].copy()
    if valid["aic"].dtype.kind not in "iuf" or valid["bic"].dtype.kind not in "iuf":
        raise ValueError("search criteria must be numeric")
    valid = valid.loc[np.isfinite(valid["aic"]) & np.isfinite(valid["bic"])]

    available = valid.groupby("ticker").size()
    for ticker in sorted(search_results["ticker"].unique()):
        if int(available.get(ticker, 0)) < size:
            raise ValueError(f"ticker {ticker!r} has fewer than {size} valid candidates")

    ranked = valid.sort_values(["ticker", "bic", "p", "q"])
    result = ranked.groupby("ticker", sort=True).head(size).copy()
    result["bic_rank"] = result.groupby("ticker", sort=False).cumcount() + 1
    result["model"] = "arima_bic_" + result["bic_rank"].astype(str)
    return cast(pd.DataFrame, result.loc[:, SHORTLIST_COLUMNS].reset_index(drop=True))


def build_arima_fitters(shortlist: pd.DataFrame) -> dict[str, ARIMAFitter]:
    """Build one ticker-aware fitter for each BIC rank."""
    required = ("ticker", "p", "d", "q", "bic_rank", "model")
    _require_columns(shortlist, required, "shortlist")
    if shortlist.duplicated(["ticker", "bic_rank"]).any():
        raise ValueError("shortlist contains duplicate ticker ranks")

    expected_ranks = {1, 2, 3}
    if set(shortlist["bic_rank"]) != expected_ranks:
        raise ValueError("shortlist ranks must be exactly 1, 2, and 3")

    tickers = set(shortlist["ticker"])
    fitters: dict[str, ARIMAFitter] = {}
    for rank in sorted(expected_ranks):
        ranked = shortlist.loc[shortlist["bic_rank"].eq(rank)]
        if set(ranked["ticker"]) != tickers:
            raise ValueError("each ticker must have every shortlist rank")
        model_name = f"arima_bic_{rank}"
        if not ranked["model"].eq(model_name).all():
            raise ValueError(f"rank {rank} must use model name {model_name!r}")
        orders = {
            str(row.ticker): _validate_order((row.p, row.d, row.q))
            for row in ranked.itertuples(index=False)
        }
        fitters[model_name] = ARIMAFitter(orders, model_name=model_name)
    return fitters
