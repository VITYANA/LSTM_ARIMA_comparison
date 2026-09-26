"""Select ticker-level ARIMA candidates using validation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.metrics import METRIC_COLUMNS, calculate_metrics
from models.interfaces import validate_model_name

SHORTLIST_COLUMNS = ("ticker", "p", "d", "q", "bic", "bic_rank", "model")
CANDIDATE_METRIC_COLUMNS = (*METRIC_COLUMNS, "p", "d", "q", "bic", "bic_rank")
ORDER_COLUMNS = ("ticker", "p", "d", "q", "bic", "bic_rank", "model")
EXPECTED_BIC_RANKS = frozenset({1, 2, 3})
VALIDATION_SPLIT = "validation"


@dataclass(frozen=True)
class ARIMASelection:
    """Contain candidate metrics, selected orders and unified predictions."""

    candidate_metrics: pd.DataFrame
    orders: pd.DataFrame
    predictions: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    label: str,
) -> None:
    """Require every column used by candidate selection."""
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _validate_shortlist(shortlist: pd.DataFrame, tickers: set[object]) -> set[str]:
    """Validate ticker-rank mappings and return candidate names."""
    _require_columns(shortlist, SHORTLIST_COLUMNS, "shortlist")
    if shortlist.empty:
        raise ValueError("shortlist must not be empty")
    if shortlist.duplicated(["ticker", "bic_rank"]).any():
        raise ValueError("shortlist contains duplicate ticker ranks")
    if set(shortlist["ticker"]) != tickers:
        raise ValueError("shortlist ticker mapping does not match predictions")
    for ticker, rows in shortlist.groupby("ticker"):
        if set(rows["bic_rank"]) != EXPECTED_BIC_RANKS:
            raise ValueError(f"ticker {ticker!r} must have BIC ranks 1, 2, and 3")
        expected_names = rows["bic_rank"].map(lambda rank: f"arima_bic_{rank}")
        if not rows["model"].equals(expected_names):
            raise ValueError(f"ticker {ticker!r} has inconsistent candidate model names")
    return set(shortlist["model"])


def select_arima_candidates(
    predictions: pd.DataFrame,
    shortlist: pd.DataFrame,
    baseline_model: str = "naive_zero",
    selected_model: str = "arima",
) -> ARIMASelection:
    """Select each ticker's validation winner and unify its predictions."""
    baseline_name = validate_model_name(baseline_model)
    selected_name = validate_model_name(selected_model)
    _require_columns(predictions, PREDICTION_COLUMNS, "predictions")
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    if not predictions["split"].eq(VALIDATION_SPLIT).all():
        raise ValueError("predictions must contain validation rows only")

    prediction_models = set(predictions["model"])
    if baseline_name not in prediction_models:
        raise ValueError(f"baseline model {baseline_name!r} is absent")
    baseline_tickers = set(predictions.loc[predictions["model"].eq(baseline_name), "ticker"])
    candidate_names = _validate_shortlist(shortlist, baseline_tickers)
    expected_models = {baseline_name, *candidate_names}
    if prediction_models != expected_models:
        raise ValueError("prediction models do not match baseline and shortlist models")

    metrics = calculate_metrics(predictions, baseline_model=baseline_name)
    candidate_metrics = metrics.loc[
        metrics["ticker"].ne("ALL") & metrics["model"].isin(candidate_names)
    ].merge(
        shortlist.loc[:, SHORTLIST_COLUMNS],
        on=["ticker", "model"],
        how="inner",
        validate="one_to_one",
    )
    candidate_metrics = candidate_metrics.loc[:, CANDIDATE_METRIC_COLUMNS].sort_values(
        ["ticker", "bic_rank"]
    )
    ranked = candidate_metrics.assign(
        complexity=(candidate_metrics["p"] + candidate_metrics["d"] + candidate_metrics["q"])
    )
    winners = (
        ranked.sort_values(["ticker", "mae_bps", "rmse_bps", "complexity", "bic", "p", "q"])
        .groupby("ticker", sort=True, as_index=False)
        .head(1)
    )

    orders = winners.loc[:, ORDER_COLUMNS].sort_values("ticker").reset_index(drop=True)
    selected_keys = orders.loc[:, ["ticker", "model"]]
    selected_predictions = predictions.merge(
        selected_keys,
        on=["ticker", "model"],
        how="inner",
        validate="many_to_one",
    )
    selected_predictions["model"] = selected_name
    selected_predictions = selected_predictions.loc[:, PREDICTION_COLUMNS].sort_values(
        ["ticker", "target_date", "date"]
    )

    return ARIMASelection(
        candidate_metrics=candidate_metrics.reset_index(drop=True),
        orders=orders,
        predictions=selected_predictions.reset_index(drop=True),
    )
