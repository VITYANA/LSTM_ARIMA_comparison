"""Tests for independent final-test return and price metrics."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.final_metrics import (
    PRICE_PREDICTION_COLUMNS,
    build_price_predictions,
    calculate_final_price_metrics,
    calculate_final_return_metrics,
    calculate_seed_return_metrics,
)
from evaluation.final_models import FINAL_TEST_PROTOCOL, FinalSeedResult
from models.lstm import LSTMConfig

CORE_TICKERS = ("AAPL", "JPM", "SPY", "XOM")
MODELS = ("naive_zero", "arima", "lstm", "arima_lstm")


def source_dataset() -> pd.DataFrame:
    """Return a compact canonical dataset with price scales"""
    rows: list[dict[str, object]] = []
    schedules = {
        ticker: [
            ("2015-12-29", "2015-12-30", "train", np.log(1.10)),
            ("2019-12-30", "2019-12-31", "validation", np.log(0.95)),
            ("2019-12-31", "2020-01-02", "test", 0.01),
            ("2025-12-30", "2025-12-31", "test", -0.02),
        ]
        for ticker in CORE_TICKERS
    }
    schedules["JPM"].insert(3, ("2020-01-02", "2020-01-03", "test", 0.03))
    for ticker_position, ticker in enumerate(CORE_TICKERS):
        for row_position, (date, target_date, split, target_return) in enumerate(schedules[ticker]):
            adj_close = 100.0 + ticker_position * 10 + row_position
            rows.append(
                {
                    "date": date,
                    "target_date": target_date,
                    "ticker": ticker,
                    "adj_close": adj_close,
                    "log_return": target_return / 2,
                    "target_return": target_return,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def final_predictions() -> pd.DataFrame:
    """Return four aligned models over every test observation"""
    test_rows = source_dataset().loc[lambda frame: frame["split"].eq("test")]
    rows: list[pd.DataFrame] = []
    factors = {"naive_zero": 0.0, "arima": 0.5, "lstm": 0.75, "arima_lstm": 1.0}
    for model, factor in factors.items():
        predictions = test_rows.loc[:, ["date", "target_date", "ticker", "split"]].copy()
        predictions["model"] = model
        predictions["actual_return"] = test_rows["target_return"].to_numpy(dtype="float64")
        predictions["predicted_return"] = predictions["actual_return"] * factor
        rows.append(predictions.loc[:, PREDICTION_COLUMNS])
    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result["target_date"] = pd.to_datetime(result["target_date"])
    return result


def seed_results(family: str) -> tuple[FinalSeedResult, ...]:
    """Return ten aligned seed forecasts per ticker"""
    results: list[FinalSeedResult] = []
    for spec in FINAL_TEST_PROTOCOL.ticker_specs:
        config = spec.lstm_config if family == "lstm" else spec.residual_config
        for seed in FINAL_TEST_PROTOCOL.seeds:
            model = f"{family}_seed_{seed:02d}"
            base_predictions = final_predictions()
            selected = base_predictions["model"].eq("naive_zero") & base_predictions["ticker"].eq(
                spec.ticker
            )
            predictions = base_predictions.loc[selected].copy()
            predictions["model"] = model
            predictions["predicted_return"] = predictions["actual_return"] * (seed / 10)
            results.append(
                FinalSeedResult(
                    ticker=spec.ticker,
                    config=config,
                    seed=seed,
                    model=model,
                    best_epoch=2,
                    parameter_count=73,
                    predictions=predictions.reset_index(drop=True),
                )
            )
    return tuple(results)


def test_calculate_final_return_metrics_uses_observations_not_ticker_averages() -> None:
    """Calculate exact ticker and pooled return metrics"""
    predictions = final_predictions()

    result = calculate_final_return_metrics(predictions)

    aapl = result.loc[result["ticker"].eq("AAPL") & result["model"].eq("arima")].iloc[0]
    assert aapl["observations"] == 2
    assert aapl["mae_bps"] == pytest.approx(75.0)
    assert aapl["rmse_bps"] == pytest.approx(79.05694150420948)
    assert aapl["rel_mae"] == pytest.approx(0.5)
    assert aapl["oos_r2"] == pytest.approx(0.75)
    assert aapl["directional_accuracy"] == pytest.approx(1.0)

    pooled = result.loc[result["ticker"].eq("ALL") & result["model"].eq("arima")].iloc[0]
    arima = predictions.loc[predictions["model"].eq("arima")]
    errors = arima["actual_return"] - arima["predicted_return"]
    assert pooled["observations"] == len(arima)
    assert pooled["mae_bps"] == pytest.approx(errors.abs().mean() * 10_000)
    assert pooled["rmse_bps"] == pytest.approx(np.sqrt(np.square(errors).mean()) * 10_000)


@pytest.mark.parametrize(
    "case",
    [
        "model-column",
        "split",
        "alignment",
        "actuals",
        "models",
        "baseline",
        "zero-baseline",
    ],
)
def test_calculate_final_return_metrics_rejects_invalid_tables(case: str) -> None:
    """Reject nonfinal misaligned or undefined return metrics"""
    predictions = final_predictions()
    if case == "model-column":
        predictions = predictions.drop(columns="model")
    elif case == "split":
        predictions["split"] = "validation"
    elif case == "alignment":
        predictions = predictions.drop(predictions.loc[predictions["model"].eq("arima")].index[0])
    elif case == "actuals":
        index = predictions.loc[predictions["model"].eq("arima")].index[0]
        predictions.loc[index, "actual_return"] += 0.01
    elif case == "models":
        predictions.loc[predictions["model"].eq("arima"), "model"] = "wrong"
    elif case == "zero-baseline":
        predictions.loc[predictions["model"].eq("naive_zero"), "predicted_return"] = (
            predictions.loc[predictions["model"].eq("naive_zero"), "actual_return"].to_numpy()
        )

    with pytest.raises(ValueError):
        calculate_final_return_metrics(
            predictions,
            baseline_model="wrong" if case == "baseline" else "naive_zero",
        )


def test_calculate_seed_return_metrics_preserves_ticker_seed_identity() -> None:
    """Summarize every ordinary and hybrid seed independently"""
    ordinary = seed_results("lstm")
    residual = seed_results("residual_lstm")
    hybrid = tuple(
        replace(
            result,
            model=f"arima_lstm_seed_{result.seed:02d}",
            predictions=result.predictions.assign(model=f"arima_lstm_seed_{result.seed:02d}"),
        )
        for result in residual
    )

    metrics = calculate_seed_return_metrics(ordinary, hybrid)

    assert len(metrics) == 80
    assert metrics.groupby(["ticker", "family"]).size().eq(10).all()
    assert set(metrics["seed"]) == set(range(10))
    expected = metrics.loc[
        metrics["ticker"].eq("AAPL") & metrics["family"].eq("lstm") & metrics["seed"].eq(5)
    ].iloc[0]
    assert expected["mae_bps"] == pytest.approx(75.0)
    assert expected["rmse_bps"] == pytest.approx(79.05694150420948)


@pytest.mark.parametrize(
    "case",
    ["missing-seed", "misaligned", "wrong-type", "ticker", "config", "cross-family"],
)
def test_calculate_seed_return_metrics_rejects_incomplete_results(case: str) -> None:
    """Reject incomplete or misaligned seed collections"""
    ordinary = list(seed_results("lstm"))
    residual = seed_results("residual_lstm")
    hybrid = [
        replace(
            result,
            model=f"arima_lstm_seed_{result.seed:02d}",
            predictions=result.predictions.assign(model=f"arima_lstm_seed_{result.seed:02d}"),
        )
        for result in residual
    ]
    if case == "missing-seed":
        ordinary.pop()
    elif case == "misaligned":
        broken = hybrid[0].predictions.iloc[1:].reset_index(drop=True)
        hybrid[0] = replace(hybrid[0], predictions=broken)
    elif case == "wrong-type":
        ordinary[0] = cast(FinalSeedResult, object())
    elif case == "ticker":
        ordinary[0] = replace(ordinary[0], ticker="TSLA")
    elif case == "config":
        replacement = LSTMConfig(window=6, units=32, dropout=0.0)
        ordinary[:10] = [replace(result, config=replacement) for result in ordinary[:10]]
    else:
        hybrid[:10] = [
            replace(
                result,
                predictions=result.predictions.assign(
                    actual_return=result.predictions["actual_return"] + 0.001
                ),
            )
            for result in hybrid[:10]
        ]

    with pytest.raises(ValueError):
        calculate_seed_return_metrics(ordinary, hybrid)


def test_build_price_predictions_and_calculate_metrics() -> None:
    """Reconstruct prices and calculate ticker and macro metrics"""
    dataset = source_dataset()
    predictions = final_predictions()

    prices = build_price_predictions(dataset, predictions)
    metrics = calculate_final_price_metrics(prices, dataset)

    assert prices.columns.tolist() == list(PRICE_PREDICTION_COLUMNS)
    first = prices.iloc[0]
    assert first["actual_price"] == pytest.approx(
        first["adj_close"] * np.exp(first["actual_return"])
    )
    assert first["predicted_price"] == pytest.approx(
        first["adj_close"] * np.exp(first["predicted_return"])
    )
    aapl_arima = prices.loc[prices["ticker"].eq("AAPL") & prices["model"].eq("arima")]
    errors = aapl_arima["actual_price"] - aapl_arima["predicted_price"]
    row = metrics.loc[metrics["ticker"].eq("AAPL") & metrics["model"].eq("arima")].iloc[0]
    assert row["price_mae"] == pytest.approx(errors.abs().mean())
    assert row["price_rmse"] == pytest.approx(np.sqrt(np.square(errors).mean()))
    scale_rows = dataset.loc[
        dataset["ticker"].eq("AAPL") & dataset["split"].isin(["train", "validation"])
    ]
    scale = (
        (scale_rows["adj_close"] * np.exp(scale_rows["target_return"]) - scale_rows["adj_close"])
        .abs()
        .mean()
    )
    assert row["mase"] == pytest.approx(errors.abs().mean() / scale)
    pooled = metrics.loc[metrics["ticker"].eq("ALL") & metrics["model"].eq("arima")].iloc[0]
    ticker_rows = metrics.loc[metrics["ticker"].isin(CORE_TICKERS) & metrics["model"].eq("arima")]
    assert np.isnan(pooled["price_mae"])
    assert np.isnan(pooled["price_rmse"])
    assert pooled["smape"] == pytest.approx(ticker_rows["smape"].mean())
    assert pooled["mase"] == pytest.approx(ticker_rows["mase"].mean())


@pytest.mark.parametrize(
    "case",
    [
        "missing-join",
        "nonpositive",
        "overflow",
        "actual-mismatch",
        "zero-scale",
        "schema",
        "calculation-adj-close",
        "negative-price",
    ],
)
def test_price_metrics_reject_invalid_inputs(case: str) -> None:
    """Reject invalid joins prices forecasts and scales"""
    dataset = source_dataset()
    predictions = final_predictions()
    if case == "missing-join":
        dataset = dataset.drop(dataset.loc[dataset["split"].eq("test")].index[0])
    elif case == "nonpositive":
        dataset.loc[dataset["split"].eq("test"), "adj_close"] = 0.0
    elif case == "overflow":
        predictions.loc[0, "predicted_return"] = 1_000.0
    elif case == "actual-mismatch":
        first = predictions.iloc[0]
        selected = predictions["ticker"].eq(first["ticker"]) & predictions["target_date"].eq(
            first["target_date"]
        )
        predictions.loc[selected, "actual_return"] += 0.01
    elif case == "zero-scale":
        training = dataset["split"].isin(["train", "validation"])
        dataset.loc[training, "target_return"] = 0.0

    with pytest.raises(ValueError):
        prices = build_price_predictions(dataset, predictions)
        if case == "schema":
            prices = prices.drop(columns="predicted_price")
        elif case == "calculation-adj-close":
            prices["adj_close"] = 0.0
        elif case == "negative-price":
            prices["predicted_price"] = -1.0
        calculate_final_price_metrics(prices, dataset)


def test_price_metrics_treat_zero_smape_denominator_as_zero() -> None:
    """Assign zero contribution to zero sMAPE denominator"""
    dataset = source_dataset()
    prices = build_price_predictions(dataset, final_predictions())
    first_date = prices["date"].iloc[0]
    first_target_date = prices["target_date"].iloc[0]
    first_ticker = prices["ticker"].iloc[0]
    selected = (
        prices["date"].eq(first_date)
        & prices["target_date"].eq(first_target_date)
        & prices["ticker"].eq(first_ticker)
    )
    prices.loc[selected, ["actual_price", "predicted_price"]] = 0.0

    metrics = calculate_final_price_metrics(prices, dataset)

    assert np.isfinite(metrics["smape"]).all()
