"""Tests for final expanding ARIMA and hybrid backtesting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import evaluation.final_backtest as final_backtest_module
from evaluation.contracts import PREDICTION_COLUMNS, PREDICTION_NAME
from evaluation.final_backtest import (
    FinalARIMAData,
    align_final_predictions,
    build_final_arima_data,
    build_final_arima_ticker_predictions,
    build_final_hybrid_predictions,
    build_final_residual_dataset,
    build_final_seed_hybrid_predictions,
    build_naive_test_predictions,
)
from evaluation.final_models import FINAL_TEST_PROTOCOL, FinalSeedResult
from evaluation.walk_forward import ModelFitter
from models.interfaces import ForecastModel
from models.lstm import LSTMConfig

DATASET_COLUMNS = [
    "date",
    "target_date",
    "ticker",
    "adj_close",
    "log_return",
    "target_return",
    "split",
]
CORE_TICKERS = ("AAPL", "JPM", "SPY", "XOM")
TEST_RESIDUAL_CONFIG = LSTMConfig(5, 32, 0.0)


def final_dataset() -> pd.DataFrame:
    """Return short canonical consecutive three-split histories"""
    schedule = (
        ("2015-12-28", "2015-12-29", "train"),
        ("2015-12-29", "2015-12-30", "train"),
        ("2015-12-30", "2015-12-31", "train"),
        ("2015-12-31", "2019-12-30", "validation"),
        ("2019-12-30", "2019-12-31", "validation"),
        ("2019-12-31", "2020-01-02", "test"),
        ("2020-01-02", "2020-01-03", "test"),
        ("2020-01-03", "2025-12-30", "test"),
        ("2025-12-30", "2025-12-31", "test"),
    )
    rows: list[dict[str, object]] = []
    for ticker_position, ticker in enumerate(CORE_TICKERS, start=1):
        for row_position, (date, target_date, split) in enumerate(schedule, start=1):
            log_return = ticker_position / 100 + row_position / 1_000
            rows.append(
                {
                    "date": date,
                    "target_date": target_date,
                    "ticker": ticker,
                    "adj_close": 100.0 + ticker_position + row_position,
                    "log_return": log_return,
                    "target_return": log_return + 0.0005,
                    "split": split,
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=41).reset_index(drop=True)


def short_protocol() -> Any:
    """Return final protocol with a compact ARIMA warm-up"""
    return replace(FINAL_TEST_PROTOCOL, arima_warmup=2)


class RecordingARIMAModel:
    """Return the final fitted history value as forecast."""

    def __init__(self, value: float, observations: list[pd.DataFrame]) -> None:
        self.value = value
        self.observations = observations

    @property
    def name(self) -> str:
        """Return frozen ARIMA model name"""
        return "arima"

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Record one target row and return an aligned forecast"""
        self.observations.append(observations.copy(deep=True))
        return pd.Series(
            self.value,
            index=observations.index,
            dtype="float64",
            name=PREDICTION_NAME,
        )


class RecordingARIMAFitter:
    """Record every expanding final ARIMA history."""

    def __init__(self, error_at: int | None = None) -> None:
        self.error_at = error_at
        self.calls: list[tuple[str, pd.DataFrame]] = []
        self.observations: list[pd.DataFrame] = []

    def __call__(self, ticker: str, history: pd.DataFrame) -> ForecastModel:
        """Record history and return its latest return forecast"""
        self.calls.append((ticker, history.copy(deep=True)))
        if self.error_at == len(self.calls):
            raise RuntimeError("controlled fit failure")
        return RecordingARIMAModel(float(history["log_return"].iloc[-1]), self.observations)


def test_build_final_arima_data_uses_expanding_histories_and_progress() -> None:
    """Forecast complete schedule from causal expanding histories"""
    dataset = final_dataset()
    original = dataset.copy(deep=True)
    fitter = RecordingARIMAFitter()
    progress: list[tuple[int, int, str]] = []

    result = build_final_arima_data(
        dataset,
        protocol=short_protocol(),
        model_fitter=cast(ModelFitter, fitter),
        progress=lambda completed, total, ticker: progress.append((completed, total, ticker)),
    )

    pd.testing.assert_frame_equal(dataset, original)
    assert isinstance(result, FinalARIMAData)
    assert result.arima_predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert len(result.arima_predictions) == 8 * len(CORE_TICKERS)
    assert len(result.residual_dataset) == 7 * len(CORE_TICKERS)
    assert set(result.arima_predictions["split"]) == {"train", "validation", "test"}
    assert set(result.residual_dataset["split"]) == {"train", "validation", "test"}
    assert result.arima_predictions["model"].eq("arima").all()
    assert len(fitter.calls) == 8 * len(CORE_TICKERS)
    assert [len(history) for _, history in fitter.calls[:8]] == list(range(2, 10))
    assert progress[0] == (1, 32, "AAPL")
    assert progress[-1] == (32, 32, "XOM")
    assert [entry[0] for entry in progress] == list(range(1, 33))

    for (ticker, history), observation in zip(
        fitter.calls,
        fitter.observations,
        strict=True,
    ):
        origin = pd.Timestamp(observation["date"].iloc[0])
        assert "target_return" not in history.columns
        assert "target_return" not in observation.columns
        assert history["ticker"].eq(ticker).all()
        assert history["date"].max() <= origin

    first_test_position = next(
        position
        for position, (_, history) in enumerate(fitter.calls)
        if history["ticker"].iloc[0] == "AAPL"
        and history["date"].max() == pd.Timestamp("2019-12-31")
    )
    first_test_history = fitter.calls[first_test_position][1]
    assert pd.Timestamp("2019-12-30") in set(first_test_history["date"])
    assert pd.Timestamp("2020-01-02") not in set(first_test_history["date"])


def test_build_final_arima_data_uses_frozen_orders_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct default fitter from frozen ticker orders"""
    captured: list[tuple[dict[str, tuple[int, int, int]], str]] = []
    recording = RecordingARIMAFitter()

    def fake_arima_fitter(
        orders: dict[str, tuple[int, int, int]],
        model_name: str = "arima",
    ) -> ModelFitter:
        captured.append((dict(orders), model_name))
        return cast(ModelFitter, recording)

    monkeypatch.setattr(final_backtest_module, "ARIMAFitter", fake_arima_fitter)

    build_final_arima_data(final_dataset(), protocol=short_protocol())

    assert captured == [
        (
            {
                "AAPL": (0, 0, 0),
                "JPM": (0, 0, 1),
                "SPY": (0, 0, 2),
                "XOM": (2, 0, 1),
            },
            "arima",
        )
    ]


def test_build_final_arima_ticker_predictions_isolates_one_ticker() -> None:
    """Build one resumable ticker unit with local progress"""
    fitter = RecordingARIMAFitter()
    progress: list[tuple[int, int, str]] = []

    result = build_final_arima_ticker_predictions(
        final_dataset(),
        "JPM",
        protocol=short_protocol(),
        model_fitter=cast(ModelFitter, fitter),
        progress=lambda completed, total, ticker: progress.append((completed, total, ticker)),
    )

    assert result["ticker"].eq("JPM").all()
    assert len(result) == 8
    assert len(fitter.calls) == 8
    assert progress == [(position, 8, "JPM") for position in range(1, 9)]


def test_build_final_arima_ticker_predictions_uses_ticker_order_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct default fitter for only requested ticker"""
    captured: list[tuple[dict[str, tuple[int, int, int]], str]] = []
    recording = RecordingARIMAFitter()

    def fake_arima_fitter(
        orders: dict[str, tuple[int, int, int]],
        model_name: str = "arima",
    ) -> ModelFitter:
        captured.append((dict(orders), model_name))
        return cast(ModelFitter, recording)

    monkeypatch.setattr(final_backtest_module, "ARIMAFitter", fake_arima_fitter)

    result = build_final_arima_ticker_predictions(
        final_dataset(),
        "JPM",
        protocol=short_protocol(),
    )

    assert captured == [({"JPM": (0, 0, 1)}, "arima")]
    assert result["ticker"].eq("JPM").all()


@pytest.mark.parametrize("argument", ["model_fitter", "progress"])
def test_build_final_arima_ticker_predictions_rejects_invalid_callbacks(
    argument: str,
) -> None:
    """Reject noncallable ticker builder callbacks"""
    kwargs: dict[str, object] = {
        "protocol": short_protocol(),
        "model_fitter": RecordingARIMAFitter(),
    }
    kwargs[argument] = "invalid"

    with pytest.raises(ValueError, match=argument):
        build_final_arima_ticker_predictions(
            final_dataset(),
            "AAPL",
            **cast(Any, kwargs),
        )


def test_build_final_arima_ticker_predictions_rejects_unknown_ticker() -> None:
    """Reject ticker units outside frozen final protocol"""
    with pytest.raises(ValueError, match="ticker"):
        build_final_arima_ticker_predictions(
            final_dataset(),
            "TSLA",
            protocol=short_protocol(),
            model_fitter=cast(ModelFitter, RecordingARIMAFitter()),
        )


def test_build_final_arima_data_builds_consecutive_residuals() -> None:
    """Preserve residual values across both split boundaries"""
    result = build_final_arima_data(
        final_dataset(),
        protocol=short_protocol(),
        model_fitter=cast(ModelFitter, RecordingARIMAFitter()),
    )
    predictions = result.arima_predictions.loc[
        result.arima_predictions["ticker"].eq("AAPL")
    ].reset_index(drop=True)
    residuals = result.residual_dataset.loc[
        result.residual_dataset["ticker"].eq("AAPL")
    ].reset_index(drop=True)

    assert residuals[["date", "target_date", "split"]].values.tolist() == [
        [pd.Timestamp("2015-12-30"), pd.Timestamp("2015-12-31"), "train"],
        [pd.Timestamp("2015-12-31"), pd.Timestamp("2019-12-30"), "validation"],
        [pd.Timestamp("2019-12-30"), pd.Timestamp("2019-12-31"), "validation"],
        [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-02"), "test"],
        [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"), "test"],
        [pd.Timestamp("2020-01-03"), pd.Timestamp("2025-12-30"), "test"],
        [pd.Timestamp("2025-12-30"), pd.Timestamp("2025-12-31"), "test"],
    ]
    assert residuals["log_return"].tolist() == pytest.approx(
        (predictions["actual_return"] - predictions["predicted_return"]).iloc[:-1]
    )
    assert residuals["target_return"].tolist() == pytest.approx(
        (predictions["actual_return"] - predictions["predicted_return"]).iloc[1:]
    )

    public_result = build_final_residual_dataset(
        final_dataset(),
        result.arima_predictions,
        short_protocol(),
    )
    pd.testing.assert_frame_equal(public_result, result.residual_dataset)


@pytest.mark.parametrize("case", ["missing-leading", "missing-interior", "duplicate", "extra"])
def test_final_residual_builder_rejects_incomplete_arima_schedule(case: str) -> None:
    """Reject missing duplicate and extra ARIMA schedule rows"""
    dataset = final_dataset()
    protocol = short_protocol()
    predictions = build_final_arima_data(
        dataset,
        protocol=protocol,
        model_fitter=cast(ModelFitter, RecordingARIMAFitter()),
    ).arima_predictions
    aapl = predictions.loc[predictions["ticker"].eq("AAPL")]
    if case == "missing-leading":
        malformed = predictions.drop(aapl.index[0])
    elif case == "missing-interior":
        malformed = predictions.drop(aapl.index[2])
    elif case == "duplicate":
        malformed = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    else:
        extra = predictions.iloc[[0]].copy()
        extra["date"] = pd.Timestamp("2015-12-27")
        extra["target_date"] = pd.Timestamp("2015-12-28")
        malformed = pd.concat([predictions, extra], ignore_index=True)

    with pytest.raises(ValueError, match="schedule|duplicate"):
        build_final_residual_dataset(dataset, malformed, protocol)


def test_final_residual_builder_rejects_wrong_actual_and_chronology() -> None:
    """Reject inconsistent returns and residual chronology gaps"""
    dataset = final_dataset()
    protocol = short_protocol()
    predictions = build_final_arima_data(
        dataset,
        protocol=protocol,
        model_fitter=cast(ModelFitter, RecordingARIMAFitter()),
    ).arima_predictions
    wrong_actual = predictions.copy()
    wrong_actual.loc[wrong_actual.index[0], "actual_return"] += 0.1

    with pytest.raises(ValueError, match="actual_return"):
        build_final_residual_dataset(dataset, wrong_actual, protocol)

    changed_dataset = dataset.copy()
    changed_predictions = predictions.copy()
    target = changed_dataset.loc[
        changed_dataset["ticker"].eq("AAPL") & changed_dataset["target_date"].eq("2015-12-31")
    ].index[0]
    changed_dataset.loc[target, "date"] = pd.Timestamp("2015-12-29")
    prediction_target = changed_predictions.loc[
        changed_predictions["ticker"].eq("AAPL")
        & changed_predictions["target_date"].eq("2015-12-31")
    ].index[0]
    changed_predictions.loc[prediction_target, "date"] = pd.Timestamp("2015-12-29")

    with pytest.raises(ValueError, match="chronology"):
        build_final_residual_dataset(
            changed_dataset,
            changed_predictions,
            protocol,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("schema", "columns", id="schema"),
        pytest.param("empty", "empty", id="empty"),
        pytest.param("date-type", "date.*datetime", id="date-type"),
        pytest.param("target-type", "target_date.*datetime", id="target-type"),
        pytest.param("missing-date", "finite dates", id="missing-date"),
        pytest.param("split", "split", id="split"),
        pytest.param("model", "model", id="model"),
    ],
)
def test_final_residual_builder_rejects_malformed_arima_predictions(
    case: str,
    message: str,
) -> None:
    """Reject malformed complete-schedule ARIMA tables"""
    dataset = final_dataset()
    protocol = short_protocol()
    predictions = build_final_arima_data(
        dataset,
        protocol=protocol,
        model_fitter=cast(ModelFitter, RecordingARIMAFitter()),
    ).arima_predictions
    if case == "schema":
        predictions = predictions.drop(columns="date")
    elif case == "empty":
        predictions = predictions.iloc[0:0]
    elif case == "date-type":
        predictions["date"] = predictions["date"].astype(str)
    elif case == "target-type":
        predictions["target_date"] = predictions["target_date"].astype(str)
    elif case == "missing-date":
        predictions.loc[predictions.index[0], "date"] = pd.NaT
    elif case == "split":
        predictions.loc[predictions.index[0], "split"] = "holdout"
    else:
        predictions["model"] = "wrong"

    with pytest.raises(ValueError, match=message):
        build_final_residual_dataset(dataset, predictions, protocol)


def test_build_final_arima_data_rejects_insufficient_warmup_history() -> None:
    """Reject schedules lacking two post-warmup forecasts"""
    protocol = replace(FINAL_TEST_PROTOCOL, arima_warmup=9)

    with pytest.raises(ValueError, match="insufficient ARIMA history"):
        build_final_arima_data(
            final_dataset(),
            protocol=protocol,
            model_fitter=cast(ModelFitter, RecordingARIMAFitter()),
        )


def test_build_final_arima_data_adds_failure_context() -> None:
    """Report ticker and origin for failed ARIMA fit"""
    with pytest.raises(ValueError, match="AAPL.*2015-12-30.*controlled fit failure"):
        build_final_arima_data(
            final_dataset(),
            protocol=short_protocol(),
            model_fitter=cast(ModelFitter, RecordingARIMAFitter(error_at=2)),
        )


def test_build_final_arima_data_rejects_wrong_runtime_model_name() -> None:
    """Reject fitters returning non-ARIMA model identities"""

    class WrongNameModel(RecordingARIMAModel):
        @property
        def name(self) -> str:
            """Return an invalid model identifier"""
            return "wrong"

    def fitter(_ticker: str, history: pd.DataFrame) -> ForecastModel:
        return WrongNameModel(float(history["log_return"].iloc[-1]), [])

    with pytest.raises(ValueError, match="return model 'arima'"):
        build_final_arima_data(
            final_dataset(),
            protocol=short_protocol(),
            model_fitter=fitter,
        )


def test_build_final_arima_data_rejects_name_change_during_prediction() -> None:
    """Reject ARIMA names changing while forecasting"""

    class ChangingNameModel:
        def __init__(self) -> None:
            self.predicted = False

        @property
        def name(self) -> str:
            """Return state-dependent model identifier"""
            return "changed" if self.predicted else "arima"

        def predict(self, observations: pd.DataFrame) -> pd.Series:
            """Change identity after producing valid forecast"""
            self.predicted = True
            return pd.Series(
                0.0,
                index=observations.index,
                dtype="float64",
                name=PREDICTION_NAME,
            )

    def fitter(_ticker: str, _history: pd.DataFrame) -> ForecastModel:
        return ChangingNameModel()

    with pytest.raises(ValueError, match="remain 'arima'"):
        build_final_arima_data(
            final_dataset(),
            protocol=short_protocol(),
            model_fitter=fitter,
        )


def test_build_final_arima_data_rejects_noncallable_fitter() -> None:
    """Reject invalid custom ARIMA fitter objects"""
    with pytest.raises(ValueError, match="model_fitter"):
        build_final_arima_data(
            final_dataset(),
            protocol=short_protocol(),
            model_fitter=cast(ModelFitter, 1),
        )


def test_build_naive_test_predictions_returns_complete_zero_baseline() -> None:
    """Build zero forecasts for every canonical test row"""
    dataset = final_dataset()
    original = dataset.copy(deep=True)

    result = build_naive_test_predictions(dataset)

    pd.testing.assert_frame_equal(dataset, original)
    assert result.columns.tolist() == list(PREDICTION_COLUMNS)
    assert len(result) == 4 * len(CORE_TICKERS)
    assert result["split"].eq("test").all()
    assert result["model"].eq("naive_zero").all()
    assert result["predicted_return"].eq(0.0).all()
    assert result["target_date"].min() == pd.Timestamp("2020-01-02")
    assert result["target_date"].max() == pd.Timestamp("2025-12-31")


def component_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned final ARIMA and residual predictions"""
    naive = build_naive_test_predictions(final_dataset())
    arima = naive.copy(deep=True)
    arima["model"] = "arima"
    arima["predicted_return"] = 0.002
    residual = naive.copy(deep=True)
    residual["model"] = "residual_lstm"
    residual["actual_return"] = arima["actual_return"] - arima["predicted_return"]
    residual["predicted_return"] = np.arange(len(residual), dtype="float64") / 100_000
    return arima, residual


def test_build_final_hybrid_predictions_adds_aligned_components() -> None:
    """Add residual forecasts while retaining raw actual returns"""
    arima, residual = component_predictions()
    original_arima = arima.copy(deep=True)
    original_residual = residual.copy(deep=True)

    result = build_final_hybrid_predictions(arima, residual)

    pd.testing.assert_frame_equal(arima, original_arima)
    pd.testing.assert_frame_equal(residual, original_residual)
    assert result.columns.tolist() == list(PREDICTION_COLUMNS)
    assert result["model"].eq("arima_lstm").all()
    assert result["actual_return"].tolist() == pytest.approx(arima["actual_return"])
    assert result["predicted_return"].tolist() == pytest.approx(
        arima["predicted_return"] + residual["predicted_return"]
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("key", "align|complete", id="key"),
        pytest.param("identity", "residual identity", id="identity"),
        pytest.param("arima-model", "model", id="arima-model"),
        pytest.param("residual-model", "model", id="residual-model"),
        pytest.param("split", "test", id="split"),
        pytest.param("overflow", "finite", id="overflow"),
        pytest.param("interior-key", "align", id="interior-key"),
    ],
)
def test_build_final_hybrid_predictions_rejects_invalid_components(
    case: str,
    message: str,
) -> None:
    """Reject misaligned mislabeled or inconsistent components"""
    arima, residual = component_predictions()
    if case == "key":
        residual = residual.iloc[1:].copy()
    elif case == "identity":
        residual.loc[residual.index[0], "actual_return"] += 2e-12
    elif case == "arima-model":
        arima["model"] = "wrong"
    elif case == "residual-model":
        residual["model"] = "wrong"
    elif case == "split":
        residual["split"] = "validation"
    elif case == "interior-key":
        interior = residual.index[residual["target_date"].eq(pd.Timestamp("2020-01-03"))][0]
        residual.loc[interior, "target_date"] = pd.Timestamp("2020-01-04")
    else:
        arima["actual_return"] = 1.5e308
        arima["predicted_return"] = 1.0e308
        residual["actual_return"] = 5.0e307
        residual["predicted_return"] = 1.0e308

    with pytest.raises(ValueError, match=message):
        build_final_hybrid_predictions(arima, residual)


def residual_seed_results() -> tuple[pd.DataFrame, tuple[FinalSeedResult, ...]]:
    """Return one ticker ARIMA table and ten residual seeds"""
    arima, residual = component_predictions()
    arima = arima.loc[arima["ticker"].eq("AAPL")].reset_index(drop=True)
    residual = residual.loc[residual["ticker"].eq("AAPL")].reset_index(drop=True)
    results = []
    for seed in range(10):
        predictions = residual.copy(deep=True)
        predictions["model"] = f"residual_lstm_seed_{seed:02d}"
        predictions["predicted_return"] = seed / 1_000
        results.append(
            FinalSeedResult(
                ticker="AAPL",
                config=TEST_RESIDUAL_CONFIG,
                seed=seed,
                model=f"residual_lstm_seed_{seed:02d}",
                best_epoch=seed + 1,
                parameter_count=73,
                predictions=predictions,
            )
        )
    return arima, tuple(results)


def test_build_final_seed_hybrid_predictions_preserves_each_seed() -> None:
    """Compose every residual seed with matching ARIMA rows"""
    arima, residual_results = residual_seed_results()

    results = build_final_seed_hybrid_predictions(arima, tuple(reversed(residual_results)))

    assert [result.seed for result in results] == list(range(10))
    assert [result.model for result in results] == [
        f"arima_lstm_seed_{seed:02d}" for seed in range(10)
    ]
    assert [result.best_epoch for result in results] == list(range(1, 11))
    for seed, result in enumerate(results):
        assert result.predictions["model"].eq(f"arima_lstm_seed_{seed:02d}").all()
        assert result.predictions["predicted_return"].tolist() == pytest.approx(
            arima["predicted_return"] + seed / 1_000
        )


def test_build_final_seed_hybrid_predictions_requires_all_seeds() -> None:
    """Reject incomplete residual seed ensembles"""
    arima, residual_results = residual_seed_results()

    with pytest.raises(ValueError, match="seeds"):
        build_final_seed_hybrid_predictions(arima, residual_results[:-1])


def test_build_final_seed_hybrid_predictions_requires_residual_config() -> None:
    """Reject seed ensembles using ordinary model configuration"""
    arima, residual_results = residual_seed_results()
    changed = tuple(replace(result, config=LSTMConfig(5, 16, 0.0)) for result in residual_results)

    with pytest.raises(ValueError, match="config"):
        build_final_seed_hybrid_predictions(arima, changed)


def test_build_final_seed_hybrid_predictions_requires_matching_arima_ticker() -> None:
    """Reject ARIMA tables omitting the residual ticker"""
    arima, residual_results = residual_seed_results()
    replacement_arima = arima.copy(deep=True)
    replacement_arima["ticker"] = "JPM"

    with pytest.raises(ValueError, match="residual ticker"):
        build_final_seed_hybrid_predictions(replacement_arima, residual_results)


def four_model_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return four complete aligned final model tables"""
    naive = build_naive_test_predictions(final_dataset())
    arima = naive.copy(deep=True)
    arima["model"] = "arima"
    arima["predicted_return"] = 0.002
    lstm = naive.copy(deep=True)
    lstm["model"] = "lstm"
    lstm["predicted_return"] = 0.003
    hybrid = naive.copy(deep=True)
    hybrid["model"] = "arima_lstm"
    hybrid["predicted_return"] = 0.004
    return naive, arima, lstm, hybrid


def test_align_final_predictions_returns_four_complete_models() -> None:
    """Align and sort all complete final prediction tables"""
    frames = four_model_predictions()
    originals = tuple(frame.copy(deep=True) for frame in frames)

    result = align_final_predictions(*frames)

    for frame, original in zip(frames, originals, strict=True):
        pd.testing.assert_frame_equal(frame, original)
    assert result.columns.tolist() == list(PREDICTION_COLUMNS)
    assert len(result) == sum(len(frame) for frame in frames)
    assert result["model"].drop_duplicates().tolist() == [
        "arima",
        "arima_lstm",
        "lstm",
        "naive_zero",
    ]
    assert result["split"].eq("test").all()


@pytest.mark.parametrize(
    ("table_position", "case", "message"),
    [
        pytest.param(0, "missing", "align|complete", id="missing-naive"),
        pytest.param(1, "actual", "actual", id="arima-actual"),
        pytest.param(2, "model", "model", id="lstm-model"),
        pytest.param(3, "split", "test", id="hybrid-split"),
        pytest.param(0, "date", "date", id="missing-date"),
        pytest.param(1, "ticker", "ticker|complete", id="missing-ticker"),
        pytest.param(2, "nonfinite", "finite", id="nonfinite"),
        pytest.param(1, "interior-key", "align", id="interior-key"),
    ],
)
def test_align_final_predictions_rejects_invalid_model_tables(
    table_position: int,
    case: str,
    message: str,
) -> None:
    """Reject partial mislabeled and inconsistent final tables"""
    frames = [frame.copy(deep=True) for frame in four_model_predictions()]
    target = frames[table_position]
    if case == "missing":
        frames[table_position] = target.iloc[1:].copy()
    elif case == "actual":
        target.loc[target.index[0], "actual_return"] += 0.1
    elif case == "model":
        target["model"] = "wrong"
    elif case == "split":
        target["split"] = "validation"
    elif case == "date":
        target.loc[target.index[0], "date"] = pd.NaT
    elif case == "ticker":
        frames[table_position] = target.loc[target["ticker"].ne("XOM")].copy()
    elif case == "interior-key":
        interior = target.index[target["target_date"].eq(pd.Timestamp("2020-01-03"))][0]
        target.loc[interior, "target_date"] = pd.Timestamp("2020-01-04")
    else:
        target.loc[target.index[0], "predicted_return"] = np.inf

    with pytest.raises(ValueError, match=message):
        align_final_predictions(*frames)


@pytest.mark.parametrize("progress", [1, "callback"])
def test_build_final_arima_data_rejects_noncallable_progress(progress: object) -> None:
    """Reject progress handlers that cannot receive updates"""
    with pytest.raises(ValueError, match="progress"):
        build_final_arima_data(
            final_dataset(),
            protocol=short_protocol(),
            model_fitter=cast(ModelFitter, RecordingARIMAFitter()),
            progress=cast(Callable[[int, int, str], None], progress),
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("schema", "columns", id="schema"),
        pytest.param("empty", "empty", id="empty"),
        pytest.param("date-type", "date.*datetime", id="date-type"),
        pytest.param("target-type", "target_date.*datetime", id="target-type"),
        pytest.param("missing-date", "finite dates", id="missing-date"),
        pytest.param("chronology", "precede", id="chronology"),
        pytest.param("duplicate", "duplicate", id="duplicate"),
        pytest.param("ticker", "ticker", id="ticker"),
    ],
)
def test_build_final_hybrid_predictions_rejects_malformed_test_tables(
    case: str,
    message: str,
) -> None:
    """Reject malformed final component table contracts"""
    arima, residual = component_predictions()
    if case == "schema":
        residual = residual.drop(columns="date")
    elif case == "empty":
        residual = residual.iloc[0:0]
    elif case == "date-type":
        residual["date"] = residual["date"].astype(str)
    elif case == "target-type":
        residual["target_date"] = residual["target_date"].astype(str)
    elif case == "missing-date":
        residual.loc[residual.index[0], "date"] = pd.NaT
    elif case == "chronology":
        residual.loc[residual.index[0], "date"] = residual.loc[residual.index[0], "target_date"]
    elif case == "duplicate":
        residual = pd.concat([residual, residual.iloc[[0]]], ignore_index=True)
    else:
        residual.loc[residual.index[0], "ticker"] = "TSLA"

    with pytest.raises(ValueError, match=message):
        build_final_hybrid_predictions(arima, residual)
