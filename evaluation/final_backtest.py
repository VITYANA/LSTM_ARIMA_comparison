"""Build final expanding ARIMA, residual and hybrid forecasts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.contracts import (
    PREDICTION_COLUMNS,
    PREDICTION_NAME,
    build_model_input,
    require_finite_numeric,
    validate_model_dataset,
    validate_prediction_series,
)
from evaluation.final_models import (
    FINAL_KEY_COLUMNS,
    FINAL_SORT_COLUMNS,
    FINAL_TEST_PROTOCOL,
    FinalSeedResult,
    FinalTestProtocol,
    average_final_seed_predictions,
    validate_final_dataset,
)
from evaluation.loading import REQUIRED_DATASET_COLUMNS
from evaluation.walk_forward import ModelFitter
from models.arima import ARIMAFitter
from models.interfaces import validate_model_name

ARIMA_MODEL_NAME = "arima"
NAIVE_MODEL_NAME = "naive_zero"
LSTM_MODEL_NAME = "lstm"
RESIDUAL_MODEL_NAME = "residual_lstm"
HYBRID_MODEL_NAME = "arima_lstm"
RESIDUAL_IDENTITY_ATOL = 1e-12
FINAL_SPLITS = frozenset({"train", "validation", "test"})

FinalProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class FinalARIMAData:
    """Contain complete expanding ARIMA forecasts and residual rows."""

    arima_predictions: pd.DataFrame
    residual_dataset: pd.DataFrame


def _eligible_arima_observations(
    dataset: pd.DataFrame,
    protocol: FinalTestProtocol,
) -> pd.DataFrame:
    """Return the complete protocol-derived expanding forecast schedule."""
    frames: list[pd.DataFrame] = []
    for spec in protocol.ticker_specs:
        rows = dataset.loc[dataset["ticker"].eq(spec.ticker)].sort_values(
            ["date", "target_date"],
            kind="mergesort",
        )
        eligible: list[int] = []
        dates = pd.DatetimeIndex(rows["date"])
        for position, origin in enumerate(dates):
            available = int(dates.searchsorted(origin, side="right"))
            if available >= protocol.arima_warmup:
                eligible.append(position)
        selected = rows.iloc[eligible].copy()
        if len(selected) < 2:
            raise ValueError(f"ticker {spec.ticker!r} has insufficient ARIMA history")
        frames.append(selected)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["ticker", "target_date", "date"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _validated_arima_schedule(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    protocol: FinalTestProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned source observations and complete ARIMA forecasts."""
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError("ARIMA predictions must use standard columns in standard order")
    if predictions.empty:
        raise ValueError("ARIMA predictions must not be empty")
    for column in ("date", "target_date"):
        if not pd.api.types.is_datetime64_any_dtype(predictions[column]):
            raise ValueError(f"ARIMA prediction {column} must be datetime")
        if predictions[column].isna().any():
            raise ValueError("ARIMA predictions must contain finite dates")
    if not predictions["split"].isin(FINAL_SPLITS).all():
        raise ValueError("ARIMA predictions contain unknown split values")
    if not predictions["model"].eq(ARIMA_MODEL_NAME).all():
        raise ValueError("ARIMA predictions must use model 'arima'")
    if predictions.duplicated(list(FINAL_KEY_COLUMNS)).any():
        raise ValueError("ARIMA predictions contain duplicate observation keys")
    require_finite_numeric(predictions["actual_return"], "ARIMA actual_return")
    require_finite_numeric(predictions["predicted_return"], "ARIMA predicted_return")

    source = _eligible_arima_observations(dataset, protocol)
    expected_keys = source.loc[:, FINAL_KEY_COLUMNS].reset_index(drop=True)
    ordered = predictions.sort_values(
        list(FINAL_SORT_COLUMNS),
        kind="mergesort",
    ).reset_index(drop=True)
    if not ordered.loc[:, FINAL_KEY_COLUMNS].equals(expected_keys):
        raise ValueError("ARIMA prediction keys must match the complete schedule")
    if not np.allclose(
        ordered["actual_return"].to_numpy(dtype="float64"),
        source["target_return"].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=RESIDUAL_IDENTITY_ATOL,
    ):
        raise ValueError("ARIMA actual_return must match source target_return")
    return source, ordered.copy(deep=True)


def build_final_residual_dataset(
    dataset: pd.DataFrame,
    arima_predictions: pd.DataFrame,
    protocol: FinalTestProtocol = FINAL_TEST_PROTOCOL,
) -> pd.DataFrame:
    """Build a complete consecutive residual dataset across all splits."""
    validated = validate_final_dataset(dataset, protocol)
    source, predictions = _validated_arima_schedule(validated, arima_predictions, protocol)
    source = source.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for ticker, ticker_predictions in predictions.groupby("ticker", sort=True):
        ordered = ticker_predictions.reset_index()
        for position in range(len(ordered) - 1):
            current = ordered.iloc[position]
            following = ordered.iloc[position + 1]
            if current["target_date"] != following["date"]:
                raise ValueError(f"residual chronology gap for ticker {ticker!r}")
            following_source = source.iloc[int(following["index"])]
            rows.append(
                {
                    "date": current["target_date"],
                    "target_date": following["target_date"],
                    "ticker": ticker,
                    "adj_close": following_source["adj_close"],
                    "log_return": current["actual_return"] - current["predicted_return"],
                    "target_return": following["actual_return"] - following["predicted_return"],
                    "split": following["split"],
                }
            )

    result = pd.DataFrame(rows, columns=REQUIRED_DATASET_COLUMNS)
    validate_model_dataset(result)
    require_finite_numeric(result["adj_close"], "residual adj_close")
    require_finite_numeric(result["log_return"], "residual log_return")
    return result.sort_values(
        ["ticker", "target_date", "date"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_arima_predictions(
    dataset: pd.DataFrame,
    protocol: FinalTestProtocol,
    fitter: ModelFitter,
    progress: FinalProgressCallback | None,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Run one expanding ARIMA fit for every eligible observation."""
    observations = _eligible_arima_observations(dataset, protocol)
    if ticker is not None:
        observations = observations.loc[observations["ticker"].eq(ticker)].reset_index(drop=True)
    total = len(observations)
    rows: list[dict[str, object]] = []
    for position in range(total):
        observation = observations.iloc[[position]].copy()
        ticker = str(observation["ticker"].iloc[0])
        origin = pd.Timestamp(observation["date"].iloc[0])
        history = dataset.loc[
            dataset["ticker"].eq(ticker) & dataset["date"].le(origin)
        ].sort_values(["date", "target_date"], kind="mergesort")
        history_input = build_model_input(history)
        observation_input = build_model_input(observation)
        try:
            model = fitter(ticker, history_input)
            if validate_model_name(model.name) != ARIMA_MODEL_NAME:
                raise ValueError("final ARIMA fitter must return model 'arima'")
            forecast = validate_prediction_series(
                model.predict(observation_input),
                observation_input,
            )
            if validate_model_name(model.name) != ARIMA_MODEL_NAME:
                raise ValueError("final ARIMA model name must remain 'arima'")
        except Exception as error:
            context = f"ticker {ticker!r} at origin {origin.date()}"
            raise ValueError(f"final ARIMA failed for {context}: {error}") from error

        rows.append(
            {
                "date": observation["date"].iloc[0],
                "target_date": observation["target_date"].iloc[0],
                "ticker": ticker,
                "split": observation["split"].iloc[0],
                "model": ARIMA_MODEL_NAME,
                "actual_return": observation["target_return"].iloc[0],
                PREDICTION_NAME: forecast.iloc[0],
            }
        )
        if progress is not None:
            progress(position + 1, total, ticker)
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS).reset_index(drop=True)


def build_final_arima_ticker_predictions(
    dataset: pd.DataFrame,
    ticker: str,
    protocol: FinalTestProtocol = FINAL_TEST_PROTOCOL,
    model_fitter: ModelFitter | None = None,
    progress: FinalProgressCallback | None = None,
) -> pd.DataFrame:
    """Build one independently checkpointable final ARIMA ticker."""
    validated = validate_final_dataset(dataset, protocol)
    specs = {spec.ticker: spec for spec in protocol.ticker_specs}
    if ticker not in specs:
        raise ValueError("ticker must belong to the final-test protocol")
    if model_fitter is not None and not callable(model_fitter):
        raise ValueError("model_fitter must be callable or None")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    if model_fitter is None:
        model_fitter = ARIMAFitter(
            {ticker: specs[ticker].arima_order},
            model_name=ARIMA_MODEL_NAME,
        )
    return _build_arima_predictions(
        validated,
        protocol,
        model_fitter,
        progress,
        ticker,
    )


def build_final_arima_data(
    dataset: pd.DataFrame,
    protocol: FinalTestProtocol = FINAL_TEST_PROTOCOL,
    model_fitter: ModelFitter | None = None,
    progress: FinalProgressCallback | None = None,
) -> FinalARIMAData:
    """Build complete final expanding ARIMA forecasts and residual data."""
    validated = validate_final_dataset(dataset, protocol)
    if model_fitter is not None and not callable(model_fitter):
        raise ValueError("model_fitter must be callable or None")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    if model_fitter is None:
        orders = {spec.ticker: spec.arima_order for spec in protocol.ticker_specs}
        model_fitter = ARIMAFitter(orders, model_name=ARIMA_MODEL_NAME)

    predictions = _build_arima_predictions(validated, protocol, model_fitter, progress)
    residual_dataset = build_final_residual_dataset(validated, predictions, protocol)
    return FinalARIMAData(
        arima_predictions=predictions.copy(deep=True),
        residual_dataset=residual_dataset,
    )


def build_naive_test_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return the complete zero-return final-test baseline."""
    validated = validate_final_dataset(dataset)
    test = validated.loc[validated["split"].eq("test")].copy()
    result = test.loc[:, FINAL_KEY_COLUMNS].copy()
    result["model"] = NAIVE_MODEL_NAME
    result["actual_return"] = test["target_return"].to_numpy(dtype="float64")
    result["predicted_return"] = 0.0
    return result.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)


def _validated_test_component(
    predictions: pd.DataFrame,
    *,
    label: str,
    expected_model: str,
    require_all_tickers: bool = False,
) -> pd.DataFrame:
    """Return one isolated canonical test prediction component."""
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError(f"{label} predictions must use standard columns in standard order")
    if predictions.empty:
        raise ValueError(f"{label} predictions must not be empty")
    for column in ("date", "target_date"):
        if not pd.api.types.is_datetime64_any_dtype(predictions[column]):
            raise ValueError(f"{label} prediction {column} must be datetime")
        if predictions[column].isna().any():
            raise ValueError(f"{label} predictions must contain finite dates")
    if not predictions["date"].lt(predictions["target_date"]).all():
        raise ValueError(f"{label} prediction date must precede target_date")
    if not predictions["split"].eq("test").all():
        raise ValueError(f"{label} predictions must contain test rows only")
    if not predictions["model"].eq(expected_model).all():
        raise ValueError(f"{label} predictions must use model {expected_model!r}")
    if predictions.duplicated(list(FINAL_KEY_COLUMNS)).any():
        raise ValueError(f"{label} predictions contain duplicate observation keys")
    expected_tickers = {spec.ticker for spec in FINAL_TEST_PROTOCOL.ticker_specs}
    observed_tickers = set(predictions["ticker"])
    if not observed_tickers <= expected_tickers:
        raise ValueError(f"{label} predictions contain invalid ticker values")
    if require_all_tickers and observed_tickers != expected_tickers:
        raise ValueError(f"{label} predictions must contain every final ticker")
    require_finite_numeric(predictions["actual_return"], f"{label} actual_return")
    require_finite_numeric(predictions["predicted_return"], f"{label} predicted_return")

    start = pd.Timestamp(FINAL_TEST_PROTOCOL.split_config.test_start)
    end = pd.Timestamp(FINAL_TEST_PROTOCOL.split_config.test_end)
    for ticker in observed_tickers:
        target_dates = predictions.loc[predictions["ticker"].eq(ticker), "target_date"]
        if target_dates.min() != start or target_dates.max() != end:
            raise ValueError(f"{label} predictions must cover the complete test boundaries")
    return (
        predictions.loc[:, PREDICTION_COLUMNS]
        .sort_values(list(FINAL_SORT_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )


def _compose_final_hybrid(
    arima_predictions: pd.DataFrame,
    residual_predictions: pd.DataFrame,
    *,
    residual_model: str,
    hybrid_model: str,
) -> pd.DataFrame:
    """Add one aligned final residual forecast table to ARIMA."""
    arima = _validated_test_component(
        arima_predictions,
        label="ARIMA",
        expected_model=ARIMA_MODEL_NAME,
    )
    residual = _validated_test_component(
        residual_predictions,
        label="residual",
        expected_model=residual_model,
    )
    if not arima.loc[:, FINAL_KEY_COLUMNS].equals(residual.loc[:, FINAL_KEY_COLUMNS]):
        raise ValueError("ARIMA and residual prediction keys do not align")

    actual = arima["actual_return"].to_numpy(dtype="float64")
    arima_forecast = arima["predicted_return"].to_numpy(dtype="float64")
    residual_actual = residual["actual_return"].to_numpy(dtype="float64")
    if not np.allclose(
        residual_actual,
        actual - arima_forecast,
        rtol=0.0,
        atol=RESIDUAL_IDENTITY_ATOL,
    ):
        raise ValueError("residual identity does not match actual_return minus ARIMA forecast")

    with np.errstate(over="ignore", invalid="ignore"):
        hybrid_forecast = arima_forecast + residual["predicted_return"].to_numpy(dtype="float64")
    if not np.isfinite(hybrid_forecast).all():
        raise ValueError("hybrid predictions must contain only finite values")

    result = arima.loc[:, FINAL_KEY_COLUMNS].copy()
    result["model"] = validate_model_name(hybrid_model)
    result["actual_return"] = actual
    result["predicted_return"] = hybrid_forecast
    return result.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)


def build_final_hybrid_predictions(
    arima_predictions: pd.DataFrame,
    residual_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build final ensemble ARIMA-LSTM forecasts."""
    return _compose_final_hybrid(
        arima_predictions,
        residual_predictions,
        residual_model=RESIDUAL_MODEL_NAME,
        hybrid_model=HYBRID_MODEL_NAME,
    )


def build_final_seed_hybrid_predictions(
    arima_predictions: pd.DataFrame,
    residual_results: Sequence[FinalSeedResult],
) -> tuple[FinalSeedResult, ...]:
    """Build one aligned final hybrid result for every residual seed."""
    values = tuple(residual_results)
    average_final_seed_predictions(values, RESIDUAL_MODEL_NAME)
    ordered = tuple(sorted(values, key=lambda result: result.seed))
    ticker = ordered[0].ticker
    expected_config = next(
        spec.residual_config for spec in FINAL_TEST_PROTOCOL.ticker_specs if spec.ticker == ticker
    )
    if any(result.config != expected_config for result in ordered):
        raise ValueError("residual seed config must match the frozen ticker configuration")

    arima = _validated_test_component(
        arima_predictions,
        label="ARIMA",
        expected_model=ARIMA_MODEL_NAME,
    )
    ticker_arima = arima.loc[arima["ticker"].eq(ticker)].reset_index(drop=True)
    if ticker_arima.empty:
        raise ValueError("ARIMA predictions do not contain the residual ticker")

    results: list[FinalSeedResult] = []
    for result in ordered:
        hybrid_name = f"{HYBRID_MODEL_NAME}_seed_{result.seed:02d}"
        predictions = _compose_final_hybrid(
            ticker_arima,
            result.predictions,
            residual_model=result.model,
            hybrid_model=hybrid_name,
        )
        results.append(
            FinalSeedResult(
                ticker=result.ticker,
                config=result.config,
                seed=result.seed,
                model=hybrid_name,
                best_epoch=result.best_epoch,
                parameter_count=result.parameter_count,
                predictions=predictions,
            )
        )
    return tuple(results)


def align_final_predictions(
    naive: pd.DataFrame,
    arima: pd.DataFrame,
    lstm: pd.DataFrame,
    hybrid: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and concatenate four complete aligned final models."""
    tables = (
        _validated_test_component(
            naive,
            label="naive",
            expected_model=NAIVE_MODEL_NAME,
            require_all_tickers=True,
        ),
        _validated_test_component(
            arima,
            label="ARIMA",
            expected_model=ARIMA_MODEL_NAME,
            require_all_tickers=True,
        ),
        _validated_test_component(
            lstm,
            label="LSTM",
            expected_model=LSTM_MODEL_NAME,
            require_all_tickers=True,
        ),
        _validated_test_component(
            hybrid,
            label="hybrid",
            expected_model=HYBRID_MODEL_NAME,
            require_all_tickers=True,
        ),
    )
    expected_keys = tables[0].loc[:, FINAL_KEY_COLUMNS]
    expected_actual = tables[0]["actual_return"]
    for table in tables[1:]:
        if not table.loc[:, FINAL_KEY_COLUMNS].equals(expected_keys):
            raise ValueError("final prediction keys do not align")
        if not table["actual_return"].equals(expected_actual):
            raise ValueError("final actual returns do not align")

    return (
        pd.concat(tables, ignore_index=True)
        .sort_values(
            ["model", "ticker", "target_date", "date"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
