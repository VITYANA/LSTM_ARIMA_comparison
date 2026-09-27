"""Tests for one complete LSTM validation run."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

import evaluation.lstm_validation as validation_module
from data.sequences import (
    build_return_sequences,
    fit_return_scaler,
    lag_columns,
    split_sequence_batch,
    transform_features,
    transform_targets,
)
from evaluation.contracts import BPS_FACTOR, PREDICTION_COLUMNS, PREDICTION_NAME
from evaluation.lstm_validation import (
    CORE_TICKERS,
    LSTMRunKey,
    LSTMRunResult,
    run_lstm_candidate,
    run_lstm_grid,
)
from models.lstm import LSTMConfig, LSTMTrainer, build_lstm_grid

TRAIN_ROWS = 10
VALIDATION_ROWS = 3
TEST_ROWS = 2


def make_dataset(
    *,
    ticker: str = "AAPL",
    train_rows: int = TRAIN_ROWS,
    validation_rows: int = VALIDATION_ROWS,
    test_rows: int = TEST_ROWS,
) -> pd.DataFrame:
    """Build deterministic rows spanning every research split"""
    row_count = train_rows + validation_rows + test_rows
    dates = pd.bdate_range("2019-01-01", periods=row_count + 1)
    returns = np.arange(1, row_count + 1, dtype="float64") / 1_000
    return pd.DataFrame(
        {
            "date": dates[:-1],
            "target_date": dates[1:],
            "ticker": ticker,
            "adj_close": 100.0 + np.arange(row_count),
            "log_return": returns,
            "target_return": returns + 0.0005,
            "split": [
                *("train" for _ in range(train_rows)),
                *("validation" for _ in range(validation_rows)),
                *("test" for _ in range(test_rows)),
            ],
        },
        index=pd.Index(range(100, 100 + row_count), name="source_index"),
    )


def make_core_dataset(*, include_extra: bool = False) -> pd.DataFrame:
    """Build all core ticker rows plus optional excluded ticker"""
    tickers = [*CORE_TICKERS, *(["TSLA"] if include_extra else [])]
    frames = []
    for position, ticker in enumerate(tickers):
        frame = make_dataset(ticker=ticker)
        frame.index = pd.Index(
            np.arange(len(frame)) + 1_000 * (position + 1),
            name="source_index",
        )
        frames.append(frame)
    return pd.concat(frames).sample(frac=1, random_state=19)


class RecordingModel:
    """Record model observations and return controlled forecasts."""

    def __init__(
        self,
        prediction: float = 0.0,
        *,
        parameter_count: int = 73,
        error: Exception | None = None,
        malformed: bool = False,
    ) -> None:
        self.parameter_count = parameter_count
        self.prediction = prediction
        self.error = error
        self.malformed = malformed
        self.received: pd.DataFrame | None = None

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Capture raw lags and return aligned controlled values"""
        self.received = observations.copy(deep=True)
        if self.error is not None:
            raise self.error
        index = observations.index[:-1] if self.malformed else observations.index
        return pd.Series(
            self.prediction,
            index=index,
            dtype="float64",
            name=PREDICTION_NAME,
        )


class RecordingTrainer:
    """Record the two-stage training lifecycle."""

    def __init__(
        self,
        model: RecordingModel | None = None,
        *,
        best_epoch: int = 3,
        determine_error: Exception | None = None,
        fit_error: Exception | None = None,
    ) -> None:
        self.model = model or RecordingModel()
        self.best_epoch = best_epoch
        self.determine_error = determine_error
        self.fit_error = fit_error
        self.determine_calls: list[dict[str, object]] = []
        self.fit_calls: list[dict[str, object]] = []

    def determine_best_epoch(
        self,
        config: LSTMConfig,
        seed: int,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
    ) -> int:
        """Record inner arrays and return a controlled epoch"""
        self.determine_calls.append(
            {
                "config": config,
                "seed": seed,
                "train_features": train_features.copy(),
                "train_targets": train_targets.copy(),
                "validation_features": validation_features.copy(),
                "validation_targets": validation_targets.copy(),
            }
        )
        if self.determine_error is not None:
            raise self.determine_error
        return self.best_epoch

    def fit(
        self,
        config: LSTMConfig,
        seed: int,
        features: np.ndarray,
        targets: np.ndarray,
        scaler: StandardScaler,
        feature_columns: tuple[str, ...],
        epochs: int,
        name: str,
    ) -> RecordingModel:
        """Record full-train arrays and return a controlled model"""
        self.fit_calls.append(
            {
                "config": config,
                "seed": seed,
                "features": features.copy(),
                "targets": targets.copy(),
                "scaler": scaler,
                "feature_columns": feature_columns,
                "epochs": epochs,
                "name": name,
            }
        )
        if self.fit_error is not None:
            raise self.fit_error
        return self.model


def run_candidate(
    dataset: pd.DataFrame,
    trainer: RecordingTrainer,
    *,
    config: LSTMConfig | None = None,
    seed: int = 3,
) -> LSTMRunResult:
    """Run one candidate while accepting the structural fake trainer"""
    key = LSTMRunKey(
        ticker="AAPL",
        config=config or LSTMConfig(window=5, units=16, dropout=0.2),
        seed=seed,
    )
    return run_lstm_candidate(dataset, key, cast(LSTMTrainer, trainer))


def recorded_array(call: dict[str, object], name: str) -> np.ndarray:
    """Return one recorded numeric trainer argument"""
    return cast(np.ndarray, call[name])


def test_lstm_run_key_builds_stable_unique_identifier() -> None:
    """Encode ticker configuration and seed without ambiguity"""
    key = LSTMRunKey(
        ticker="AAPL",
        config=LSTMConfig(window=5, units=16, dropout=0.2),
        seed=3,
    )

    assert key.run_id == "AAPL_lstm_w5_u16_d20_seed_03"


@pytest.mark.parametrize(
    ("ticker", "config", "seed", "message"),
    [
        ("TSLA", LSTMConfig(5, 16, 0.0), 0, "ticker"),
        (1, LSTMConfig(5, 16, 0.0), 0, "ticker"),
        ("AAPL", object(), 0, "config"),
        ("AAPL", LSTMConfig(5, 16, 0.0), True, "seed"),
        ("AAPL", LSTMConfig(5, 16, 0.0), -1, "seed"),
        ("AAPL", LSTMConfig(5, 16, 0.0), 10, "seed"),
    ],
)
def test_lstm_run_key_rejects_invalid_fields(
    ticker: object,
    config: object,
    seed: object,
    message: str,
) -> None:
    """Reject unsupported tickers configurations and seeds"""
    with pytest.raises(ValueError, match=message):
        LSTMRunKey(
            ticker=cast(str, ticker),
            config=cast(LSTMConfig, config),
            seed=cast(int, seed),
        )


def test_run_lstm_candidate_executes_leakage_safe_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select epochs refit train and predict validation safely"""
    dataset = make_dataset()
    original = dataset.copy(deep=True)
    config = LSTMConfig(window=5, units=16, dropout=0.2)
    model = RecordingModel(prediction=0.01, parameter_count=73)
    trainer = RecordingTrainer(model, best_epoch=3)
    scaler_calls: list[tuple[pd.DataFrame, str, pd.Timestamp]] = []
    actual_fit_scaler = fit_return_scaler

    def record_scaler(
        frame: pd.DataFrame,
        ticker: str,
        through_date: pd.Timestamp,
    ) -> StandardScaler:
        scaler_calls.append((frame.copy(deep=True), ticker, pd.Timestamp(through_date)))
        return actual_fit_scaler(frame, ticker, through_date)

    monkeypatch.setattr(validation_module, "fit_return_scaler", record_scaler)

    result = run_candidate(dataset, trainer, config=config)

    pd.testing.assert_frame_equal(dataset, original)
    train_batch = build_return_sequences(dataset, "AAPL", 5, "train")
    inner_train, inner_validation = split_sequence_batch(train_batch)
    inner_cutoff = inner_train.keys["date"].max()
    full_cutoff = train_batch.keys["date"].max()
    inner_scaler = actual_fit_scaler(dataset, "AAPL", inner_cutoff)
    full_scaler = actual_fit_scaler(dataset, "AAPL", full_cutoff)

    assert len(trainer.determine_calls) == 1
    determine_call = trainer.determine_calls[0]
    assert determine_call["config"] == config
    assert determine_call["seed"] == 3
    np.testing.assert_allclose(
        recorded_array(determine_call, "train_features"),
        transform_features(inner_train.features, inner_scaler),
    )
    np.testing.assert_allclose(
        recorded_array(determine_call, "train_targets"),
        transform_targets(inner_train.targets, inner_scaler),
    )
    np.testing.assert_allclose(
        recorded_array(determine_call, "validation_features"),
        transform_features(inner_validation.features, inner_scaler),
    )
    np.testing.assert_allclose(
        recorded_array(determine_call, "validation_targets"),
        transform_targets(inner_validation.targets, inner_scaler),
    )

    assert len(trainer.fit_calls) == 1
    fit_call = trainer.fit_calls[0]
    assert fit_call["config"] == config
    assert fit_call["seed"] == 3
    assert fit_call["epochs"] == 3
    assert fit_call["name"] == "lstm_w5_u16_d20_seed_03"
    assert fit_call["feature_columns"] == lag_columns(5)
    np.testing.assert_allclose(
        recorded_array(fit_call, "features"),
        transform_features(train_batch.features, full_scaler),
    )
    np.testing.assert_allclose(
        recorded_array(fit_call, "targets"),
        transform_targets(train_batch.targets, full_scaler),
    )
    assert cast(StandardScaler, fit_call["scaler"]).mean_ == pytest.approx(full_scaler.mean_)

    assert [call[2] for call in scaler_calls] == [inner_cutoff, full_cutoff]
    assert all(call[1] == "AAPL" for call in scaler_calls)
    assert all(not call[0]["split"].eq("test").any() for call in scaler_calls)
    assert model.received is not None
    assert model.received.columns.tolist() == list(lag_columns(5))
    assert model.received.index.tolist() == dataset.index[10:13].tolist()
    assert result.status == "ok"
    assert result.best_epoch == 3
    assert result.parameter_count == 73
    assert result.error is None
    assert result.predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert result.predictions.index.tolist() == dataset.index[10:13].tolist()
    assert result.predictions["split"].eq("validation").all()
    assert result.predictions["target_date"].is_unique
    assert result.predictions["model"].eq("lstm_w5_u16_d20_seed_03").all()
    assert result.predictions["actual_return"].tolist() == pytest.approx(
        dataset.loc[dataset["split"].eq("validation"), "target_return"].tolist()
    )
    assert result.predictions["predicted_return"].tolist() == pytest.approx([0.01] * 3)
    errors = result.predictions["actual_return"].to_numpy() - 0.01
    assert result.mae_bps == pytest.approx(np.mean(np.abs(errors)) * BPS_FACTOR)
    assert result.rmse_bps == pytest.approx(np.sqrt(np.mean(np.square(errors))) * BPS_FACTOR)


@pytest.mark.parametrize("prediction", [np.nan, np.inf, -np.inf])
def test_run_lstm_candidate_records_nonfinite_predictions(prediction: float) -> None:
    """Classify nonfinite forecasts without aborting grid execution"""
    result = run_candidate(make_dataset(), RecordingTrainer(RecordingModel(prediction)))

    assert result.status == "non_finite"
    assert result.best_epoch == 3
    assert result.parameter_count == 73
    assert result.predictions.empty
    assert result.predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert np.isnan(result.mae_bps)
    assert np.isnan(result.rmse_bps)
    assert result.error is not None
    assert result.key.run_id in result.error
    assert "finite" in result.error


def test_run_lstm_candidate_classifies_model_nonfinite_exception() -> None:
    """Recognize finite-value rejection raised inside fitted model"""
    model = RecordingModel(error=ValueError("values must contain only finite values"))

    result = run_candidate(make_dataset(), RecordingTrainer(model))

    assert result.status == "non_finite"
    assert result.parameter_count == 73


@pytest.mark.parametrize("failure_stage", ["determine", "fit", "predict"])
def test_run_lstm_candidate_records_runtime_failures(failure_stage: str) -> None:
    """Capture independent training and prediction failures contextually"""
    error = RuntimeError(f"{failure_stage} exploded")
    model = RecordingModel(error=error if failure_stage == "predict" else None)
    trainer = RecordingTrainer(
        model,
        determine_error=error if failure_stage == "determine" else None,
        fit_error=error if failure_stage == "fit" else None,
    )

    result = run_candidate(make_dataset(), trainer)

    assert result.status == "failed"
    assert result.best_epoch == (None if failure_stage == "determine" else 3)
    assert result.parameter_count == (73 if failure_stage == "predict" else None)
    assert result.predictions.empty
    assert result.predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert np.isnan(result.mae_bps)
    assert np.isnan(result.rmse_bps)
    assert result.error is not None
    assert result.key.run_id in result.error
    assert f"{failure_stage} exploded" in result.error


def test_run_lstm_candidate_records_malformed_prediction_contract() -> None:
    """Capture prediction alignment failures as terminal results"""
    result = run_candidate(
        make_dataset(),
        RecordingTrainer(RecordingModel(malformed=True)),
    )

    assert result.status == "failed"
    assert result.error is not None
    assert "length" in result.error


def test_run_lstm_candidate_rejects_invalid_key_type() -> None:
    """Reject malformed run keys before preparing any arrays"""
    with pytest.raises(ValueError, match="key"):
        run_lstm_candidate(
            make_dataset(),
            cast(LSTMRunKey, object()),
            cast(LSTMTrainer, RecordingTrainer()),
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.drop(columns="log_return"),
        lambda frame: frame.assign(split="test"),
        lambda frame: frame.assign(ticker="JPM"),
    ],
)
def test_run_lstm_candidate_raises_for_structural_dataset_errors(
    mutator: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Raise structural errors before creating terminal statuses"""
    invalid = mutator(make_dataset())

    with pytest.raises(ValueError):
        run_candidate(invalid, RecordingTrainer())


def terminal_result(key: LSTMRunKey, status: str = "failed") -> LSTMRunResult:
    """Build one lightweight result for grid scheduling tests"""
    return LSTMRunResult(
        key=key,
        status=status,  # type: ignore[arg-type]
        best_epoch=None,
        parameter_count=None,
        mae_bps=float("nan"),
        rmse_bps=float("nan"),
        predictions=pd.DataFrame(columns=PREDICTION_COLUMNS),
        error=f"controlled {status}",
    )


def test_run_lstm_grid_schedules_default_480_runs_without_test_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Schedule core grid deterministically and exclude final test"""
    dataset = make_core_dataset(include_extra=True)
    original = dataset.copy(deep=True)
    calls: list[tuple[LSTMRunKey, pd.DataFrame]] = []

    def fake_run(
        frame: pd.DataFrame,
        key: LSTMRunKey,
        trainer: LSTMTrainer,
    ) -> LSTMRunResult:
        del trainer
        calls.append((key, frame.copy(deep=True)))
        return terminal_result(key)

    monkeypatch.setattr(validation_module, "run_lstm_candidate", fake_run)
    monkeypatch.setattr("evaluation.lstm_checkpoint.load_lstm_runs", lambda *_: {})
    monkeypatch.setattr("evaluation.lstm_checkpoint.save_lstm_run", lambda *_: None)

    results = run_lstm_grid(
        dataset,
        "1" * 64,
        tmp_path / "artifacts" / "lstm",
        cast(LSTMTrainer, RecordingTrainer()),
    )

    pd.testing.assert_frame_equal(dataset, original)
    assert len(results) == 4 * 12 * 10 == 480
    assert len(calls) == 480
    assert [result.key for result in results] == [call[0] for call in calls]
    assert [key.ticker for key, _ in calls[:120]] == ["AAPL"] * 120
    assert tuple(dict.fromkeys(key.config for key, _ in calls[:120])) == build_lstm_grid()
    assert [key.seed for key, _ in calls[:10]] == list(range(10))
    assert all(set(frame["ticker"]) == set(CORE_TICKERS) for _, frame in calls)
    assert all(set(frame["split"]) == {"train", "validation"} for _, frame in calls)


def test_run_lstm_grid_resumes_all_terminal_statuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Skip cached statuses and report progress through completion"""
    from evaluation.lstm_checkpoint import build_lstm_fingerprint, save_lstm_run

    dataset = make_core_dataset()
    config = LSTMConfig(5, 16, 0.0)
    seeds = (0, 1)
    directory = tmp_path / "artifacts" / "lstm"
    fingerprint = build_lstm_fingerprint("1" * 64, (config,), seeds)
    cached = [
        terminal_result(LSTMRunKey("AAPL", config, 0), "failed"),
        terminal_result(LSTMRunKey("JPM", config, 0), "non_finite"),
        LSTMRunResult(
            key=LSTMRunKey("SPY", config, 0),
            status="ok",
            best_epoch=2,
            parameter_count=10,
            mae_bps=1.0,
            rmse_bps=1.0,
            predictions=pd.DataFrame(
                {
                    "date": [pd.Timestamp("2020-01-01")],
                    "target_date": [pd.Timestamp("2020-01-02")],
                    "ticker": ["SPY"],
                    "split": ["validation"],
                    "model": ["lstm_w5_u16_d00_seed_00"],
                    "actual_return": [0.0],
                    "predicted_return": [0.0],
                },
                columns=PREDICTION_COLUMNS,
            ),
            error=None,
        ),
    ]
    for result in cached:
        save_lstm_run(directory, fingerprint, result)
    executed: list[str] = []
    progress: list[tuple[int, int, str]] = []

    def fake_run(
        frame: pd.DataFrame,
        key: LSTMRunKey,
        trainer: LSTMTrainer,
    ) -> LSTMRunResult:
        del frame, trainer
        executed.append(key.run_id)
        return terminal_result(key)

    monkeypatch.setattr(validation_module, "run_lstm_candidate", fake_run)

    results = run_lstm_grid(
        dataset,
        "1" * 64,
        directory,
        cast(LSTMTrainer, RecordingTrainer()),
        configs=(config,),
        seeds=seeds,
        progress=lambda completed, total, run_id: progress.append((completed, total, run_id)),
    )

    expected_keys = [LSTMRunKey(ticker, config, seed) for ticker in CORE_TICKERS for seed in seeds]
    cached_ids = {result.key.run_id for result in cached}
    assert [result.key for result in results] == expected_keys
    assert executed == [key.run_id for key in expected_keys if key.run_id not in cached_ids]
    assert progress[0] == (len(cached), len(expected_keys), "")
    assert progress[-1] == (len(expected_keys), len(expected_keys), expected_keys[-1].run_id)


@pytest.mark.parametrize(
    ("configs", "seeds", "message"),
    [
        ((), (0,), "configs"),
        ((LSTMConfig(5, 16, 0.0),) * 2, (0,), "duplicate config"),
        ((LSTMConfig(5, 16, 0.0),), (), "seeds"),
        ((LSTMConfig(5, 16, 0.0),), (0, 0), "duplicate seed"),
        ((LSTMConfig(5, 16, 0.0),), (10,), "seed"),
    ],
)
def test_run_lstm_grid_rejects_invalid_schedules(
    tmp_path: Path,
    configs: tuple[LSTMConfig, ...],
    seeds: tuple[int, ...],
    message: str,
) -> None:
    """Reject empty duplicate and unsupported schedules"""
    with pytest.raises(ValueError, match=message):
        run_lstm_grid(
            make_core_dataset(),
            "1" * 64,
            tmp_path / "artifacts" / "lstm",
            cast(LSTMTrainer, RecordingTrainer()),
            configs=configs,
            seeds=seeds,
        )


def test_run_lstm_grid_rejects_invalid_config_type_and_progress(
    tmp_path: Path,
) -> None:
    """Reject nonconfig candidates and noncallable progress values"""
    dataset = make_core_dataset()
    directory = tmp_path / "artifacts" / "lstm"
    trainer = cast(LSTMTrainer, RecordingTrainer())
    with pytest.raises(ValueError, match="configs"):
        run_lstm_grid(
            dataset,
            "1" * 64,
            directory,
            trainer,
            configs=cast(tuple[LSTMConfig, ...], (object(),)),
            seeds=(0,),
        )
    with pytest.raises(ValueError, match="progress"):
        run_lstm_grid(
            dataset,
            "1" * 64,
            directory,
            trainer,
            configs=(LSTMConfig(5, 16, 0.0),),
            seeds=(0,),
            progress=cast(Callable[[int, int, str], None], 7),
        )


def test_run_lstm_grid_rejects_unexpected_cached_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject cache entries outside current fingerprint schedule"""
    config = LSTMConfig(5, 16, 0.0)
    unexpected = terminal_result(LSTMRunKey("AAPL", config, 1))
    monkeypatch.setattr(
        "evaluation.lstm_checkpoint.load_lstm_runs",
        lambda *_: {unexpected.key.run_id: unexpected},
    )

    with pytest.raises(ValueError, match="unexpected"):
        run_lstm_grid(
            make_core_dataset(),
            "1" * 64,
            tmp_path / "artifacts" / "lstm",
            cast(LSTMTrainer, RecordingTrainer()),
            configs=(config,),
            seeds=(0,),
        )


@pytest.mark.parametrize("failure", ["missing_ticker", "test_only"])
def test_run_lstm_grid_rejects_incomplete_evaluation_data(
    tmp_path: Path,
    failure: str,
) -> None:
    """Require train and validation rows for every core ticker"""
    dataset = make_core_dataset()
    if failure == "missing_ticker":
        dataset = dataset.loc[dataset["ticker"].ne("XOM")]
    else:
        dataset = dataset.assign(split="test")

    with pytest.raises(ValueError, match="ticker|train.*validation"):
        run_lstm_grid(
            dataset,
            "1" * 64,
            tmp_path / "artifacts" / "lstm",
            cast(LSTMTrainer, RecordingTrainer()),
            configs=(LSTMConfig(5, 16, 0.0),),
            seeds=(0,),
        )
