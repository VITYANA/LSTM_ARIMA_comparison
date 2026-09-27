"""Tests for fingerprinted resumable ARIMA residual checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import evaluation.residual_checkpoint as checkpoint_module
from evaluation.arima_residuals import FIXED_ARIMA_ORDERS
from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.residual_checkpoint import (
    build_residual_fingerprint,
    load_residual_predictions,
    residual_dataset_digest,
    run_arima_residual_backtest,
    save_residual_predictions,
)
from evaluation.walk_forward import build_walk_forward_predictions
from models.arima import ARIMAOrder
from models.interfaces import ForecastModel

FINGERPRINT = "a" * 64
DATASET_DIGEST = "1" * 64
DATASET_COLUMNS = [
    "date",
    "target_date",
    "ticker",
    "adj_close",
    "log_return",
    "target_return",
    "split",
]


def checkpoint_directory(tmp_path: Path) -> Path:
    """Return an allowed isolated residual checkpoint directory"""
    return tmp_path / "artifacts" / "arima_residuals"


def file_digest(path: Path) -> str:
    """Hash file bytes independently from production code"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_dataset() -> pd.DataFrame:
    """Build core ticker rows around exact warmup boundary"""
    row_count = 255
    dates = pd.bdate_range("2018-01-02", periods=row_count + 1)
    frames = []
    for position, ticker in enumerate(FIXED_ARIMA_ORDERS):
        values = (position + 1) * 0.001 + np.arange(row_count) / 1_000_000
        frames.append(
            pd.DataFrame(
                {
                    "date": dates[:-1],
                    "target_date": dates[1:],
                    "ticker": ticker,
                    "adj_close": 100.0 + 10 * position + np.arange(row_count),
                    "log_return": values,
                    "target_return": values + 0.0001,
                    "split": [
                        *(["train"] * 252),
                        *(["validation"] * 2),
                        "test",
                    ],
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=31)


def prediction_table(
    dataset: pd.DataFrame,
    ticker: str = "AAPL",
    predicted_return: float = 0.00025,
) -> pd.DataFrame:
    """Build complete eligible forecasts for one ticker"""
    eligible = (
        dataset.loc[dataset["ticker"].eq(ticker) & dataset["split"].isin(["train", "validation"])]
        .sort_values("target_date")
        .iloc[251:]
    )
    result = eligible.loc[:, ["date", "target_date", "ticker", "split"]].copy()
    result["model"] = "arima"
    result["actual_return"] = eligible["target_return"].to_numpy()
    result["predicted_return"] = predicted_return
    return result.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)


def residual_table() -> pd.DataFrame:
    """Build a small valid residual model dataset"""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-02", "2019-01-03"]),
            "target_date": pd.to_datetime(["2019-01-03", "2019-01-04"]),
            "ticker": ["AAPL", "AAPL"],
            "adj_close": [101.25, 102.5],
            "log_return": [0.0012345678912345678, -0.0002],
            "target_return": [-0.0002, 0.003456789123456789],
            "split": ["train", "validation"],
        }
    ).loc[:, DATASET_COLUMNS]


def test_build_residual_fingerprint_is_deterministic_sha256() -> None:
    """Return stable lowercase digest for complete protocol"""
    first = build_residual_fingerprint(DATASET_DIGEST, FIXED_ARIMA_ORDERS, 252)
    second = build_residual_fingerprint(DATASET_DIGEST, FIXED_ARIMA_ORDERS, 252)

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_build_residual_fingerprint_canonicalizes_ticker_order() -> None:
    """Ignore mapping insertion order while preserving orders"""
    reversed_orders = dict(reversed(tuple(FIXED_ARIMA_ORDERS.items())))

    assert build_residual_fingerprint(
        DATASET_DIGEST, reversed_orders, 252
    ) == build_residual_fingerprint(DATASET_DIGEST, FIXED_ARIMA_ORDERS, 252)


@pytest.mark.parametrize(
    ("dataset_digest", "orders", "warmup"),
    [
        ("2" * 64, dict(FIXED_ARIMA_ORDERS), 252),
        (
            DATASET_DIGEST,
            {**FIXED_ARIMA_ORDERS, "AAPL": (1, 0, 0)},
            252,
        ),
        (
            DATASET_DIGEST,
            {
                "AAPL": FIXED_ARIMA_ORDERS["JPM"],
                "JPM": FIXED_ARIMA_ORDERS["AAPL"],
                "SPY": FIXED_ARIMA_ORDERS["SPY"],
                "XOM": FIXED_ARIMA_ORDERS["XOM"],
            },
            252,
        ),
        (DATASET_DIGEST, dict(FIXED_ARIMA_ORDERS), 251),
    ],
)
def test_build_residual_fingerprint_changes_with_protocol_inputs(
    dataset_digest: str,
    orders: Mapping[str, ARIMAOrder],
    warmup: int,
) -> None:
    """Bind source digest orders and warmup boundary"""
    baseline = build_residual_fingerprint(DATASET_DIGEST, FIXED_ARIMA_ORDERS, 252)

    assert build_residual_fingerprint(dataset_digest, orders, warmup) != baseline


@pytest.mark.parametrize(
    "attribute",
    [
        "CHECKPOINT_FORMAT_VERSION",
        "PYTHON_VERSION",
        "NUMPY_VERSION",
        "PANDAS_VERSION",
        "STATSMODELS_VERSION",
    ],
)
def test_build_residual_fingerprint_binds_format_and_versions(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
) -> None:
    """Invalidate cache after format or runtime changes"""
    baseline = build_residual_fingerprint(DATASET_DIGEST, FIXED_ARIMA_ORDERS, 252)
    replacement: object = 2 if attribute == "CHECKPOINT_FORMAT_VERSION" else "9.9.9"

    monkeypatch.setattr(checkpoint_module, attribute, replacement)

    assert build_residual_fingerprint(DATASET_DIGEST, FIXED_ARIMA_ORDERS, 252) != baseline


@pytest.mark.parametrize(
    ("dataset_digest", "orders", "warmup", "message"),
    [
        ("invalid", dict(FIXED_ARIMA_ORDERS), 252, "SHA-256"),
        (DATASET_DIGEST, {}, 252, "orders"),
        (DATASET_DIGEST, {"": (0, 0, 0)}, 252, "ticker"),
        (DATASET_DIGEST, {"AAPL": (0, True, 0)}, 252, "order"),
        (DATASET_DIGEST, dict(FIXED_ARIMA_ORDERS), True, "warmup"),
        (DATASET_DIGEST, dict(FIXED_ARIMA_ORDERS), 0, "warmup"),
    ],
)
def test_build_residual_fingerprint_rejects_invalid_inputs(
    dataset_digest: str,
    orders: Mapping[str, ARIMAOrder],
    warmup: int,
    message: str,
) -> None:
    """Reject malformed protocol identity values"""
    with pytest.raises(ValueError, match=message):
        build_residual_fingerprint(dataset_digest, orders, warmup)


def test_residual_dataset_digest_is_canonical_and_stable() -> None:
    """Ignore row order while retaining exact values"""
    dataset = residual_table()

    first = residual_dataset_digest(dataset)
    copied = residual_dataset_digest(dataset.copy(deep=True))
    shuffled = residual_dataset_digest(dataset.iloc[::-1].reset_index(drop=True))

    assert first == copied == shuffled
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("date", pd.Timestamp("2019-01-01")),
        ("target_date", pd.Timestamp("2019-01-06")),
        ("ticker", "JPM"),
        ("adj_close", 101.2500000001),
        ("log_return", 0.001234567891234568),
        ("target_return", 0.004),
        ("split", "validation"),
    ],
)
def test_residual_dataset_digest_changes_with_content(
    column: str,
    replacement: object,
) -> None:
    """Bind every key numeric value and split"""
    baseline = residual_table()
    changed = baseline.copy()
    changed.loc[changed.index[0], column] = cast(Any, replacement)

    assert residual_dataset_digest(changed) != residual_dataset_digest(baseline)


@pytest.mark.parametrize("damage", ["date", "non_finite", "columns", "column_order"])
def test_residual_dataset_digest_rejects_invalid_tables(damage: str) -> None:
    """Reject malformed residual dataset contracts"""
    dataset = residual_table()
    if damage == "date":
        dataset["date"] = dataset["date"].astype(str)
    elif damage == "non_finite":
        dataset.loc[0, "log_return"] = np.inf
    elif damage == "columns":
        dataset = dataset.drop(columns="adj_close")
    else:
        dataset = dataset.loc[:, list(reversed(DATASET_COLUMNS))]

    with pytest.raises(ValueError):
        residual_dataset_digest(dataset)


def test_checkpoint_round_trip_preserves_predictions_and_metadata(tmp_path: Path) -> None:
    """Restore exact forecasts dates columns and metadata"""
    dataset = make_dataset()
    expected = prediction_table(dataset)
    expected["actual_return"] = [
        0.012345678912345678,
        -0.00012345678912345678,
        0.0009876543219876543,
    ]
    expected["predicted_return"] = [
        -0.019876543219876543,
        0.0009876543219876543,
        -0.00011111111111111112,
    ]
    directory = checkpoint_directory(tmp_path)

    save_residual_predictions(directory, FINGERPRINT, "AAPL", expected)
    restored = load_residual_predictions(directory, FINGERPRINT)

    assert set(restored) == {"AAPL"}
    pd.testing.assert_frame_equal(restored["AAPL"], expected, check_exact=True)
    assert restored["AAPL"].columns.tolist() == list(PREDICTION_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(restored["AAPL"]["date"])
    assert pd.api.types.is_datetime64_any_dtype(restored["AAPL"]["target_date"])

    metadata = json.loads((directory / "AAPL.json").read_text(encoding="utf-8"))
    assert metadata["ticker"] == "AAPL"
    assert metadata["rows"] == len(expected)
    assert metadata["predictions_file"] == "AAPL.csv"
    assert metadata["sha256"] == file_digest(directory / "AAPL.csv")


def test_save_residual_predictions_atomically_replaces_ticker(tmp_path: Path) -> None:
    """Replace both files without temporary leftovers"""
    dataset = make_dataset()
    directory = checkpoint_directory(tmp_path)
    save_residual_predictions(directory, FINGERPRINT, "AAPL", prediction_table(dataset))
    replacement = prediction_table(dataset, predicted_return=0.009)

    save_residual_predictions(directory, FINGERPRINT, "AAPL", replacement)

    loaded = load_residual_predictions(directory, FINGERPRINT)
    pd.testing.assert_frame_equal(loaded["AAPL"], replacement, check_exact=True)
    assert not list(directory.glob("*.tmp"))


@pytest.mark.parametrize("operation", ["load", "save"])
def test_checkpoint_rejects_incompatible_fingerprint(
    tmp_path: Path,
    operation: str,
) -> None:
    """Prevent mixing forecasts from different protocols"""
    dataset = make_dataset()
    directory = checkpoint_directory(tmp_path)
    predictions = prediction_table(dataset)
    save_residual_predictions(directory, FINGERPRINT, "AAPL", predictions)

    with pytest.raises(ValueError, match="fingerprint"):
        if operation == "load":
            load_residual_predictions(directory, "b" * 64)
        else:
            save_residual_predictions(directory, "b" * 64, "AAPL", predictions)


@pytest.mark.parametrize(
    "damage",
    [
        "json",
        "missing_json",
        "missing_csv",
        "csv_checksum",
        "columns",
        "non_finite",
        "unexpected_ticker",
        "unexpected_file",
    ],
)
def test_load_residual_predictions_rejects_corrupt_pairs(
    tmp_path: Path,
    damage: str,
) -> None:
    """Reject corrupt incomplete and unexpected ticker files"""
    dataset = make_dataset()
    directory = checkpoint_directory(tmp_path)
    save_residual_predictions(directory, FINGERPRINT, "AAPL", prediction_table(dataset))
    metadata_path = directory / "AAPL.json"
    predictions_path = directory / "AAPL.csv"
    if damage == "json":
        metadata_path.write_text("{invalid", encoding="utf-8")
    elif damage == "missing_json":
        metadata_path.unlink()
    elif damage == "missing_csv":
        predictions_path.unlink()
    elif damage == "csv_checksum":
        predictions_path.write_text("corrupt", encoding="utf-8")
    elif damage in {"columns", "non_finite"}:
        predictions = pd.read_csv(predictions_path)
        if damage == "columns":
            predictions = predictions.drop(columns="actual_return")
        else:
            predictions.loc[0, "predicted_return"] = np.inf
        predictions.to_csv(predictions_path, index=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["sha256"] = file_digest(predictions_path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "unexpected_ticker":
        (directory / "TSLA.csv").write_text("unexpected", encoding="utf-8")
        (directory / "TSLA.json").write_text("{}", encoding="utf-8")
    else:
        (directory / "orphan.tmp").write_text("partial", encoding="utf-8")

    with pytest.raises(ValueError):
        load_residual_predictions(directory, FINGERPRINT)


@pytest.mark.parametrize(
    "damage",
    [
        "root",
        "version",
        "fingerprint",
        "ticker",
        "filename",
        "rows",
        "row_mismatch",
        "checksum",
    ],
)
def test_load_residual_predictions_rejects_invalid_metadata(
    tmp_path: Path,
    damage: str,
) -> None:
    """Reject inconsistent ticker checkpoint metadata fields"""
    directory = checkpoint_directory(tmp_path)
    save_residual_predictions(directory, FINGERPRINT, "AAPL", prediction_table(make_dataset()))
    path = directory / "AAPL.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if damage == "root":
        value: object = []
    else:
        value = metadata
        if damage == "version":
            metadata["format_version"] = 2
        elif damage == "fingerprint":
            metadata["fingerprint"] = "b" * 64
        elif damage == "ticker":
            metadata["ticker"] = "JPM"
        elif damage == "filename":
            metadata["predictions_file"] = "JPM.csv"
        elif damage == "rows":
            metadata["rows"] = -1
        elif damage == "row_mismatch":
            metadata["rows"] += 1
        else:
            metadata["sha256"] = "invalid"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="AAPL"):
        load_residual_predictions(directory, FINGERPRINT)


@pytest.mark.parametrize("damage", ["json", "root", "version"])
def test_load_residual_predictions_rejects_invalid_manifest(
    tmp_path: Path,
    damage: str,
) -> None:
    """Reject malformed manifest syntax structure and version"""
    directory = checkpoint_directory(tmp_path)
    save_residual_predictions(directory, FINGERPRINT, "AAPL", prediction_table(make_dataset()))
    path = directory / "manifest.json"
    if damage == "json":
        content = "{invalid"
    elif damage == "root":
        content = "[]"
    else:
        content = json.dumps({"format_version": 2, "fingerprint": FINGERPRINT})
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        load_residual_predictions(directory, FINGERPRINT)


def invalid_predictions(dataset: pd.DataFrame, case: str) -> pd.DataFrame:
    """Build one deliberately invalid ARIMA prediction table"""
    predictions = prediction_table(dataset)
    if case == "columns":
        return predictions.drop(columns="actual_return")
    if case == "empty":
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    if case == "date":
        predictions["date"] = predictions["date"].astype(str)
    elif case == "target_date":
        predictions["target_date"] = predictions["target_date"].astype(str)
    elif case == "ticker":
        predictions["ticker"] = "JPM"
    elif case == "checkpoint_ticker":
        predictions["ticker"] = "TSLA"
    elif case == "split":
        predictions["split"] = "test"
    elif case == "model":
        predictions["model"] = "naive_zero"
    elif case == "duplicate":
        predictions.iloc[1] = predictions.iloc[0]
    elif case == "non_finite":
        predictions.loc[0, "predicted_return"] = np.inf
    return predictions


@pytest.mark.parametrize(
    "case",
    [
        "columns",
        "empty",
        "date",
        "target_date",
        "ticker",
        "checkpoint_ticker",
        "split",
        "model",
        "duplicate",
        "non_finite",
    ],
)
def test_save_residual_predictions_rejects_invalid_tables(
    tmp_path: Path,
    case: str,
) -> None:
    """Reject malformed ticker prediction contracts before writing"""
    ticker = "TSLA" if case == "checkpoint_ticker" else "AAPL"
    with pytest.raises(ValueError):
        save_residual_predictions(
            checkpoint_directory(tmp_path),
            FINGERPRINT,
            ticker,
            invalid_predictions(make_dataset(), case),
        )


def test_load_residual_predictions_returns_empty_for_new_directories(tmp_path: Path) -> None:
    """Treat absent and empty directories as unfinished"""
    directory = checkpoint_directory(tmp_path)
    assert load_residual_predictions(directory, FINGERPRINT) == {}

    directory.mkdir(parents=True)
    assert load_residual_predictions(directory, FINGERPRINT) == {}


@pytest.mark.parametrize("operation", ["load", "save"])
def test_checkpoint_rejects_nonempty_directory_without_manifest(
    tmp_path: Path,
    operation: str,
) -> None:
    """Reject orphaned files without checkpoint identity"""
    directory = checkpoint_directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / "orphan.txt").write_text("orphan", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        if operation == "load":
            load_residual_predictions(directory, FINGERPRINT)
        else:
            save_residual_predictions(
                directory,
                FINGERPRINT,
                "AAPL",
                prediction_table(make_dataset()),
            )


def test_checkpoint_rejects_invalid_locations_and_digests(tmp_path: Path) -> None:
    """Restrict checkpoints to artifacts and valid paths"""
    with pytest.raises(ValueError, match="artifacts"):
        load_residual_predictions(tmp_path / "outside", FINGERPRINT)
    with pytest.raises(ValueError, match="SHA-256"):
        load_residual_predictions(checkpoint_directory(tmp_path), "invalid")
    with pytest.raises(ValueError, match="Path"):
        load_residual_predictions(cast(Path, "artifacts/checkpoint"), FINGERPRINT)

    file_path = tmp_path / "artifacts" / "checkpoint"
    file_path.parent.mkdir()
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        load_residual_predictions(file_path, FINGERPRINT)


class ZeroModel:
    """Return aligned zero ARIMA forecasts."""

    @property
    def name(self) -> str:
        """Return the stable ARIMA model name"""
        return "arima"

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Return one zero for every observation"""
        return pd.Series(
            0.0,
            index=observations.index,
            dtype="float64",
            name="predicted_return",
        )


class RecordingFitter:
    """Record fitted tickers and optionally fail one."""

    def __init__(self, fail_ticker: str | None = None) -> None:
        self.fail_ticker = fail_ticker
        self.calls: list[str] = []
        self.orders: Mapping[str, ARIMAOrder] | None = None

    def construct(self, orders: Mapping[str, ARIMAOrder]) -> RecordingFitter:
        """Capture configured orders and return this fitter"""
        self.orders = orders
        return self

    def __call__(self, ticker: str, history: pd.DataFrame) -> ForecastModel:
        """Record ticker and return controlled model"""
        self.calls.append(ticker)
        if ticker == self.fail_ticker:
            raise RuntimeError("controlled ARIMA failure")
        return ZeroModel()


def test_run_arima_residual_backtest_resumes_completed_tickers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume cached tickers and persist fresh results immediately"""
    dataset = make_dataset()
    directory = checkpoint_directory(tmp_path)
    fingerprint = build_residual_fingerprint(DATASET_DIGEST, FIXED_ARIMA_ORDERS, 252)
    for ticker in ("AAPL", "JPM"):
        save_residual_predictions(
            directory,
            fingerprint,
            ticker,
            prediction_table(dataset, ticker),
        )

    fitter = RecordingFitter()
    monkeypatch.setattr(
        checkpoint_module,
        "ARIMAFitter",
        lambda orders: fitter.construct(orders),
    )
    original_walk_forward = build_walk_forward_predictions
    schedules: list[tuple[str, tuple[str, ...], int]] = []

    def recording_walk_forward(
        ticker_dataset: pd.DataFrame,
        model_fitter: Callable[[str, pd.DataFrame], ForecastModel],
        *,
        target_splits: tuple[str, ...],
        min_history: int,
        progress: Callable[[int, int], None] | None,
    ) -> pd.DataFrame:
        ticker = str(ticker_dataset["ticker"].iloc[0])
        if ticker == "XOM":
            assert (directory / "SPY.json").exists()
            assert (directory / "SPY.csv").exists()
        schedules.append((ticker, target_splits, min_history))
        return original_walk_forward(
            ticker_dataset,
            model_fitter,
            target_splits=target_splits,
            min_history=min_history,
            progress=progress,
        )

    monkeypatch.setattr(
        checkpoint_module,
        "build_walk_forward_predictions",
        recording_walk_forward,
    )
    progress_events: list[tuple[int, int, str]] = []

    result = run_arima_residual_backtest(
        dataset,
        DATASET_DIGEST,
        directory,
        progress=lambda completed, total, label: progress_events.append((completed, total, label)),
    )

    assert schedules == [
        ("SPY", ("train", "validation"), 252),
        ("XOM", ("train", "validation"), 252),
    ]
    assert set(fitter.calls) == {"SPY", "XOM"}
    assert fitter.orders == FIXED_ARIMA_ORDERS
    assert progress_events[0] == (6, 12, "checkpoint")
    assert progress_events[-1] == (12, 12, "XOM")
    assert result.arima_predictions.groupby("ticker").size().to_dict() == {
        "AAPL": 3,
        "JPM": 3,
        "SPY": 3,
        "XOM": 3,
    }
    assert result.residual_dataset.groupby("ticker").size().to_dict() == {
        "AAPL": 2,
        "JPM": 2,
        "SPY": 2,
        "XOM": 2,
    }
    assert not result.arima_predictions["split"].eq("test").any()
    assert list(load_residual_predictions(directory, fingerprint)) == [
        "AAPL",
        "JPM",
        "SPY",
        "XOM",
    ]


def test_run_arima_residual_backtest_preserves_completed_work_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep completed tickers while discarding partial failures"""
    dataset = make_dataset()
    directory = checkpoint_directory(tmp_path)
    fingerprint = build_residual_fingerprint(DATASET_DIGEST, FIXED_ARIMA_ORDERS, 252)
    save_residual_predictions(
        directory,
        fingerprint,
        "AAPL",
        prediction_table(dataset, "AAPL"),
    )
    fitter = RecordingFitter(fail_ticker="SPY")
    monkeypatch.setattr(
        checkpoint_module,
        "ARIMAFitter",
        lambda orders: fitter.construct(orders),
    )
    first_spy_origin = (
        dataset.loc[dataset["ticker"].eq("SPY") & dataset["split"].isin(["train", "validation"])]
        .sort_values("target_date")
        .iloc[251]["date"]
    )

    with pytest.raises(
        ValueError,
        match=rf"ticker 'SPY'.*origin {pd.Timestamp(first_spy_origin).date()}",
    ):
        run_arima_residual_backtest(dataset, DATASET_DIGEST, directory)

    loaded = load_residual_predictions(directory, fingerprint)
    assert list(loaded) == ["AAPL", "JPM"]
    assert not (directory / "SPY.json").exists()
    assert not (directory / "SPY.csv").exists()


@pytest.mark.parametrize(
    ("orders", "warmup", "progress", "message"),
    [
        ({"AAPL": (0, 0, 0)}, 252, None, "tickers"),
        (dict(FIXED_ARIMA_ORDERS), True, None, "warmup"),
        (dict(FIXED_ARIMA_ORDERS), 252, cast(Callable[[int, int, str], None], 1), "progress"),
    ],
)
def test_run_arima_residual_backtest_rejects_invalid_schedule(
    tmp_path: Path,
    orders: Mapping[str, ARIMAOrder],
    warmup: int,
    progress: Callable[[int, int, str], None] | None,
    message: str,
) -> None:
    """Reject invalid ticker warmup and callback schedules"""
    with pytest.raises(ValueError, match=message):
        run_arima_residual_backtest(
            make_dataset(),
            DATASET_DIGEST,
            checkpoint_directory(tmp_path),
            orders=orders,
            warmup=warmup,
            progress=progress,
        )
