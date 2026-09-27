"""Tests for fingerprinted atomic LSTM checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import evaluation.lstm_checkpoint as checkpoint_module
from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.lstm_checkpoint import (
    build_lstm_fingerprint,
    load_lstm_runs,
    save_lstm_run,
)
from evaluation.lstm_validation import LSTMRunKey, LSTMRunResult
from models.lstm import LSTMConfig

FINGERPRINT = "a" * 64


def checkpoint_directory(tmp_path: Path) -> Path:
    """Return an allowed isolated checkpoint directory"""
    return tmp_path / "artifacts" / "lstm_validation"


def prediction_table(value: float = 0.001) -> pd.DataFrame:
    """Build one finite validation prediction table"""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "target_date": pd.to_datetime(["2020-01-03", "2020-01-06"]),
            "ticker": ["AAPL", "AAPL"],
            "split": ["validation", "validation"],
            "model": ["lstm_w5_u16_d00_seed_00"] * 2,
            "actual_return": [0.002, -0.003],
            "predicted_return": [value, value],
        },
        index=pd.Index([41, 42], name="source_index"),
    )
    return frame.loc[:, PREDICTION_COLUMNS]


def run_result(
    *,
    status: str = "ok",
    prediction: float = 0.001,
    seed: int = 0,
) -> LSTMRunResult:
    """Build one controlled terminal run result"""
    successful = status == "ok"
    return LSTMRunResult(
        key=LSTMRunKey("AAPL", LSTMConfig(5, 16, 0.0), seed),
        status=status,  # type: ignore[arg-type]
        best_epoch=7 if successful else None,
        parameter_count=73 if successful else None,
        mae_bps=12.5 if successful else float("nan"),
        rmse_bps=15.25 if successful else float("nan"),
        predictions=(
            prediction_table(prediction) if successful else pd.DataFrame(columns=PREDICTION_COLUMNS)
        ),
        error=None if successful else f"controlled {status}",
    )


def test_build_lstm_fingerprint_is_deterministic_sha256() -> None:
    """Return stable digest for one complete protocol"""
    configs = (LSTMConfig(5, 16, 0.0), LSTMConfig(21, 32, 0.2))

    first = build_lstm_fingerprint("1" * 64, configs, (0, 1))
    second = build_lstm_fingerprint("1" * 64, configs, (0, 1))

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("dataset_digest", "configs", "seeds"),
    [
        ("2" * 64, (LSTMConfig(5, 16, 0.0), LSTMConfig(21, 32, 0.2)), (0, 1)),
        ("1" * 64, (LSTMConfig(21, 32, 0.2), LSTMConfig(5, 16, 0.0)), (0, 1)),
        ("1" * 64, (LSTMConfig(5, 32, 0.0), LSTMConfig(21, 32, 0.2)), (0, 1)),
        ("1" * 64, (LSTMConfig(5, 16, 0.0), LSTMConfig(21, 32, 0.2)), (1, 0)),
    ],
)
def test_build_lstm_fingerprint_changes_with_inputs(
    dataset_digest: str,
    configs: tuple[LSTMConfig, ...],
    seeds: tuple[int, ...],
) -> None:
    """Bind digest configuration order values and seeds"""
    baseline = build_lstm_fingerprint(
        "1" * 64,
        (LSTMConfig(5, 16, 0.0), LSTMConfig(21, 32, 0.2)),
        (0, 1),
    )

    assert build_lstm_fingerprint(dataset_digest, configs, seeds) != baseline


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("INNER_TRAIN_FRACTION", 0.75),
        ("OPTIMIZER_NAME", "SGD"),
        ("LOSS_NAME", "mae"),
        ("BATCH_SIZE", 64),
        ("MAX_EPOCHS", 100),
        ("EARLY_STOPPING_PATIENCE", 10),
        ("PYTHON_VERSION", "9.9.9"),
        ("NUMPY_VERSION", "9.9.9"),
        ("PANDAS_VERSION", "9.9.9"),
        ("SKLEARN_VERSION", "9.9.9"),
        ("TENSORFLOW_VERSION", "9.9.9"),
    ],
)
def test_build_lstm_fingerprint_binds_protocol_and_versions(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    replacement: object,
) -> None:
    """Invalidate cache after protocol or runtime changes"""
    config = (LSTMConfig(5, 16, 0.0),)
    baseline = build_lstm_fingerprint("1" * 64, config, (0,))

    monkeypatch.setattr(checkpoint_module, attribute, replacement)

    assert build_lstm_fingerprint("1" * 64, config, (0,)) != baseline


@pytest.mark.parametrize(
    ("configs", "seeds", "message"),
    [
        ((), (0,), "configs"),
        ((object(),), (0,), "configs"),
        ((LSTMConfig(5, 16, 0.0),), (), "seeds"),
        ((LSTMConfig(5, 16, 0.0),), ("0",), "seeds"),
    ],
)
def test_build_lstm_fingerprint_rejects_invalid_search_space(
    configs: tuple[object, ...],
    seeds: tuple[object, ...],
    message: str,
) -> None:
    """Reject empty and incorrectly typed fingerprint inputs"""
    with pytest.raises(ValueError, match=message):
        build_lstm_fingerprint(
            "1" * 64,
            cast(tuple[LSTMConfig, ...], configs),
            cast(tuple[int, ...], seeds),
        )


@pytest.mark.parametrize("status", ["ok", "failed", "non_finite"])
def test_checkpoint_round_trip_preserves_terminal_results(
    tmp_path: Path,
    status: str,
) -> None:
    """Restore metadata dates indexes metrics and statuses"""
    directory = checkpoint_directory(tmp_path)
    expected = run_result(status=status)

    save_lstm_run(directory, FINGERPRINT, expected)
    loaded = load_lstm_runs(directory, FINGERPRINT)

    assert set(loaded) == {expected.key.run_id}
    actual = loaded[expected.key.run_id]
    assert actual.key == expected.key
    assert actual.status == expected.status
    assert actual.best_epoch == expected.best_epoch
    assert actual.parameter_count == expected.parameter_count
    if status == "ok":
        assert actual.mae_bps == pytest.approx(expected.mae_bps)
        assert actual.rmse_bps == pytest.approx(expected.rmse_bps)
    else:
        assert np.isnan(actual.mae_bps)
        assert np.isnan(actual.rmse_bps)
    assert actual.error == expected.error
    pd.testing.assert_frame_equal(actual.predictions, expected.predictions, check_freq=False)
    assert actual.predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    if status == "ok":
        assert pd.api.types.is_datetime64_any_dtype(actual.predictions["date"])
        assert pd.api.types.is_datetime64_any_dtype(actual.predictions["target_date"])


def test_save_lstm_run_atomically_replaces_existing_run(tmp_path: Path) -> None:
    """Replace both files without leaving temporary siblings"""
    directory = checkpoint_directory(tmp_path)
    first = run_result(prediction=0.001)
    replacement = run_result(prediction=0.009)

    save_lstm_run(directory, FINGERPRINT, first)
    save_lstm_run(directory, FINGERPRINT, replacement)

    loaded = load_lstm_runs(directory, FINGERPRINT)[replacement.key.run_id]
    assert loaded.predictions["predicted_return"].tolist() == pytest.approx([0.009, 0.009])
    assert not list(directory.glob("*.tmp"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint"] == FINGERPRINT


def invalid_result(case: str) -> object:
    """Build one deliberately inconsistent persistence result"""
    success = run_result()
    failure = run_result(status="failed")
    if case == "type":
        return object()
    if case == "status":
        return replace(success, status="unknown")  # type: ignore[arg-type]
    if case == "epoch_type":
        return replace(success, best_epoch=True)
    if case == "epoch_value":
        return replace(success, best_epoch=0)
    if case == "parameters_type":
        return replace(success, parameter_count=True)
    if case == "parameters_value":
        return replace(success, parameter_count=-1)
    if case == "metrics":
        return replace(success, mae_bps=np.inf)
    if case == "success_error":
        return replace(success, error="unexpected")
    if case == "failure_metrics":
        return replace(failure, mae_bps=1.0)
    if case == "failure_error":
        return replace(failure, error=None)
    predictions = success.predictions.copy()
    if case == "columns":
        predictions = predictions.drop(columns="actual_return")
    elif case == "failure_predictions":
        return replace(failure, predictions=predictions)
    elif case == "empty":
        predictions = pd.DataFrame(columns=PREDICTION_COLUMNS)
    elif case == "date":
        predictions["date"] = predictions["date"].astype(str)
    elif case == "target_date":
        predictions["target_date"] = predictions["target_date"].astype(str)
    elif case == "ticker":
        predictions["ticker"] = "JPM"
    elif case == "split":
        predictions["split"] = "train"
    elif case == "model":
        predictions["model"] = "wrong"
    elif case == "duplicate":
        predictions.iloc[1] = predictions.iloc[0]
    return replace(success, predictions=predictions)


@pytest.mark.parametrize(
    "case",
    [
        "type",
        "status",
        "epoch_type",
        "epoch_value",
        "parameters_type",
        "parameters_value",
        "metrics",
        "success_error",
        "failure_metrics",
        "failure_error",
        "columns",
        "failure_predictions",
        "empty",
        "date",
        "target_date",
        "ticker",
        "split",
        "model",
        "duplicate",
    ],
)
def test_save_lstm_run_rejects_inconsistent_results(
    tmp_path: Path,
    case: str,
) -> None:
    """Reject inconsistent terminal metadata and predictions"""
    with pytest.raises(ValueError):
        save_lstm_run(
            checkpoint_directory(tmp_path),
            FINGERPRINT,
            cast(LSTMRunResult, invalid_result(case)),
        )


@pytest.mark.parametrize("operation", ["load", "save"])
def test_checkpoint_rejects_incompatible_fingerprint(
    tmp_path: Path,
    operation: str,
) -> None:
    """Prevent mixing results from different protocols"""
    directory = checkpoint_directory(tmp_path)
    result = run_result()
    save_lstm_run(directory, FINGERPRINT, result)

    with pytest.raises(ValueError, match="fingerprint"):
        if operation == "load":
            load_lstm_runs(directory, "b" * 64)
        else:
            save_lstm_run(directory, "b" * 64, result)


@pytest.mark.parametrize("damage", ["json", "missing_csv", "columns", "non_finite"])
def test_load_lstm_runs_rejects_malformed_run_files(
    tmp_path: Path,
    damage: str,
) -> None:
    """Reject malformed checkpoint files with run context"""
    directory = checkpoint_directory(tmp_path)
    result = run_result()
    save_lstm_run(directory, FINGERPRINT, result)
    metadata_path = directory / f"{result.key.run_id}.json"
    predictions_path = directory / f"{result.key.run_id}.csv"
    if damage == "json":
        metadata_path.write_text("{invalid", encoding="utf-8")
    elif damage == "missing_csv":
        predictions_path.unlink()
    else:
        predictions = pd.read_csv(predictions_path, index_col=0)
        if damage == "columns":
            predictions = predictions.drop(columns="actual_return")
        else:
            predictions.loc[predictions.index[0], "predicted_return"] = np.inf
        predictions.to_csv(predictions_path)

    with pytest.raises(ValueError, match=result.key.run_id):
        load_lstm_runs(directory, FINGERPRINT)


@pytest.mark.parametrize(
    "damage",
    ["root", "version", "fingerprint", "run_id", "key", "config", "filename", "status"],
)
def test_load_lstm_runs_rejects_invalid_metadata_fields(
    tmp_path: Path,
    damage: str,
) -> None:
    """Reject incompatible and structurally invalid metadata fields"""
    directory = checkpoint_directory(tmp_path)
    result = run_result()
    save_lstm_run(directory, FINGERPRINT, result)
    path = directory / f"{result.key.run_id}.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if damage == "root":
        value: object = []
    else:
        value = metadata
        if damage == "version":
            metadata["format_version"] = 2
        elif damage == "fingerprint":
            metadata["fingerprint"] = "b" * 64
        elif damage == "run_id":
            metadata["run_id"] = "wrong"
        elif damage == "key":
            metadata["key"] = None
        elif damage == "config":
            metadata["key"]["config"] = None
        elif damage == "filename":
            metadata["predictions_file"] = "wrong.csv"
        else:
            metadata["status"] = "unknown"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=result.key.run_id):
        load_lstm_runs(directory, FINGERPRINT)


@pytest.mark.parametrize("damage", ["json", "root", "version"])
def test_load_lstm_runs_rejects_invalid_manifest(
    tmp_path: Path,
    damage: str,
) -> None:
    """Reject malformed manifest syntax structure and version"""
    directory = checkpoint_directory(tmp_path)
    result = run_result()
    save_lstm_run(directory, FINGERPRINT, result)
    path = directory / "manifest.json"
    if damage == "json":
        content = "{invalid"
    elif damage == "root":
        content = "[]"
    else:
        content = json.dumps({"format_version": 2, "fingerprint": FINGERPRINT})
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        load_lstm_runs(directory, FINGERPRINT)


@pytest.mark.parametrize(
    ("directory_factory", "fingerprint", "message"),
    [
        (lambda path: path / "outside", FINGERPRINT, "artifacts"),
        (lambda path: path / "artifacts" / "runs", "invalid", "fingerprint"),
    ],
)
def test_checkpoint_rejects_invalid_location_and_digest(
    tmp_path: Path,
    directory_factory: Callable[[Path], Path],
    fingerprint: str,
    message: str,
) -> None:
    """Restrict checkpoints to artifacts and SHA fingerprints"""
    directory = directory_factory(tmp_path)

    with pytest.raises(ValueError, match=message):
        load_lstm_runs(directory, fingerprint)


def test_load_lstm_runs_returns_empty_for_new_directory(tmp_path: Path) -> None:
    """Treat an absent checkpoint directory as no completed runs"""
    assert load_lstm_runs(checkpoint_directory(tmp_path), FINGERPRINT) == {}


def test_load_lstm_runs_returns_empty_for_existing_directory(tmp_path: Path) -> None:
    """Treat an existing empty directory as no completed runs"""
    directory = checkpoint_directory(tmp_path)
    directory.mkdir(parents=True)

    assert load_lstm_runs(directory, FINGERPRINT) == {}


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
            load_lstm_runs(directory, FINGERPRINT)
        else:
            save_lstm_run(directory, FINGERPRINT, run_result())


def test_checkpoint_rejects_non_path_and_file_location(tmp_path: Path) -> None:
    """Reject wrong path types and existing regular files"""
    with pytest.raises(ValueError, match="Path"):
        load_lstm_runs(cast(Path, "artifacts/lstm"), FINGERPRINT)

    file_path = tmp_path / "artifacts" / "checkpoint"
    file_path.parent.mkdir()
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        load_lstm_runs(file_path, FINGERPRINT)
