"""Persist fingerprinted LSTM validation runs without pickle."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
import tensorflow as tf  # type: ignore[import-untyped]

from evaluation.contracts import PREDICTION_COLUMNS, require_finite_numeric
from evaluation.lstm_validation import LSTMRunKey, LSTMRunResult, RunStatus
from models.lstm import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    LSTMConfig,
)

CHECKPOINT_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
INNER_TRAIN_FRACTION = 0.8
OPTIMIZER_NAME = "Adam"
LOSS_NAME = "mse"
PYTHON_VERSION = platform.python_version()
NUMPY_VERSION = np.__version__
PANDAS_VERSION = pd.__version__
SKLEARN_VERSION = sklearn.__version__
TENSORFLOW_VERSION = tf.__version__
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _validate_digest(value: object, label: str) -> str:
    """Return one lowercase SHA-256 digest."""
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _fingerprint_payload(
    dataset_sha256: str,
    configs: Sequence[LSTMConfig],
    seeds: Sequence[int],
) -> dict[str, object]:
    """Return the complete ordered protocol identity."""
    return {
        "dataset_sha256": dataset_sha256,
        "configs": [
            {
                "window": config.window,
                "units": config.units,
                "dropout": config.dropout,
            }
            for config in configs
        ],
        "seeds": list(seeds),
        "training": {
            "inner_train_fraction": INNER_TRAIN_FRACTION,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "loss": LOSS_NAME,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "shuffle": False,
        },
        "versions": {
            "python": PYTHON_VERSION,
            "numpy": NUMPY_VERSION,
            "pandas": PANDAS_VERSION,
            "scikit_learn": SKLEARN_VERSION,
            "tensorflow": TENSORFLOW_VERSION,
        },
    }


def build_lstm_fingerprint(
    dataset_sha256: str,
    configs: Sequence[LSTMConfig],
    seeds: Sequence[int],
) -> str:
    """Hash dataset, ordered search space and runtime protocol."""
    digest = _validate_digest(dataset_sha256, "dataset_sha256")
    config_values = tuple(configs)
    seed_values = tuple(seeds)
    if not config_values or not all(isinstance(config, LSTMConfig) for config in config_values):
        raise ValueError("configs must contain LSTMConfig values")
    if not seed_values or not all(type(seed) is int for seed in seed_values):
        raise ValueError("seeds must contain integers")
    payload = _fingerprint_payload(digest, config_values, seed_values)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_directory(directory: object) -> Path:
    """Return a checkpoint directory restricted to artifacts."""
    if not isinstance(directory, Path):
        raise ValueError("checkpoint directory must be a Path")
    if "artifacts" not in directory.parts:
        raise ValueError("checkpoint directory must be under artifacts")
    if directory.exists() and not directory.is_dir():
        raise ValueError("checkpoint path must be a directory")
    return directory


def _atomic_text(path: Path, content: str) -> None:
    """Replace one text file only after complete serialization."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_payload(fingerprint: str) -> dict[str, object]:
    """Return human-readable checkpoint identity metadata."""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "fingerprint": fingerprint,
    }


def _read_manifest(directory: Path, fingerprint: str) -> Mapping[str, object] | None:
    """Read and validate an existing checkpoint manifest."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("manifest root must be an object")
        manifest = cast(Mapping[str, object], value)
        if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("checkpoint format version is incompatible")
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("checkpoint fingerprint is incompatible")
        return manifest
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint manifest: {error}") from error


def _ensure_manifest(directory: Path, fingerprint: str) -> None:
    """Create or verify the checkpoint manifest."""
    manifest = _read_manifest(directory, fingerprint)
    if manifest is not None:
        return
    unexpected = [path for path in directory.iterdir() if not path.name.endswith(".tmp")]
    if unexpected:
        raise ValueError("checkpoint manifest is missing from a non-empty directory")
    content = json.dumps(_manifest_payload(fingerprint), indent=2, sort_keys=True)
    _atomic_text(directory / MANIFEST_NAME, f"{content}\n")


def _validate_predictions(result: LSTMRunResult) -> None:
    """Validate prediction contents before persistence or after loading."""
    predictions = result.predictions
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError("predictions must use standard columns in standard order")
    if result.status != "ok":
        if not predictions.empty:
            raise ValueError("unsuccessful runs must have empty predictions")
        return
    if predictions.empty:
        raise ValueError("successful runs must contain predictions")
    if not pd.api.types.is_datetime64_any_dtype(predictions["date"]):
        raise ValueError("prediction date must be datetime")
    if not pd.api.types.is_datetime64_any_dtype(predictions["target_date"]):
        raise ValueError("prediction target_date must be datetime")
    if not predictions["ticker"].eq(result.key.ticker).all():
        raise ValueError("prediction ticker does not match run key")
    if not predictions["split"].eq("validation").all():
        raise ValueError("predictions must contain validation rows only")
    expected_model = result.key.run_id.removeprefix(f"{result.key.ticker}_")
    if not predictions["model"].eq(expected_model).all():
        raise ValueError("prediction model does not match run key")
    duplicate_columns = ["date", "target_date", "ticker", "split", "model"]
    if predictions.duplicated(duplicate_columns).any():
        raise ValueError("predictions contain duplicate observation keys")
    require_finite_numeric(predictions["actual_return"], "actual_return")
    require_finite_numeric(predictions["predicted_return"], "predicted_return")


def _validate_result(result: object) -> LSTMRunResult:
    """Return one internally consistent terminal result."""
    if not isinstance(result, LSTMRunResult):
        raise ValueError("result must be an LSTMRunResult")
    if result.status not in {"ok", "failed", "non_finite"}:
        raise ValueError("result has invalid status")
    if result.best_epoch is not None and (
        type(result.best_epoch) is not int or result.best_epoch <= 0
    ):
        raise ValueError("best_epoch must be a positive integer or None")
    if result.parameter_count is not None and (
        type(result.parameter_count) is not int or result.parameter_count < 0
    ):
        raise ValueError("parameter_count must be a nonnegative integer or None")
    if result.status == "ok":
        if not np.isfinite([result.mae_bps, result.rmse_bps]).all():
            raise ValueError("successful metrics must be finite")
        if result.error is not None:
            raise ValueError("successful runs must not contain an error")
    else:
        if not np.isnan(result.mae_bps) or not np.isnan(result.rmse_bps):
            raise ValueError("unsuccessful metrics must be NaN")
        if not isinstance(result.error, str) or not result.error:
            raise ValueError("unsuccessful runs must contain an error")
    _validate_predictions(result)
    return result


def _metadata(result: LSTMRunResult, fingerprint: str) -> dict[str, object]:
    """Serialize one run without non-standard JSON numbers."""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "run_id": result.key.run_id,
        "key": {
            "ticker": result.key.ticker,
            "config": {
                "window": result.key.config.window,
                "units": result.key.config.units,
                "dropout": result.key.config.dropout,
            },
            "seed": result.key.seed,
        },
        "status": result.status,
        "best_epoch": result.best_epoch,
        "parameter_count": result.parameter_count,
        "mae_bps": result.mae_bps if result.status == "ok" else None,
        "rmse_bps": result.rmse_bps if result.status == "ok" else None,
        "predictions_file": f"{result.key.run_id}.csv",
        "error": result.error,
    }


def save_lstm_run(
    directory: Path,
    fingerprint: str,
    result: LSTMRunResult,
) -> None:
    """Atomically persist one terminal result as JSON and CSV."""
    target_directory = _validate_directory(directory)
    validated_fingerprint = _validate_digest(fingerprint, "fingerprint")
    validated_result = _validate_result(result)
    target_directory.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(target_directory, validated_fingerprint)

    metadata_path = target_directory / f"{validated_result.key.run_id}.json"
    predictions_path = target_directory / f"{validated_result.key.run_id}.csv"
    predictions_temporary = predictions_path.with_suffix(".csv.tmp")
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    try:
        validated_result.predictions.to_csv(predictions_temporary)
        serialized = json.dumps(
            _metadata(validated_result, validated_fingerprint),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        metadata_temporary.write_text(f"{serialized}\n", encoding="utf-8")
        predictions_temporary.replace(predictions_path)
        metadata_temporary.replace(metadata_path)
    finally:
        predictions_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Return one decoded JSON object."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _load_run(path: Path, fingerprint: str) -> LSTMRunResult:
    """Load one validated JSON/CSV run pair."""
    run_context = path.stem
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
        metadata = _mapping(value, "metadata")
        if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("run format version is incompatible")
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError("run fingerprint is incompatible")
        if metadata.get("run_id") != run_context:
            raise ValueError("run id does not match metadata filename")
        key_data = _mapping(metadata.get("key"), "key")
        config_data = _mapping(key_data.get("config"), "config")
        key = LSTMRunKey(
            ticker=cast(str, key_data.get("ticker")),
            config=LSTMConfig(
                window=cast(int, config_data.get("window")),
                units=cast(int, config_data.get("units")),
                dropout=cast(float, config_data.get("dropout")),
            ),
            seed=cast(int, key_data.get("seed")),
        )
        predictions_name = metadata.get("predictions_file")
        if not isinstance(predictions_name, str) or predictions_name != f"{run_context}.csv":
            raise ValueError("predictions filename is invalid")
        predictions_path = path.parent / predictions_name
        predictions = pd.read_csv(predictions_path, index_col=0)
        if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
            raise ValueError("predictions must use standard columns in standard order")
        status = metadata.get("status")
        if status not in {"ok", "failed", "non_finite"}:
            raise ValueError("run status is invalid")
        if status == "ok":
            predictions["date"] = pd.to_datetime(predictions["date"])
            predictions["target_date"] = pd.to_datetime(predictions["target_date"])
        else:
            predictions = pd.DataFrame(columns=PREDICTION_COLUMNS)
        result = LSTMRunResult(
            key=key,
            status=cast(RunStatus, status),
            best_epoch=cast(int | None, metadata.get("best_epoch")),
            parameter_count=cast(int | None, metadata.get("parameter_count")),
            mae_bps=(
                float(cast(float, metadata["mae_bps"]))
                if metadata.get("mae_bps") is not None
                else float("nan")
            ),
            rmse_bps=(
                float(cast(float, metadata["rmse_bps"]))
                if metadata.get("rmse_bps") is not None
                else float("nan")
            ),
            predictions=predictions,
            error=cast(str | None, metadata.get("error")),
        )
        return _validate_result(result)
    except Exception as error:
        raise ValueError(f"invalid LSTM run {run_context!r}: {error}") from error


def load_lstm_runs(
    directory: Path,
    fingerprint: str,
) -> dict[str, LSTMRunResult]:
    """Load all compatible completed runs in deterministic order."""
    target_directory = _validate_directory(directory)
    validated_fingerprint = _validate_digest(fingerprint, "fingerprint")
    if not target_directory.exists():
        return {}
    manifest = _read_manifest(target_directory, validated_fingerprint)
    run_paths = sorted(
        path for path in target_directory.glob("*.json") if path.name != MANIFEST_NAME
    )
    if manifest is None:
        if run_paths or any(target_directory.iterdir()):
            raise ValueError("checkpoint manifest is missing")
        return {}
    runs = [_load_run(path, validated_fingerprint) for path in run_paths]
    return {result.key.run_id: result for result in runs}
