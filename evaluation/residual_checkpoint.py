"""Persist and resume expanding ARIMA residual forecasts."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import statsmodels  # type: ignore[import-untyped]

from evaluation.arima_residuals import (
    ARIMA_RESIDUAL_WARMUP,
    FIXED_ARIMA_ORDERS,
    SOURCE_KEY,
    ARIMAResidualData,
    validate_arima_residual_data,
)
from evaluation.contracts import (
    EVALUATION_SPLITS,
    PREDICTION_COLUMNS,
    require_finite_numeric,
    validate_model_dataset,
)
from evaluation.loading import REQUIRED_DATASET_COLUMNS
from evaluation.walk_forward import build_walk_forward_predictions
from models.arima import ARIMAFitter, ARIMAOrder

ResidualProgressCallback = Callable[[int, int, str], None]
CHECKPOINT_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
PYTHON_VERSION = platform.python_version()
NUMPY_VERSION = np.__version__
PANDAS_VERSION = pd.__version__
STATSMODELS_VERSION = statsmodels.__version__
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _validate_digest(value: object, label: str) -> str:
    """Return one lowercase SHA-256 digest."""
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_warmup(value: object) -> int:
    """Return a positive integer warmup boundary."""
    if type(value) is not int or value <= 0:
        raise ValueError("warmup must be a positive integer")
    return value


def _validate_orders(orders: object) -> dict[str, ARIMAOrder]:
    """Return ticker-sorted nonnegative ARIMA orders."""
    if not isinstance(orders, Mapping) or not orders:
        raise ValueError("orders must be a non-empty mapping")
    validated: dict[str, ARIMAOrder] = {}
    for ticker, value in orders.items():
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("orders must use non-empty ticker names")
        if (
            not isinstance(value, tuple)
            or len(value) != 3
            or any(type(part) is not int or part < 0 for part in value)
        ):
            raise ValueError(f"order for ticker {ticker!r} must contain nonnegative integers")
        validated[ticker] = cast(ARIMAOrder, value)
    return dict(sorted(validated.items()))


def _fingerprint_payload(
    dataset_sha256: str,
    orders: Mapping[str, ARIMAOrder],
    warmup: int,
) -> dict[str, object]:
    """Return the complete canonical residual protocol identity."""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "dataset_sha256": dataset_sha256,
        "orders": [
            {"ticker": ticker, "order": list(order)} for ticker, order in sorted(orders.items())
        ],
        "warmup": warmup,
        "versions": {
            "python": PYTHON_VERSION,
            "numpy": NUMPY_VERSION,
            "pandas": PANDAS_VERSION,
            "statsmodels": STATSMODELS_VERSION,
        },
    }


def build_residual_fingerprint(
    dataset_sha256: str,
    orders: Mapping[str, ARIMAOrder],
    warmup: int,
) -> str:
    """Hash the source data and complete ARIMA protocol."""
    digest = _validate_digest(dataset_sha256, "dataset_sha256")
    validated_orders = _validate_orders(orders)
    validated_warmup = _validate_warmup(warmup)
    payload = _fingerprint_payload(digest, validated_orders, validated_warmup)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def residual_dataset_digest(dataset: pd.DataFrame) -> str:
    """Hash canonical residual rows without CSV round-trip ambiguity."""
    validate_model_dataset(dataset)
    if dataset.columns.tolist() != list(REQUIRED_DATASET_COLUMNS):
        raise ValueError("residual dataset must use standard columns in standard order")
    for column in ("date", "target_date"):
        if not pd.api.types.is_datetime64_any_dtype(dataset[column]):
            raise ValueError(f"residual dataset {column} must be datetime")
    for column in ("adj_close", "log_return", "target_return"):
        require_finite_numeric(dataset[column], column)

    ordered = dataset.sort_values(["ticker", "target_date", "date"]).reset_index(drop=True)
    records = [
        {
            "date": cast(pd.Timestamp, row.date).isoformat(),
            "target_date": cast(pd.Timestamp, row.target_date).isoformat(),
            "ticker": str(row.ticker),
            "adj_close": cast(float, row.adj_close).hex(),
            "log_return": cast(float, row.log_return).hex(),
            "target_return": cast(float, row.target_return).hex(),
            "split": str(row.split),
        }
        for row in ordered.itertuples(index=False)
    ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
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


def _file_digest(path: Path) -> str:
    """Calculate one file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    """Replace one text file after complete serialization."""
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


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Return one decoded JSON object."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _read_manifest(directory: Path, fingerprint: str) -> Mapping[str, object] | None:
    """Read and validate an existing checkpoint manifest."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
        manifest = _mapping(value, "manifest root")
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
    if any(directory.iterdir()):
        raise ValueError("checkpoint manifest is missing from a non-empty directory")
    content = json.dumps(_manifest_payload(fingerprint), indent=2, sort_keys=True)
    _atomic_text(directory / MANIFEST_NAME, f"{content}\n")


def _validate_predictions(
    predictions: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Return one sorted prediction table following its ticker contract."""
    if ticker not in FIXED_ARIMA_ORDERS:
        raise ValueError(f"unexpected checkpoint ticker {ticker!r}")
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError("predictions must use standard columns in standard order")
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    if not pd.api.types.is_datetime64_any_dtype(predictions["date"]):
        raise ValueError("prediction date must be datetime")
    if not pd.api.types.is_datetime64_any_dtype(predictions["target_date"]):
        raise ValueError("prediction target_date must be datetime")
    if not predictions["ticker"].eq(ticker).all():
        raise ValueError("prediction ticker does not match checkpoint ticker")
    if not predictions["split"].isin(EVALUATION_SPLITS).all():
        raise ValueError("predictions must contain train and validation rows only")
    if not predictions["model"].eq("arima").all():
        raise ValueError("prediction model must be 'arima'")
    if predictions.duplicated(list(SOURCE_KEY)).any():
        raise ValueError("predictions contain duplicate observation keys")
    require_finite_numeric(predictions["actual_return"], "actual_return")
    require_finite_numeric(predictions["predicted_return"], "predicted_return")
    return (
        predictions.loc[:, PREDICTION_COLUMNS]
        .sort_values(["ticker", "target_date", "date"])
        .reset_index(drop=True)
        .copy(deep=True)
    )


def _ticker_metadata(
    fingerprint: str,
    ticker: str,
    predictions_file: str,
    rows: int,
    digest: str,
) -> dict[str, object]:
    """Return metadata for one completed ticker forecast."""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "ticker": ticker,
        "predictions_file": predictions_file,
        "rows": rows,
        "sha256": digest,
    }


def save_residual_predictions(
    directory: Path,
    fingerprint: str,
    ticker: str,
    predictions: pd.DataFrame,
) -> None:
    """Atomically persist one completed ticker prediction table."""
    target_directory = _validate_directory(directory)
    validated_fingerprint = _validate_digest(fingerprint, "fingerprint")
    validated_predictions = _validate_predictions(predictions, ticker)
    target_directory.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(target_directory, validated_fingerprint)

    predictions_path = target_directory / f"{ticker}.csv"
    metadata_path = target_directory / f"{ticker}.json"
    predictions_temporary = predictions_path.with_suffix(".csv.tmp")
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    try:
        validated_predictions.to_csv(
            predictions_temporary,
            index=False,
            date_format="%Y-%m-%dT%H:%M:%S.%f",
        )
        metadata = _ticker_metadata(
            validated_fingerprint,
            ticker,
            predictions_path.name,
            len(validated_predictions),
            _file_digest(predictions_temporary),
        )
        serialized = json.dumps(metadata, indent=2, sort_keys=True)
        metadata_temporary.write_text(f"{serialized}\n", encoding="utf-8")
        predictions_temporary.replace(predictions_path)
        metadata_temporary.replace(metadata_path)
    finally:
        predictions_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)


def _load_ticker(path: Path, fingerprint: str) -> tuple[str, pd.DataFrame]:
    """Load one validated ticker JSON and CSV pair."""
    ticker = path.stem
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
        metadata = _mapping(value, "metadata")
        if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("ticker format version is incompatible")
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError("ticker fingerprint is incompatible")
        if metadata.get("ticker") != ticker:
            raise ValueError("ticker does not match metadata filename")
        predictions_name = metadata.get("predictions_file")
        if predictions_name != f"{ticker}.csv":
            raise ValueError("predictions filename is invalid")
        rows = metadata.get("rows")
        if type(rows) is not int or rows < 0:
            raise ValueError("row count must be a nonnegative integer")
        expected_digest = _validate_digest(metadata.get("sha256"), "sha256")
        predictions_path = path.parent / predictions_name
        if _file_digest(predictions_path) != expected_digest:
            raise ValueError("prediction checksum does not match metadata")
        predictions = pd.read_csv(predictions_path, float_precision="round_trip")
        if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
            raise ValueError("predictions must use standard columns in standard order")
        predictions["date"] = pd.to_datetime(predictions["date"])
        predictions["target_date"] = pd.to_datetime(predictions["target_date"])
        validated = _validate_predictions(predictions, ticker)
        if len(validated) != rows:
            raise ValueError("prediction row count does not match metadata")
        return ticker, validated
    except Exception as error:
        raise ValueError(f"invalid residual checkpoint {ticker!r}: {error}") from error


def load_residual_predictions(
    directory: Path,
    fingerprint: str,
) -> dict[str, pd.DataFrame]:
    """Load all compatible completed ticker forecasts."""
    target_directory = _validate_directory(directory)
    validated_fingerprint = _validate_digest(fingerprint, "fingerprint")
    if not target_directory.exists():
        return {}
    manifest = _read_manifest(target_directory, validated_fingerprint)
    entries = sorted(path for path in target_directory.iterdir() if path.name != MANIFEST_NAME)
    if manifest is None:
        if entries:
            raise ValueError("checkpoint manifest is missing")
        return {}

    unexpected = [
        path
        for path in entries
        if not path.is_file()
        or path.suffix not in {".json", ".csv"}
        or path.stem not in FIXED_ARIMA_ORDERS
    ]
    if unexpected:
        names = [path.name for path in unexpected]
        raise ValueError(f"unexpected checkpoint files: {names}")
    json_stems = {path.stem for path in entries if path.suffix == ".json"}
    csv_stems = {path.stem for path in entries if path.suffix == ".csv"}
    if json_stems != csv_stems:
        raise ValueError("checkpoint contains a missing JSON/CSV pair")
    loaded = [
        _load_ticker(target_directory / f"{ticker}.json", validated_fingerprint)
        for ticker in sorted(json_stems)
    ]
    return dict(loaded)


def _eligible_schedule(dataset: pd.DataFrame, ticker: str, warmup: int) -> pd.DataFrame:
    """Return the exact evaluation keys meeting the history boundary."""
    observations = dataset.loc[
        dataset["ticker"].eq(ticker) & dataset["split"].isin(EVALUATION_SPLITS)
    ].sort_values(["target_date", "date"])
    history_dates = pd.DatetimeIndex(observations["date"].sort_values())
    eligible_positions = [
        position
        for position, origin in enumerate(observations["date"])
        if int(history_dates.searchsorted(pd.Timestamp(origin), side="right")) >= warmup
    ]
    return observations.iloc[eligible_positions].loc[:, SOURCE_KEY].reset_index(drop=True)


def _validate_cached_schedules(
    cached: Mapping[str, pd.DataFrame],
    schedules: Mapping[str, pd.DataFrame],
) -> None:
    """Require cached rows to match each complete eligible schedule."""
    for ticker, predictions in cached.items():
        actual = list(predictions.loc[:, SOURCE_KEY].itertuples(index=False, name=None))
        expected = list(schedules[ticker].itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError(f"cached predictions for ticker {ticker!r} do not match schedule")


def run_arima_residual_backtest(
    dataset: pd.DataFrame,
    dataset_sha256: str,
    checkpoint_dir: Path,
    orders: Mapping[str, ARIMAOrder] | None = None,
    warmup: int = ARIMA_RESIDUAL_WARMUP,
    progress: ResidualProgressCallback | None = None,
) -> ARIMAResidualData:
    """Run or resume fixed-order expanding ARIMA residual forecasts."""
    validate_model_dataset(dataset)
    validated_orders = _validate_orders(FIXED_ARIMA_ORDERS if orders is None else orders)
    validated_warmup = _validate_warmup(warmup)
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    dataset_tickers = set(dataset["ticker"])
    if dataset_tickers != set(validated_orders):
        raise ValueError("orders tickers must exactly match dataset tickers")

    fingerprint = build_residual_fingerprint(
        dataset_sha256,
        validated_orders,
        validated_warmup,
    )
    cached = load_residual_predictions(checkpoint_dir, fingerprint)
    eligible_schedules = {
        ticker: _eligible_schedule(dataset, ticker, validated_warmup) for ticker in validated_orders
    }
    _validate_cached_schedules(cached, eligible_schedules)
    total = sum(len(schedule) for schedule in eligible_schedules.values())
    completed = sum(len(predictions) for predictions in cached.values())
    if progress is not None:
        progress(completed, total, "checkpoint")

    fitter = ARIMAFitter(validated_orders)
    predictions_by_ticker = dict(cached)
    for ticker in validated_orders:
        if ticker in predictions_by_ticker:
            continue
        ticker_dataset = dataset.loc[dataset["ticker"].eq(ticker)].copy(deep=True)
        completed_before_ticker = completed

        def report_ticker(
            ticker_completed: int,
            _ticker_total: int,
            *,
            label: str = ticker,
            offset: int = completed_before_ticker,
        ) -> None:
            if progress is not None:
                progress(offset + ticker_completed, total, label)

        predictions = build_walk_forward_predictions(
            ticker_dataset,
            fitter,
            target_splits=("train", "validation"),
            min_history=validated_warmup,
            progress=report_ticker,
        )
        save_residual_predictions(
            checkpoint_dir,
            fingerprint,
            ticker,
            predictions,
        )
        predictions_by_ticker[ticker] = predictions
        completed += len(predictions)

    combined = pd.concat(
        [predictions_by_ticker[ticker] for ticker in validated_orders],
        ignore_index=True,
    )
    return validate_arima_residual_data(dataset, combined)
