"""Persist and resume the independent final-test forecast pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
import statsmodels  # type: ignore[import-untyped]
import tensorflow as tf  # type: ignore[import-untyped]

from evaluation.contracts import PREDICTION_COLUMNS, require_finite_numeric
from evaluation.final_backtest import (
    FinalProgressCallback,
    align_final_predictions,
    build_final_arima_ticker_predictions,
    build_final_hybrid_predictions,
    build_final_residual_dataset,
    build_final_seed_hybrid_predictions,
    build_naive_test_predictions,
)
from evaluation.final_models import (
    FINAL_KEY_COLUMNS,
    FINAL_SORT_COLUMNS,
    FINAL_TEST_PROTOCOL,
    FinalSeedResult,
    FinalTestProtocol,
    average_final_seed_predictions,
    run_final_lstm_seed,
    validate_final_dataset,
)
from models.lstm import MAX_EPOCHS, LSTMConfig, LSTMTrainer

CheckpointFamily = Literal["lstm", "residual_lstm"]

CHECKPOINT_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
PYTHON_VERSION = platform.python_version()
NUMPY_VERSION = np.__version__
PANDAS_VERSION = pd.__version__
STATSMODELS_VERSION = statsmodels.__version__
SKLEARN_VERSION = sklearn.__version__
TENSORFLOW_VERSION = tf.__version__
OPTIMIZER_NAME = "Adam"
LOSS_NAME = "mse"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FAMILIES: tuple[CheckpointFamily, ...] = ("lstm", "residual_lstm")
ROOT_ENTRIES = frozenset(
    {
        MANIFEST_NAME,
        "arima",
        "lstm",
        "residual_lstm",
        "final_predictions.json",
        "final_predictions.csv",
    }
)


@dataclass(frozen=True)
class FinalTestResult:
    """Contain every forecast artifact from final-test execution."""

    fingerprint: str
    predictions: pd.DataFrame
    arima_predictions: pd.DataFrame
    residual_dataset: pd.DataFrame
    lstm_seed_results: tuple[FinalSeedResult, ...]
    residual_seed_results: tuple[FinalSeedResult, ...]
    hybrid_seed_results: tuple[FinalSeedResult, ...]


def _validate_digest(value: object, label: str) -> str:
    """Return one lowercase SHA-256 digest."""
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _protocol_payload(protocol: FinalTestProtocol) -> dict[str, object]:
    """Return the complete canonical final-test protocol identity."""
    return {
        "ticker_specs": [
            {
                "ticker": spec.ticker,
                "arima_order": list(spec.arima_order),
                "lstm_config": {
                    "window": spec.lstm_config.window,
                    "units": spec.lstm_config.units,
                    "dropout": spec.lstm_config.dropout,
                },
                "residual_config": {
                    "window": spec.residual_config.window,
                    "units": spec.residual_config.units,
                    "dropout": spec.residual_config.dropout,
                },
            }
            for spec in protocol.ticker_specs
        ],
        "split_config": {
            "train_start": protocol.split_config.train_start,
            "train_end": protocol.split_config.train_end,
            "validation_start": protocol.split_config.validation_start,
            "validation_end": protocol.split_config.validation_end,
            "test_start": protocol.split_config.test_start,
            "test_end": protocol.split_config.test_end,
        },
        "seeds": list(protocol.seeds),
        "training": {
            "inner_train_fraction": protocol.training.inner_train_fraction,
            "max_epochs": protocol.training.max_epochs,
            "batch_size": protocol.training.batch_size,
            "early_stopping_patience": protocol.training.early_stopping_patience,
            "min_delta": protocol.training.min_delta,
            "learning_rate": protocol.training.learning_rate,
            "shuffle": protocol.training.shuffle,
            "optimizer": OPTIMIZER_NAME,
            "loss": LOSS_NAME,
        },
        "arima_warmup": protocol.arima_warmup,
        "volatility_window": protocol.volatility_window,
        "annualization_days": protocol.annualization_days,
        "model_comparisons": [list(comparison) for comparison in protocol.model_comparisons],
        "dm_lag": protocol.dm_lag,
        "holm_alpha": protocol.holm_alpha,
        "bootstrap_block": protocol.bootstrap_block,
        "bootstrap_repetitions": protocol.bootstrap_repetitions,
        "bootstrap_seed": protocol.bootstrap_seed,
        "bootstrap_confidence": protocol.bootstrap_confidence,
    }


def build_final_fingerprint(
    dataset_sha256: str,
    protocol: FinalTestProtocol = FINAL_TEST_PROTOCOL,
) -> str:
    """Hash source data, frozen final protocol and runtime identity."""
    digest = _validate_digest(dataset_sha256, "dataset_sha256")
    if not isinstance(protocol, FinalTestProtocol):
        raise ValueError("protocol must be a FinalTestProtocol")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "dataset_sha256": digest,
        "protocol": _protocol_payload(protocol),
        "versions": {
            "python": PYTHON_VERSION,
            "numpy": NUMPY_VERSION,
            "pandas": PANDAS_VERSION,
            "statsmodels": STATSMODELS_VERSION,
            "scikit_learn": SKLEARN_VERSION,
            "tensorflow": TENSORFLOW_VERSION,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_directory(directory: object) -> Path:
    """Return a final checkpoint directory restricted to artifacts."""
    if not isinstance(directory, Path):
        raise ValueError("checkpoint directory must be a Path")
    parts = directory.parts
    allowed = any(
        parts[position : position + 2] == ("artifacts", "final_test")
        for position in range(len(parts) - 1)
    )
    if not allowed:
        raise ValueError("checkpoint directory must be under artifacts/final_test")
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


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Return one decoded JSON object."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _atomic_text(path: Path, content: str) -> None:
    """Replace one text file only after complete serialization."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_payload(fingerprint: str) -> dict[str, object]:
    """Return root checkpoint identity metadata."""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "fingerprint": fingerprint,
    }


def _read_manifest(directory: Path, fingerprint: str) -> Mapping[str, object] | None:
    """Read and validate an existing root manifest."""
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


def _validate_root_entries(directory: Path) -> None:
    """Reject files and directories outside the final checkpoint layout."""
    unexpected = sorted(path.name for path in directory.iterdir() if path.name not in ROOT_ENTRIES)
    if unexpected:
        raise ValueError(f"unexpected final checkpoint entries: {unexpected}")


def _ensure_manifest(directory: Path, fingerprint: str) -> None:
    """Create the root checkpoint manifest in an empty directory."""
    if any(directory.iterdir()):
        raise ValueError("checkpoint manifest is missing from a non-empty directory")
    content = json.dumps(_manifest_payload(fingerprint), indent=2, sort_keys=True)
    _atomic_text(directory / MANIFEST_NAME, f"{content}\n")


def _prepare_root(directory: Path, fingerprint: str, *, create: bool) -> Path | None:
    """Validate an existing root or initialize it for writing."""
    target = _validate_directory(directory)
    validated_fingerprint = _validate_digest(fingerprint, "fingerprint")
    if not target.exists():
        if not create:
            return None
        target.mkdir(parents=True)
    manifest = _read_manifest(target, validated_fingerprint)
    if manifest is None:
        if not create:
            if any(target.iterdir()):
                raise ValueError("checkpoint manifest is missing")
            return None
        _ensure_manifest(target, validated_fingerprint)
    else:
        _validate_root_entries(target)
    return target


def _pair_entries(directory: Path) -> tuple[set[str], set[str]]:
    """Return matching JSON and CSV stems from one strict pair directory."""
    if not directory.exists():
        return set(), set()
    if not directory.is_dir():
        raise ValueError("checkpoint family path must be a directory")
    entries = list(directory.iterdir())
    unexpected = [
        path.name for path in entries if not path.is_file() or path.suffix not in {".json", ".csv"}
    ]
    if unexpected:
        raise ValueError(f"unexpected checkpoint files: {sorted(unexpected)}")
    json_stems = {path.stem for path in entries if path.suffix == ".json"}
    csv_stems = {path.stem for path in entries if path.suffix == ".csv"}
    if json_stems != csv_stems:
        raise ValueError("checkpoint contains a missing JSON/CSV pair")
    return json_stems, csv_stems


def _validate_prediction_table(
    predictions: pd.DataFrame,
    *,
    label: str,
    multiple_models: bool = False,
) -> pd.DataFrame:
    """Return an isolated sorted standard prediction table."""
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError(f"{label} must use standard columns in standard order")
    if predictions.empty:
        raise ValueError(f"{label} must not be empty")
    for column in ("date", "target_date"):
        if not pd.api.types.is_datetime64_any_dtype(predictions[column]):
            raise ValueError(f"{label} {column} must be datetime")
        if predictions[column].isna().any():
            raise ValueError(f"{label} must contain finite dates")
    duplicate_columns = (
        [*FINAL_KEY_COLUMNS, "model"] if multiple_models else list(FINAL_KEY_COLUMNS)
    )
    if predictions.duplicated(duplicate_columns).any():
        raise ValueError(f"{label} contains duplicate observation keys")
    require_finite_numeric(predictions["actual_return"], f"{label} actual_return")
    require_finite_numeric(predictions["predicted_return"], f"{label} predicted_return")
    return (
        predictions.loc[:, PREDICTION_COLUMNS]
        .sort_values(list(FINAL_SORT_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )


def _validate_arima(predictions: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return one structurally valid final ARIMA ticker table."""
    expected_tickers = {spec.ticker for spec in FINAL_TEST_PROTOCOL.ticker_specs}
    if ticker not in expected_tickers:
        raise ValueError(f"unexpected checkpoint ticker {ticker!r}")
    validated = _validate_prediction_table(predictions, label="ARIMA predictions")
    if not validated["ticker"].eq(ticker).all():
        raise ValueError("ARIMA prediction ticker does not match checkpoint ticker")
    if not validated["split"].isin({"train", "validation", "test"}).all():
        raise ValueError("ARIMA predictions contain unknown split values")
    if not validated["model"].eq("arima").all():
        raise ValueError("ARIMA predictions must use model 'arima'")
    return validated


def _validate_family(family: object) -> CheckpointFamily:
    """Return one supported LSTM checkpoint family."""
    if family not in FAMILIES:
        raise ValueError("family must be 'lstm' or 'residual_lstm'")
    return family


def _seed_stem(result: FinalSeedResult) -> str:
    """Return a unique ticker-qualified seed checkpoint name."""
    return f"{result.ticker}_{result.model}"


def _validate_seed_result(
    result: object,
    family: CheckpointFamily,
) -> FinalSeedResult:
    """Return one internally consistent final seed result."""
    if not isinstance(result, FinalSeedResult):
        raise ValueError("result must be a FinalSeedResult")
    specs = {spec.ticker: spec for spec in FINAL_TEST_PROTOCOL.ticker_specs}
    if result.ticker not in specs:
        raise ValueError("seed result ticker must belong to the final protocol")
    expected_config = (
        specs[result.ticker].lstm_config
        if family == "lstm"
        else specs[result.ticker].residual_config
    )
    if result.config != expected_config:
        raise ValueError("seed result config does not match the checkpoint family")
    if result.seed not in FINAL_TEST_PROTOCOL.seeds:
        raise ValueError("seed result seed is outside the final protocol")
    expected_model = f"{family}_seed_{result.seed:02d}"
    if result.model != expected_model:
        raise ValueError("seed result model does not match its family and seed")
    if type(result.best_epoch) is not int or not 1 <= result.best_epoch <= MAX_EPOCHS:
        raise ValueError("best_epoch must be within the final training range")
    if type(result.parameter_count) is not int or result.parameter_count <= 0:
        raise ValueError("parameter_count must be a positive integer")
    predictions = _validate_prediction_table(result.predictions, label="seed predictions")
    if not predictions["ticker"].eq(result.ticker).all():
        raise ValueError("seed prediction ticker does not match its result")
    if not predictions["split"].eq("test").all():
        raise ValueError("seed predictions must contain test rows only")
    if not predictions["model"].eq(result.model).all():
        raise ValueError("seed predictions model does not match its result")
    return FinalSeedResult(
        ticker=result.ticker,
        config=result.config,
        seed=result.seed,
        model=result.model,
        best_epoch=result.best_epoch,
        parameter_count=result.parameter_count,
        predictions=predictions,
    )


def _pair_metadata(
    fingerprint: str,
    stem: str,
    rows: int,
    digest: str,
    **identity: object,
) -> dict[str, object]:
    """Return metadata shared by atomic prediction pairs."""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "id": stem,
        "predictions_file": f"{stem}.csv",
        "rows": rows,
        "sha256": digest,
        **identity,
    }


def _save_pair(
    directory: Path,
    fingerprint: str,
    stem: str,
    predictions: pd.DataFrame,
    identity: Mapping[str, object],
    *,
    validate_pairs: bool = True,
) -> None:
    """Atomically persist one JSON and CSV prediction pair."""
    directory.mkdir(parents=True, exist_ok=True)
    if validate_pairs:
        _pair_entries(directory)
    csv_path = directory / f"{stem}.csv"
    json_path = directory / f"{stem}.json"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    json_temporary = json_path.with_suffix(".json.tmp")
    try:
        predictions.to_csv(
            csv_temporary,
            index=False,
            date_format="%Y-%m-%dT%H:%M:%S.%f",
        )
        metadata = _pair_metadata(
            fingerprint,
            stem,
            len(predictions),
            _file_digest(csv_temporary),
            **identity,
        )
        serialized = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)
        json_temporary.write_text(f"{serialized}\n", encoding="utf-8")
        csv_temporary.replace(csv_path)
        json_temporary.replace(json_path)
    finally:
        csv_temporary.unlink(missing_ok=True)
        json_temporary.unlink(missing_ok=True)


def _load_pair(path: Path, fingerprint: str) -> tuple[Mapping[str, object], pd.DataFrame]:
    """Load one checksum-verified generic JSON and CSV pair."""
    context = path.stem
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
        metadata = _mapping(value, "metadata")
        if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("pair format version is incompatible")
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError("pair fingerprint is incompatible")
        if metadata.get("id") != context:
            raise ValueError("pair id does not match metadata filename")
        filename = metadata.get("predictions_file")
        if filename != f"{context}.csv":
            raise ValueError("predictions filename is invalid")
        rows = metadata.get("rows")
        if type(rows) is not int or rows < 0:
            raise ValueError("row count must be a nonnegative integer")
        expected_digest = _validate_digest(metadata.get("sha256"), "sha256")
        csv_path = path.parent / filename
        if _file_digest(csv_path) != expected_digest:
            raise ValueError("prediction checksum does not match metadata")
        predictions = pd.read_csv(csv_path, float_precision="round_trip")
        if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
            raise ValueError("predictions must use standard columns in standard order")
        predictions["date"] = pd.to_datetime(predictions["date"])
        predictions["target_date"] = pd.to_datetime(predictions["target_date"])
        if len(predictions) != rows:
            raise ValueError("prediction row count does not match metadata")
        return metadata, predictions
    except Exception as error:
        raise ValueError(f"invalid final checkpoint pair {context!r}: {error}") from error


def save_final_arima(
    directory: Path,
    fingerprint: str,
    ticker: str,
    predictions: pd.DataFrame,
) -> None:
    """Atomically persist one completed final ARIMA ticker."""
    root = _prepare_root(directory, fingerprint, create=True)
    assert root is not None
    validated = _validate_arima(predictions, ticker)
    _save_pair(
        root / "arima",
        fingerprint,
        ticker,
        validated,
        {"ticker": ticker, "kind": "arima"},
    )


def load_final_arima(directory: Path, fingerprint: str) -> dict[str, pd.DataFrame]:
    """Load all compatible completed final ARIMA tickers."""
    root = _prepare_root(directory, fingerprint, create=False)
    if root is None:
        return {}
    family_directory = root / "arima"
    stems, _ = _pair_entries(family_directory)
    loaded: dict[str, pd.DataFrame] = {}
    for ticker in sorted(stems):
        metadata, predictions = _load_pair(family_directory / f"{ticker}.json", fingerprint)
        if metadata.get("kind") != "arima" or metadata.get("ticker") != ticker:
            raise ValueError(f"invalid ARIMA checkpoint identity for {ticker!r}")
        loaded[ticker] = _validate_arima(predictions, ticker)
    return loaded


def save_final_seed_result(
    directory: Path,
    fingerprint: str,
    family: CheckpointFamily,
    result: FinalSeedResult,
) -> None:
    """Atomically persist one completed final LSTM seed."""
    root = _prepare_root(directory, fingerprint, create=True)
    assert root is not None
    validated_family = _validate_family(family)
    validated = _validate_seed_result(result, validated_family)
    stem = _seed_stem(validated)
    _save_pair(
        root / validated_family,
        fingerprint,
        stem,
        validated.predictions,
        {
            "kind": "seed",
            "family": validated_family,
            "ticker": validated.ticker,
            "seed": validated.seed,
            "model": validated.model,
            "config": {
                "window": validated.config.window,
                "units": validated.config.units,
                "dropout": validated.config.dropout,
            },
            "best_epoch": validated.best_epoch,
            "parameter_count": validated.parameter_count,
        },
    )


def load_final_seed_results(
    directory: Path,
    fingerprint: str,
    family: CheckpointFamily,
) -> dict[str, FinalSeedResult]:
    """Load all compatible completed seeds for one final family."""
    validated_family = _validate_family(family)
    root = _prepare_root(directory, fingerprint, create=False)
    if root is None:
        return {}
    family_directory = root / validated_family
    stems, _ = _pair_entries(family_directory)
    loaded: dict[str, FinalSeedResult] = {}
    for stem in sorted(stems):
        metadata, predictions = _load_pair(family_directory / f"{stem}.json", fingerprint)
        if metadata.get("kind") != "seed" or metadata.get("family") != validated_family:
            raise ValueError(f"invalid seed checkpoint identity for {stem!r}")
        config_data = _mapping(metadata.get("config"), "config")
        result = FinalSeedResult(
            ticker=cast(str, metadata.get("ticker")),
            config=LSTMConfig(
                window=cast(int, config_data.get("window")),
                units=cast(int, config_data.get("units")),
                dropout=cast(float, config_data.get("dropout")),
            ),
            seed=cast(int, metadata.get("seed")),
            model=cast(str, metadata.get("model")),
            best_epoch=cast(int, metadata.get("best_epoch")),
            parameter_count=cast(int, metadata.get("parameter_count")),
            predictions=predictions,
        )
        validated = _validate_seed_result(result, validated_family)
        if _seed_stem(validated) != stem:
            raise ValueError(f"seed checkpoint id does not match metadata for {stem!r}")
        loaded[stem] = validated
    return loaded


def _validate_final_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one fully aligned four-model final table."""
    validated = _validate_prediction_table(
        predictions,
        label="final predictions",
        multiple_models=True,
    )
    tables = {
        model: validated.loc[validated["model"].eq(model)].reset_index(drop=True)
        for model in ("naive_zero", "arima", "lstm", "arima_lstm")
    }
    return align_final_predictions(
        tables["naive_zero"],
        tables["arima"],
        tables["lstm"],
        tables["arima_lstm"],
    )


def save_final_predictions(
    directory: Path,
    fingerprint: str,
    predictions: pd.DataFrame,
) -> None:
    """Atomically persist the complete aligned four-model table."""
    root = _prepare_root(directory, fingerprint, create=True)
    assert root is not None
    validated = _validate_final_predictions(predictions)
    _save_pair(
        root,
        fingerprint,
        "final_predictions",
        validated,
        {"kind": "final_predictions"},
        validate_pairs=False,
    )


def load_final_predictions(
    directory: Path,
    fingerprint: str,
) -> pd.DataFrame | None:
    """Load the complete aligned four-model table when present."""
    root = _prepare_root(directory, fingerprint, create=False)
    if root is None:
        return None
    json_path = root / "final_predictions.json"
    csv_path = root / "final_predictions.csv"
    if json_path.exists() != csv_path.exists():
        raise ValueError("checkpoint contains a missing JSON/CSV pair")
    if not json_path.exists():
        return None
    metadata, predictions = _load_pair(json_path, fingerprint)
    if metadata.get("kind") != "final_predictions":
        raise ValueError("invalid final predictions checkpoint identity")
    return _validate_final_predictions(predictions)


def _expected_arima_schedule(
    dataset: pd.DataFrame,
    ticker: str,
    protocol: FinalTestProtocol,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return source-derived keys and actuals for one ARIMA ticker."""
    rows = dataset.loc[dataset["ticker"].eq(ticker)].sort_values(
        ["date", "target_date"],
        kind="mergesort",
    )
    dates = pd.DatetimeIndex(rows["date"])
    eligible = [
        position
        for position, origin in enumerate(dates)
        if int(dates.searchsorted(origin, side="right")) >= protocol.arima_warmup
    ]
    selected = rows.iloc[eligible].reset_index(drop=True)
    return selected.loc[:, FINAL_KEY_COLUMNS], selected["target_return"]


def _expected_seed_schedule(
    dataset: pd.DataFrame,
    ticker: str,
    window: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return source-derived complete test sequences for one ticker."""
    rows = dataset.loc[dataset["ticker"].eq(ticker)].sort_values(
        ["date", "target_date"],
        kind="mergesort",
    )
    complete = pd.Series(False, index=rows.index)
    complete.iloc[window - 1 :] = True
    selected = rows.loc[complete & rows["split"].eq("test")]
    return (
        selected.loc[:, FINAL_KEY_COLUMNS].reset_index(drop=True),
        selected["target_return"].reset_index(drop=True),
    )


def _validate_cached_arima_schedules(
    cached: Mapping[str, pd.DataFrame],
    dataset: pd.DataFrame,
    protocol: FinalTestProtocol,
) -> None:
    """Require every cached ticker to match its full source schedule."""
    for ticker, predictions in cached.items():
        expected_keys, expected_actual = _expected_arima_schedule(dataset, ticker, protocol)
        if not predictions.loc[:, FINAL_KEY_COLUMNS].equals(expected_keys):
            raise ValueError(f"cached ARIMA ticker {ticker!r} does not match schedule")
        if not np.allclose(
            predictions["actual_return"].to_numpy(dtype="float64"),
            expected_actual.to_numpy(dtype="float64"),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"cached ARIMA ticker {ticker!r} actual_return does not match source")


def _validate_cached_seed_schedules(
    cached: Mapping[str, FinalSeedResult],
    dataset: pd.DataFrame,
) -> None:
    """Require every cached seed to match complete source-derived test rows."""
    for stem, result in cached.items():
        expected_keys, expected_actual = _expected_seed_schedule(
            dataset,
            result.ticker,
            result.config.window,
        )
        predictions = result.predictions
        if not predictions.loc[:, FINAL_KEY_COLUMNS].equals(expected_keys):
            raise ValueError(f"cached seed {stem!r} does not match schedule")
        if not np.allclose(
            predictions["actual_return"].to_numpy(dtype="float64"),
            expected_actual.to_numpy(dtype="float64"),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"cached seed {stem!r} actual_return does not match source")


def _seed_cache_key(ticker: str, family: CheckpointFamily, seed: int) -> str:
    """Return the persistent id for one expected final seed."""
    return f"{ticker}_{family}_seed_{seed:02d}"


def _ordered_seed_results(
    cached: Mapping[str, FinalSeedResult],
    protocol: FinalTestProtocol,
    family: CheckpointFamily,
) -> tuple[FinalSeedResult, ...]:
    """Return all final seed results in protocol order."""
    return tuple(
        cached[_seed_cache_key(spec.ticker, family, seed)]
        for spec in protocol.ticker_specs
        for seed in protocol.seeds
    )


def run_final_test_pipeline(
    dataset: pd.DataFrame,
    dataset_sha256: str,
    checkpoint_dir: Path,
    trainer: LSTMTrainer,
    protocol: FinalTestProtocol = FINAL_TEST_PROTOCOL,
    progress: FinalProgressCallback | None = None,
) -> FinalTestResult:
    """Run or resume every forecast required for final-test evaluation."""
    validated = validate_final_dataset(dataset, protocol)
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    fingerprint = build_final_fingerprint(dataset_sha256, protocol)

    arima_by_ticker = load_final_arima(checkpoint_dir, fingerprint)
    lstm_by_seed = load_final_seed_results(checkpoint_dir, fingerprint, "lstm")
    residual_by_seed = load_final_seed_results(
        checkpoint_dir,
        fingerprint,
        "residual_lstm",
    )
    cached_final = load_final_predictions(checkpoint_dir, fingerprint)
    _validate_cached_arima_schedules(arima_by_ticker, validated, protocol)
    _validate_cached_seed_schedules(lstm_by_seed, validated)

    total = len(protocol.ticker_specs) * (1 + 2 * len(protocol.seeds))
    completed = len(arima_by_ticker) + len(lstm_by_seed) + len(residual_by_seed)
    if progress is not None:
        progress(completed, total, "checkpoint")

    for spec in protocol.ticker_specs:
        if spec.ticker in arima_by_ticker:
            continue
        predictions = build_final_arima_ticker_predictions(
            validated,
            spec.ticker,
            protocol,
        )
        candidate = {spec.ticker: predictions}
        _validate_cached_arima_schedules(candidate, validated, protocol)
        save_final_arima(
            checkpoint_dir,
            fingerprint,
            spec.ticker,
            predictions,
        )
        arima_by_ticker[spec.ticker] = predictions
        completed += 1
        if progress is not None:
            progress(completed, total, f"arima:{spec.ticker}")

    arima_predictions = (
        pd.concat(
            [arima_by_ticker[spec.ticker] for spec in protocol.ticker_specs],
            ignore_index=True,
        )
        .sort_values(list(FINAL_SORT_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )
    residual_dataset = build_final_residual_dataset(validated, arima_predictions, protocol)
    _validate_cached_seed_schedules(residual_by_seed, residual_dataset)

    for spec in protocol.ticker_specs:
        for seed in protocol.seeds:
            key = _seed_cache_key(spec.ticker, "lstm", seed)
            if key in lstm_by_seed:
                continue
            result = run_final_lstm_seed(
                validated,
                spec.ticker,
                spec.lstm_config,
                seed,
                trainer,
                "lstm",
            )
            _validate_cached_seed_schedules({key: result}, validated)
            save_final_seed_result(checkpoint_dir, fingerprint, "lstm", result)
            lstm_by_seed[key] = result
            completed += 1
            if progress is not None:
                progress(completed, total, f"lstm:{spec.ticker}:{seed:02d}")

    for spec in protocol.ticker_specs:
        for seed in protocol.seeds:
            key = _seed_cache_key(spec.ticker, "residual_lstm", seed)
            if key in residual_by_seed:
                continue
            result = run_final_lstm_seed(
                residual_dataset,
                spec.ticker,
                spec.residual_config,
                seed,
                trainer,
                "residual_lstm",
            )
            _validate_cached_seed_schedules({key: result}, residual_dataset)
            save_final_seed_result(
                checkpoint_dir,
                fingerprint,
                "residual_lstm",
                result,
            )
            residual_by_seed[key] = result
            completed += 1
            if progress is not None:
                progress(completed, total, f"residual_lstm:{spec.ticker}:{seed:02d}")

    lstm_seed_results = _ordered_seed_results(lstm_by_seed, protocol, "lstm")
    residual_seed_results = _ordered_seed_results(
        residual_by_seed,
        protocol,
        "residual_lstm",
    )
    arima_test = arima_predictions.loc[arima_predictions["split"].eq("test")].reset_index(drop=True)
    lstm_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    hybrid_seed_values: list[FinalSeedResult] = []
    for spec in protocol.ticker_specs:
        ticker_lstm = tuple(result for result in lstm_seed_results if result.ticker == spec.ticker)
        ticker_residual = tuple(
            result for result in residual_seed_results if result.ticker == spec.ticker
        )
        lstm_frames.append(average_final_seed_predictions(ticker_lstm, "lstm"))
        residual_ensemble = average_final_seed_predictions(
            ticker_residual,
            "residual_lstm",
        )
        residual_frames.append(residual_ensemble)
        hybrid_seed_values.extend(build_final_seed_hybrid_predictions(arima_test, ticker_residual))

    lstm_predictions = pd.concat(lstm_frames, ignore_index=True)
    residual_predictions = pd.concat(residual_frames, ignore_index=True)
    hybrid_predictions = build_final_hybrid_predictions(
        arima_test,
        residual_predictions,
    )
    naive_predictions = build_naive_test_predictions(validated)
    predictions = align_final_predictions(
        naive_predictions,
        arima_test,
        lstm_predictions,
        hybrid_predictions,
    )
    if cached_final is None:
        save_final_predictions(checkpoint_dir, fingerprint, predictions)
    elif not cached_final.equals(predictions):
        raise ValueError("cached final predictions do not match completed component forecasts")
    else:
        predictions = cached_final

    return FinalTestResult(
        fingerprint=fingerprint,
        predictions=predictions,
        arima_predictions=arima_predictions,
        residual_dataset=residual_dataset,
        lstm_seed_results=lstm_seed_results,
        residual_seed_results=residual_seed_results,
        hybrid_seed_results=tuple(hybrid_seed_values),
    )
