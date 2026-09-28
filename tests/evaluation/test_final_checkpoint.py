"""Tests for fingerprinted resumable final-test checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

import evaluation.final_checkpoint as checkpoint_module
from evaluation.contracts import PREDICTION_COLUMNS
from evaluation.final_checkpoint import (
    FinalTestResult,
    build_final_fingerprint,
    load_final_arima,
    load_final_predictions,
    load_final_seed_results,
    run_final_test_pipeline,
    save_final_arima,
    save_final_predictions,
    save_final_seed_result,
)
from evaluation.final_models import FINAL_TEST_PROTOCOL, FinalSeedResult, FinalTestProtocol
from models.lstm import LSTMConfig, LSTMTrainer

DATASET_DIGEST = "1" * 64
FINGERPRINT = "a" * 64
CORE_TICKERS = ("AAPL", "JPM", "SPY", "XOM")


def checkpoint_directory(tmp_path: Path) -> Path:
    """Return one allowed final-test checkpoint directory"""
    return tmp_path / "artifacts" / "final_test"


def file_digest(path: Path) -> str:
    """Hash file bytes independently from production code"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(directory: Path, payload: object) -> None:
    """Write one controlled root checkpoint manifest"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def update_metadata(path: Path, **changes: object) -> None:
    """Apply controlled changes to checkpoint metadata"""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(changes)
    path.write_text(json.dumps(metadata), encoding="utf-8")


def arima_predictions(ticker: str = "AAPL") -> pd.DataFrame:
    """Return one structurally valid full-split ARIMA table"""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-12-29", "2019-12-31", "2025-12-30"]),
            "target_date": pd.to_datetime(["2015-12-30", "2020-01-02", "2025-12-31"]),
            "ticker": ticker,
            "split": ["train", "test", "test"],
            "model": "arima",
            "actual_return": [0.012345678912345678, -0.002, 0.003],
            "predicted_return": [-0.019876543219876543, 0.001, -0.0005],
        }
    )
    return frame.loc[:, PREDICTION_COLUMNS]


def seed_predictions(
    ticker: str = "AAPL",
    family: str = "lstm",
    seed: int = 0,
) -> pd.DataFrame:
    """Return complete-boundary seed predictions"""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-31", "2025-12-30"]),
            "target_date": pd.to_datetime(["2020-01-02", "2025-12-31"]),
            "ticker": ticker,
            "split": "test",
            "model": f"{family}_seed_{seed:02d}",
            "actual_return": [0.012345678912345678, -0.002345678912345678],
            "predicted_return": [0.0012345678912345678, -0.00012345678912345678],
        }
    ).loc[:, PREDICTION_COLUMNS]


def seed_result(
    ticker: str = "AAPL",
    family: str = "lstm",
    seed: int = 0,
) -> FinalSeedResult:
    """Return one valid final seed result"""
    spec = next(spec for spec in FINAL_TEST_PROTOCOL.ticker_specs if spec.ticker == ticker)
    config = spec.lstm_config if family == "lstm" else spec.residual_config
    return FinalSeedResult(
        ticker=ticker,
        config=config,
        seed=seed,
        model=f"{family}_seed_{seed:02d}",
        best_epoch=seed + 1,
        parameter_count=73,
        predictions=seed_predictions(ticker, family, seed),
    )


def aligned_predictions() -> pd.DataFrame:
    """Return a complete four-model final table"""
    rows = []
    for model_position, model in enumerate(("naive_zero", "arima", "lstm", "arima_lstm")):
        for ticker_position, ticker in enumerate(CORE_TICKERS):
            for date, target_date, actual in (
                ("2019-12-31", "2020-01-02", 0.01),
                ("2025-12-30", "2025-12-31", -0.02),
            ):
                rows.append(
                    {
                        "date": date,
                        "target_date": target_date,
                        "ticker": ticker,
                        "split": "test",
                        "model": model,
                        "actual_return": actual + ticker_position / 1_000,
                        "predicted_return": model_position / 1_000,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return (
        frame.loc[:, PREDICTION_COLUMNS]
        .sort_values(["model", "ticker", "target_date", "date"])
        .reset_index(drop=True)
    )


def test_build_final_fingerprint_is_deterministic_sha256() -> None:
    """Return stable lowercase digest for complete protocol"""
    first = build_final_fingerprint(DATASET_DIGEST)
    second = build_final_fingerprint(DATASET_DIGEST)

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "protocol",
    [
        replace(FINAL_TEST_PROTOCOL, arima_warmup=253),
        replace(FINAL_TEST_PROTOCOL, volatility_window=22),
        replace(FINAL_TEST_PROTOCOL, annualization_days=253),
        replace(
            FINAL_TEST_PROTOCOL,
            training=replace(FINAL_TEST_PROTOCOL.training, inner_train_fraction=0.75),
        ),
        replace(FINAL_TEST_PROTOCOL, dm_lag=19, bootstrap_block=20),
        replace(FINAL_TEST_PROTOCOL, holm_alpha=0.1),
        replace(FINAL_TEST_PROTOCOL, bootstrap_repetitions=9_999),
        replace(FINAL_TEST_PROTOCOL, bootstrap_seed=43),
        replace(FINAL_TEST_PROTOCOL, bootstrap_confidence=0.9),
    ],
)
def test_build_final_fingerprint_changes_with_protocol(protocol: FinalTestProtocol) -> None:
    """Bind every configurable final protocol section"""
    assert build_final_fingerprint(DATASET_DIGEST, protocol) != build_final_fingerprint(
        DATASET_DIGEST
    )


@pytest.mark.parametrize(
    "attribute",
    [
        "CHECKPOINT_FORMAT_VERSION",
        "PYTHON_VERSION",
        "NUMPY_VERSION",
        "PANDAS_VERSION",
        "STATSMODELS_VERSION",
        "SKLEARN_VERSION",
        "TENSORFLOW_VERSION",
    ],
)
def test_build_final_fingerprint_binds_format_and_versions(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
) -> None:
    """Invalidate final cache after environment changes"""
    baseline = build_final_fingerprint(DATASET_DIGEST)
    replacement: object = 2 if attribute == "CHECKPOINT_FORMAT_VERSION" else "9.9.9"

    monkeypatch.setattr(checkpoint_module, attribute, replacement)

    assert build_final_fingerprint(DATASET_DIGEST) != baseline


@pytest.mark.parametrize(
    ("digest", "protocol", "message"),
    [
        pytest.param("invalid", FINAL_TEST_PROTOCOL, "SHA-256", id="digest"),
        pytest.param(DATASET_DIGEST, "invalid", "protocol", id="protocol"),
    ],
)
def test_build_final_fingerprint_rejects_invalid_inputs(
    digest: str,
    protocol: object,
    message: str,
) -> None:
    """Reject malformed source and protocol identities"""
    with pytest.raises(ValueError, match=message):
        build_final_fingerprint(digest, cast(FinalTestProtocol, protocol))


def test_final_arima_checkpoint_round_trip_is_exact(tmp_path: Path) -> None:
    """Restore exact ARIMA floats dates and metadata"""
    directory = checkpoint_directory(tmp_path)
    expected = arima_predictions()

    save_final_arima(directory, FINGERPRINT, "AAPL", expected)
    loaded = load_final_arima(directory, FINGERPRINT)

    assert set(loaded) == {"AAPL"}
    pd.testing.assert_frame_equal(loaded["AAPL"], expected, check_exact=True)
    metadata_path = directory / "arima" / "AAPL.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["ticker"] == "AAPL"
    assert metadata["rows"] == len(expected)
    assert metadata["predictions_file"] == "AAPL.csv"
    assert metadata["sha256"] == file_digest(directory / "arima" / "AAPL.csv")
    assert not list(directory.rglob("*.tmp"))


@pytest.mark.parametrize("family", ["lstm", "residual_lstm"])
def test_final_seed_checkpoint_round_trip_is_exact(tmp_path: Path, family: str) -> None:
    """Restore exact seed metadata configuration and predictions"""
    directory = checkpoint_directory(tmp_path)
    expected = seed_result(family=family)

    save_final_seed_result(
        directory,
        FINGERPRINT,
        cast(Any, family),
        expected,
    )
    loaded = load_final_seed_results(directory, FINGERPRINT, cast(Any, family))

    key = f"AAPL_{family}_seed_00"
    assert set(loaded) == {key}
    restored = loaded[key]
    assert restored.ticker == expected.ticker
    assert restored.config == expected.config
    assert restored.seed == expected.seed
    assert restored.model == expected.model
    assert restored.best_epoch == expected.best_epoch
    assert restored.parameter_count == expected.parameter_count
    pd.testing.assert_frame_equal(restored.predictions, expected.predictions, check_exact=True)


def test_final_predictions_checkpoint_round_trip_is_exact(tmp_path: Path) -> None:
    """Restore exact aligned four-model final predictions"""
    directory = checkpoint_directory(tmp_path)
    expected = aligned_predictions()

    save_final_predictions(directory, FINGERPRINT, expected)
    loaded = load_final_predictions(directory, FINGERPRINT)

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, expected, check_exact=True)
    metadata = json.loads((directory / "final_predictions.json").read_text(encoding="utf-8"))
    assert metadata["rows"] == len(expected)
    assert metadata["sha256"] == file_digest(directory / "final_predictions.csv")


def test_final_checkpoint_accepts_integer_numeric_columns(tmp_path: Path) -> None:
    """Round trip finite integer-valued return columns"""
    directory = checkpoint_directory(tmp_path)
    predictions = arima_predictions()
    predictions["actual_return"] = [1, 2, 3]
    predictions["predicted_return"] = [0, 0, 0]

    save_final_arima(directory, FINGERPRINT, "AAPL", predictions)
    loaded = load_final_arima(directory, FINGERPRINT)["AAPL"]

    assert loaded["actual_return"].tolist() == [1, 2, 3]
    assert loaded["predicted_return"].tolist() == [0, 0, 0]


@pytest.mark.parametrize("loader", ["arima", "seed", "final"])
def test_final_checkpoint_returns_empty_for_absent_data(tmp_path: Path, loader: str) -> None:
    """Treat absent family data as unfinished work"""
    directory = checkpoint_directory(tmp_path)
    if loader == "arima":
        assert load_final_arima(directory, FINGERPRINT) == {}
    elif loader == "seed":
        assert load_final_seed_results(directory, FINGERPRINT, "lstm") == {}
    else:
        assert load_final_predictions(directory, FINGERPRINT) is None


@pytest.mark.parametrize(
    ("loader", "relative_file"),
    [
        pytest.param("arima", "arima/AAPL.csv", id="missing-arima-json"),
        pytest.param("seed", "lstm/AAPL_lstm_seed_00.csv", id="missing-seed-json"),
        pytest.param("final", "final_predictions.csv", id="missing-final-json"),
    ],
)
def test_final_checkpoint_rejects_missing_pairs(
    tmp_path: Path,
    loader: str,
    relative_file: str,
) -> None:
    """Reject orphaned CSV files without matching metadata"""
    directory = checkpoint_directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"format_version": 1, "fingerprint": FINGERPRINT}),
        encoding="utf-8",
    )
    path = directory / relative_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("orphan", encoding="utf-8")

    with pytest.raises(ValueError, match="pair"):
        if loader == "arima":
            load_final_arima(directory, FINGERPRINT)
        elif loader == "seed":
            load_final_seed_results(directory, FINGERPRINT, "lstm")
        else:
            load_final_predictions(directory, FINGERPRINT)


@pytest.mark.parametrize("damage", ["checksum", "rows", "fingerprint", "temporary"])
def test_final_arima_checkpoint_rejects_corruption(tmp_path: Path, damage: str) -> None:
    """Reject corrupted incompatible and temporary ARIMA files"""
    directory = checkpoint_directory(tmp_path)
    save_final_arima(directory, FINGERPRINT, "AAPL", arima_predictions())
    metadata_path = directory / "arima" / "AAPL.json"
    csv_path = directory / "arima" / "AAPL.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if damage == "checksum":
        csv_path.write_text("corrupt", encoding="utf-8")
    elif damage == "rows":
        metadata["rows"] += 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "fingerprint":
        metadata["fingerprint"] = "b" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        (directory / "arima" / "partial.tmp").write_text("partial", encoding="utf-8")

    with pytest.raises(ValueError):
        load_final_arima(directory, FINGERPRINT)


@pytest.mark.parametrize("operation", ["load", "save"])
def test_final_checkpoint_rejects_incompatible_manifest(
    tmp_path: Path,
    operation: str,
) -> None:
    """Prevent mixing results from different fingerprints"""
    directory = checkpoint_directory(tmp_path)
    save_final_arima(directory, FINGERPRINT, "AAPL", arima_predictions())

    with pytest.raises(ValueError, match="fingerprint"):
        if operation == "load":
            load_final_arima(directory, "b" * 64)
        else:
            save_final_arima(directory, "b" * 64, "AAPL", arima_predictions())


def test_final_checkpoint_rejects_invalid_locations(tmp_path: Path) -> None:
    """Restrict final checkpoints to artifacts final_test paths"""
    with pytest.raises(ValueError, match="artifacts/final_test"):
        load_final_arima(tmp_path / "outside", FINGERPRINT)
    with pytest.raises(ValueError, match="SHA-256"):
        load_final_arima(checkpoint_directory(tmp_path), "invalid")
    with pytest.raises(ValueError, match="Path"):
        load_final_arima(cast(Path, "artifacts/final_test"), FINGERPRINT)

    file_path = checkpoint_directory(tmp_path)
    file_path.parent.mkdir(parents=True)
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        load_final_arima(file_path, FINGERPRINT)


@pytest.mark.parametrize(
    ("case", "operation", "message"),
    [
        pytest.param("non-object", "load", "object", id="manifest-object"),
        pytest.param("version", "load", "version", id="manifest-version"),
        pytest.param("unexpected", "load", "unexpected", id="unexpected-entry"),
        pytest.param("missing", "load", "manifest is missing", id="missing-load"),
        pytest.param("missing", "save", "manifest is missing", id="missing-save"),
        pytest.param("family-file", "load", "family path", id="family-file"),
    ],
)
def test_final_checkpoint_rejects_invalid_root_layouts(
    tmp_path: Path,
    case: str,
    operation: str,
    message: str,
) -> None:
    """Reject incompatible manifests and root layouts"""
    directory = checkpoint_directory(tmp_path)
    if case == "non-object":
        write_manifest(directory, [])
    elif case == "version":
        write_manifest(directory, {"format_version": 2, "fingerprint": FINGERPRINT})
    elif case == "unexpected":
        write_manifest(directory, {"format_version": 1, "fingerprint": FINGERPRINT})
        (directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif case == "missing":
        directory.mkdir(parents=True)
        (directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    else:
        write_manifest(directory, {"format_version": 1, "fingerprint": FINGERPRINT})
        (directory / "arima").write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        if operation == "save":
            save_final_arima(directory, FINGERPRINT, "AAPL", arima_predictions())
        else:
            load_final_arima(directory, FINGERPRINT)


def test_final_checkpoint_loads_existing_empty_directory_as_absent(tmp_path: Path) -> None:
    """Treat existing empty root as unfinished work"""
    directory = checkpoint_directory(tmp_path)
    directory.mkdir(parents=True)

    assert load_final_arima(directory, FINGERPRINT) == {}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("columns", "standard columns", id="columns"),
        pytest.param("empty", "must not be empty", id="empty"),
        pytest.param("date-type", "date must be datetime", id="date-type"),
        pytest.param("missing-date", "finite dates", id="missing-date"),
        pytest.param("duplicate", "duplicate", id="duplicate"),
        pytest.param("ticker-argument", "unexpected checkpoint ticker", id="ticker-argument"),
        pytest.param("ticker-table", "does not match", id="ticker-table"),
        pytest.param("split", "unknown split", id="split"),
        pytest.param("model", "model 'arima'", id="model"),
    ],
)
def test_save_final_arima_rejects_invalid_predictions(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    """Reject malformed ARIMA checkpoint prediction tables"""
    predictions = arima_predictions()
    ticker = "AAPL"
    if case == "columns":
        predictions = predictions.drop(columns="predicted_return")
    elif case == "empty":
        predictions = predictions.iloc[0:0]
    elif case == "date-type":
        predictions["date"] = predictions["date"].astype(str)
    elif case == "missing-date":
        predictions.loc[0, "date"] = pd.NaT
    elif case == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif case == "ticker-argument":
        ticker = "TSLA"
    elif case == "ticker-table":
        predictions["ticker"] = "JPM"
    elif case == "split":
        predictions.loc[0, "split"] = "unknown"
    else:
        predictions["model"] = "wrong"

    with pytest.raises(ValueError, match=message):
        save_final_arima(
            checkpoint_directory(tmp_path),
            FINGERPRINT,
            ticker,
            predictions,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("family", "family", id="family"),
        pytest.param("result-type", "FinalSeedResult", id="result-type"),
        pytest.param("ticker", "ticker", id="ticker"),
        pytest.param("config", "config", id="config"),
        pytest.param("seed", "outside", id="seed"),
        pytest.param("model", "family and seed", id="model"),
        pytest.param("epoch-bool", "best_epoch", id="epoch-bool"),
        pytest.param("epoch-range", "best_epoch", id="epoch-range"),
        pytest.param("parameters-bool", "parameter_count", id="parameters-bool"),
        pytest.param("parameters-range", "parameter_count", id="parameters-range"),
        pytest.param("table-ticker", "ticker does not match", id="table-ticker"),
        pytest.param("table-split", "test rows", id="table-split"),
        pytest.param("table-model", "model does not match", id="table-model"),
    ],
)
def test_save_final_seed_rejects_invalid_results(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    """Reject inconsistent final seed checkpoint results"""
    family: Any = "lstm"
    result: Any = seed_result()
    if case == "family":
        family = "wrong"
    elif case == "result-type":
        result = object()
    elif case == "ticker":
        result = replace(result, ticker="TSLA")
    elif case == "config":
        result = replace(result, config=LSTMConfig(window=6, units=16, dropout=0.0))
    elif case == "seed":
        result = replace(result, seed=10)
    elif case == "model":
        result = replace(result, model="wrong")
    elif case == "epoch-bool":
        result = replace(result, best_epoch=True)
    elif case == "epoch-range":
        result = replace(result, best_epoch=0)
    elif case == "parameters-bool":
        result = replace(result, parameter_count=True)
    elif case == "parameters-range":
        result = replace(result, parameter_count=0)
    else:
        predictions = result.predictions.copy(deep=True)
        if case == "table-ticker":
            predictions["ticker"] = "JPM"
        elif case == "table-split":
            predictions["split"] = "validation"
        else:
            predictions["model"] = "wrong"
        result = replace(result, predictions=predictions)

    with pytest.raises(ValueError, match=message):
        save_final_seed_result(
            checkpoint_directory(tmp_path),
            FINGERPRINT,
            family,
            result,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("metadata-object", "object", id="metadata-object"),
        pytest.param("version", "version", id="version"),
        pytest.param("id", "pair id", id="id"),
        pytest.param("filename", "filename", id="filename"),
        pytest.param("rows-type", "row count", id="rows-type"),
        pytest.param("rows-negative", "row count", id="rows-negative"),
        pytest.param("columns", "standard columns", id="columns"),
    ],
)
def test_final_pair_loader_rejects_invalid_metadata_and_csv(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    """Reject malformed generic prediction pair contents"""
    directory = checkpoint_directory(tmp_path)
    save_final_arima(directory, FINGERPRINT, "AAPL", arima_predictions())
    metadata_path = directory / "arima" / "AAPL.json"
    csv_path = directory / "arima" / "AAPL.csv"
    if case == "metadata-object":
        metadata_path.write_text("[]", encoding="utf-8")
    elif case == "version":
        update_metadata(metadata_path, format_version=2)
    elif case == "id":
        update_metadata(metadata_path, id="JPM")
    elif case == "filename":
        update_metadata(metadata_path, predictions_file="JPM.csv")
    elif case == "rows-type":
        update_metadata(metadata_path, rows=True)
    elif case == "rows-negative":
        update_metadata(metadata_path, rows=-1)
    else:
        malformed = pd.read_csv(csv_path).rename(columns={"model": "wrong"})
        malformed.to_csv(csv_path, index=False)
        update_metadata(metadata_path, sha256=file_digest(csv_path))

    with pytest.raises(ValueError, match=message):
        load_final_arima(directory, FINGERPRINT)


def test_final_arima_loader_rejects_invalid_identity(tmp_path: Path) -> None:
    """Reject ARIMA metadata with mismatched identity"""
    directory = checkpoint_directory(tmp_path)
    save_final_arima(directory, FINGERPRINT, "AAPL", arima_predictions())
    update_metadata(directory / "arima" / "AAPL.json", kind="wrong")

    with pytest.raises(ValueError, match="ARIMA checkpoint identity"):
        load_final_arima(directory, FINGERPRINT)


@pytest.mark.parametrize("case", ["identity", "config", "stem"])
def test_final_seed_loader_rejects_invalid_metadata(
    tmp_path: Path,
    case: str,
) -> None:
    """Reject seed metadata inconsistent with stored predictions"""
    directory = checkpoint_directory(tmp_path)
    save_final_seed_result(directory, FINGERPRINT, "lstm", seed_result())
    family_directory = directory / "lstm"
    original_stem = "AAPL_lstm_seed_00"
    metadata_path = family_directory / f"{original_stem}.json"
    if case == "identity":
        update_metadata(metadata_path, family="residual_lstm")
        message = "seed checkpoint identity"
    elif case == "config":
        update_metadata(metadata_path, config=[])
        message = "config must be an object"
    else:
        new_stem = "renamed"
        csv_path = family_directory / f"{original_stem}.csv"
        renamed_csv = family_directory / f"{new_stem}.csv"
        renamed_json = family_directory / f"{new_stem}.json"
        csv_path.rename(renamed_csv)
        update_metadata(
            metadata_path,
            id=new_stem,
            predictions_file=renamed_csv.name,
        )
        metadata_path.rename(renamed_json)
        message = "id does not match metadata"

    with pytest.raises(ValueError, match=message):
        load_final_seed_results(directory, FINGERPRINT, "lstm")


def test_final_predictions_loader_rejects_invalid_identity(tmp_path: Path) -> None:
    """Reject final table with incorrect checkpoint identity"""
    directory = checkpoint_directory(tmp_path)
    save_final_predictions(directory, FINGERPRINT, aligned_predictions())
    update_metadata(directory / "final_predictions.json", kind="wrong")

    with pytest.raises(ValueError, match="final predictions checkpoint identity"):
        load_final_predictions(directory, FINGERPRINT)


def pipeline_dataset() -> pd.DataFrame:
    """Return canonical histories long enough for every final window"""
    train_targets = pd.bdate_range(end="2015-12-31", periods=70)
    schedule: list[tuple[pd.Timestamp, pd.Timestamp, str]] = [
        (target - pd.offsets.BDay(1), target, "train") for target in train_targets
    ]
    schedule.extend(
        [
            (pd.Timestamp("2015-12-31"), pd.Timestamp("2016-01-04"), "validation"),
            (pd.Timestamp("2016-01-04"), pd.Timestamp("2019-12-31"), "validation"),
            (pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-02"), "test"),
            (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"), "test"),
            (pd.Timestamp("2020-01-03"), pd.Timestamp("2025-12-31"), "test"),
        ]
    )
    rows: list[dict[str, object]] = []
    for ticker_position, ticker in enumerate(CORE_TICKERS, start=1):
        for row_position, (date, target_date, split) in enumerate(schedule, start=1):
            value = ticker_position / 100 + row_position / 100_000
            rows.append(
                {
                    "date": date,
                    "target_date": target_date,
                    "ticker": ticker,
                    "adj_close": 100.0 + ticker_position + row_position,
                    "log_return": value,
                    "target_return": value + 0.00001,
                    "split": split,
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=53).reset_index(drop=True)


def pipeline_protocol() -> FinalTestProtocol:
    """Return final protocol with compact ARIMA warm-up"""
    return replace(FINAL_TEST_PROTOCOL, arima_warmup=2)


def dataset_seed_result(
    dataset: pd.DataFrame,
    ticker: str = "AAPL",
    family: str = "lstm",
    seed: int = 0,
) -> FinalSeedResult:
    """Build one seed result on the source test schedule"""
    spec = next(spec for spec in FINAL_TEST_PROTOCOL.ticker_specs if spec.ticker == ticker)
    config = spec.lstm_config if family == "lstm" else spec.residual_config
    rows = (
        dataset.loc[dataset["ticker"].eq(ticker)]
        .sort_values(["date", "target_date"], kind="mergesort")
        .iloc[config.window - 1 :]
    )
    rows = rows.loc[rows["split"].eq("test")]
    predictions = rows.loc[:, ["date", "target_date", "ticker", "split"]].copy()
    predictions["model"] = f"{family}_seed_{seed:02d}"
    predictions["actual_return"] = rows["target_return"].to_numpy(dtype="float64")
    predictions["predicted_return"] = 0.0
    return FinalSeedResult(
        ticker=ticker,
        config=config,
        seed=seed,
        model=f"{family}_seed_{seed:02d}",
        best_epoch=2,
        parameter_count=73,
        predictions=predictions.loc[:, PREDICTION_COLUMNS].reset_index(drop=True),
    )


class FakeFinalModel:
    """Return aligned zero forecasts for final sequence tests."""

    def __init__(self, config: LSTMConfig, name: str) -> None:
        self.config = config
        self.name = name
        self.parameter_count = 73

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Return zero predictions on raw lag windows"""
        return pd.Series(
            0.0,
            index=observations.index,
            dtype="float64",
            name="predicted_return",
        )


class FakeFinalTrainer:
    """Record cheap two-stage final LSTM calls and optional failure."""

    def __init__(self, fail_family: str | None = None) -> None:
        self.fail_family = fail_family
        self.determine_calls = 0
        self.fit_names: list[str] = []

    def determine_best_epoch(
        self,
        config: LSTMConfig,
        seed: int,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
    ) -> int:
        """Record valid nonempty inner partitions"""
        assert config.window == train_features.shape[1]
        assert len(train_features) == len(train_targets)
        assert len(validation_features) == len(validation_targets)
        assert type(seed) is int
        self.determine_calls += 1
        return 2

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
    ) -> FakeFinalModel:
        """Return a controlled model or fail one family"""
        del seed, targets, scaler
        assert features.shape[1] == len(feature_columns) == config.window
        assert epochs == 2
        self.fit_names.append(name)
        if self.fail_family is not None and name.startswith(f"{self.fail_family}_seed_"):
            raise RuntimeError(f"controlled {self.fail_family} failure")
        return FakeFinalModel(config, name)


class FakeARIMABuilder:
    """Build deterministic ticker schedules and optionally fail."""

    def __init__(self, fail_ticker: str | None = None) -> None:
        self.fail_ticker = fail_ticker
        self.calls: list[str] = []

    def __call__(
        self,
        dataset: pd.DataFrame,
        ticker: str,
        protocol: FinalTestProtocol,
        model_fitter: object = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> pd.DataFrame:
        """Return complete eligible zero ARIMA predictions"""
        del model_fitter, progress
        self.calls.append(ticker)
        if ticker == self.fail_ticker:
            raise RuntimeError("controlled ARIMA failure")
        rows = (
            dataset.loc[dataset["ticker"].eq(ticker)]
            .sort_values(["date", "target_date"])
            .iloc[protocol.arima_warmup - 1 :]
        )
        result = rows.loc[:, ["date", "target_date", "ticker", "split"]].copy()
        result["model"] = "arima"
        result["actual_return"] = rows["target_return"].to_numpy(dtype="float64")
        result["predicted_return"] = 0.0
        return result.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)


def install_arima_builder(
    monkeypatch: pytest.MonkeyPatch,
    builder: FakeARIMABuilder,
) -> None:
    """Replace expensive final ARIMA construction with deterministic fake"""
    monkeypatch.setattr(checkpoint_module, "build_final_arima_ticker_predictions", builder)


def test_run_final_test_pipeline_resumes_and_reuses_complete_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist units immediately and resume only missing work"""
    dataset = pipeline_dataset()
    protocol = pipeline_protocol()
    directory = checkpoint_directory(tmp_path)
    builder = FakeARIMABuilder()
    trainer = FakeFinalTrainer()
    progress: list[tuple[int, int, str]] = []
    install_arima_builder(monkeypatch, builder)

    first = run_final_test_pipeline(
        dataset,
        DATASET_DIGEST,
        directory,
        cast(LSTMTrainer, trainer),
        protocol,
        progress=lambda completed, total, label: progress.append((completed, total, label)),
    )

    assert isinstance(first, FinalTestResult)
    assert builder.calls == list(CORE_TICKERS)
    assert len(trainer.fit_names) == 80
    assert len(first.lstm_seed_results) == 40
    assert len(first.residual_seed_results) == 40
    assert len(first.hybrid_seed_results) == 40
    assert set(first.predictions["model"]) == {"naive_zero", "arima", "lstm", "arima_lstm"}
    assert progress[0] == (0, 84, "checkpoint")
    assert progress[-1] == (84, 84, "residual_lstm:XOM:09")
    assert len(load_final_arima(directory, first.fingerprint)) == 4
    assert len(load_final_seed_results(directory, first.fingerprint, "lstm")) == 40
    assert len(load_final_seed_results(directory, first.fingerprint, "residual_lstm")) == 40
    assert load_final_predictions(directory, first.fingerprint) is not None

    for path in (
        directory / "arima" / "XOM.json",
        directory / "arima" / "XOM.csv",
        directory / "lstm" / "AAPL_lstm_seed_00.json",
        directory / "lstm" / "AAPL_lstm_seed_00.csv",
        directory / "residual_lstm" / "AAPL_residual_lstm_seed_00.json",
        directory / "residual_lstm" / "AAPL_residual_lstm_seed_00.csv",
        directory / "final_predictions.json",
        directory / "final_predictions.csv",
    ):
        path.unlink()

    resumed_builder = FakeARIMABuilder()
    resumed_trainer = FakeFinalTrainer()
    resumed_progress: list[tuple[int, int, str]] = []
    install_arima_builder(monkeypatch, resumed_builder)
    resumed = run_final_test_pipeline(
        dataset,
        DATASET_DIGEST,
        directory,
        cast(LSTMTrainer, resumed_trainer),
        protocol,
        progress=lambda completed, total, label: resumed_progress.append((completed, total, label)),
    )

    assert resumed_builder.calls == ["XOM"]
    assert resumed_trainer.fit_names == ["lstm_seed_00", "residual_lstm_seed_00"]
    assert resumed_progress[0] == (81, 84, "checkpoint")
    assert resumed_progress[-1] == (84, 84, "residual_lstm:AAPL:00")
    pd.testing.assert_frame_equal(resumed.predictions, first.predictions, check_exact=True)

    for path in (
        directory / "residual_lstm" / "XOM_residual_lstm_seed_09.json",
        directory / "residual_lstm" / "XOM_residual_lstm_seed_09.csv",
        directory / "final_predictions.json",
        directory / "final_predictions.csv",
    ):
        path.unlink()
    no_progress_builder = FakeARIMABuilder()
    no_progress_trainer = FakeFinalTrainer()
    install_arima_builder(monkeypatch, no_progress_builder)
    no_progress = run_final_test_pipeline(
        dataset,
        DATASET_DIGEST,
        directory,
        cast(LSTMTrainer, no_progress_trainer),
        protocol,
    )

    assert no_progress_builder.calls == []
    assert no_progress_trainer.fit_names == ["residual_lstm_seed_09"]
    pd.testing.assert_frame_equal(no_progress.predictions, first.predictions, check_exact=True)

    cached_builder = FakeARIMABuilder(fail_ticker="AAPL")
    cached_trainer = FakeFinalTrainer(fail_family="lstm")
    install_arima_builder(monkeypatch, cached_builder)
    cached = run_final_test_pipeline(
        dataset,
        DATASET_DIGEST,
        directory,
        cast(LSTMTrainer, cached_trainer),
        protocol,
    )

    assert cached_builder.calls == []
    assert cached_trainer.fit_names == []
    pd.testing.assert_frame_equal(cached.predictions, first.predictions, check_exact=True)

    incompatible = cached.predictions.copy(deep=True)
    incompatible.loc[0, "predicted_return"] = (
        cast(float, incompatible.loc[0, "predicted_return"]) + 0.01
    )
    save_final_predictions(directory, cached.fingerprint, incompatible)
    with pytest.raises(ValueError, match="cached final predictions"):
        run_final_test_pipeline(
            dataset,
            DATASET_DIGEST,
            directory,
            cast(LSTMTrainer, cached_trainer),
            protocol,
        )


def test_run_final_test_pipeline_rejects_noncallable_progress(tmp_path: Path) -> None:
    """Reject invalid progress callback before checkpoint work"""
    with pytest.raises(ValueError, match="progress"):
        run_final_test_pipeline(
            pipeline_dataset(),
            DATASET_DIGEST,
            checkpoint_directory(tmp_path),
            cast(LSTMTrainer, FakeFinalTrainer()),
            pipeline_protocol(),
            progress=cast(Any, "invalid"),
        )


@pytest.mark.parametrize("family", ["arima", "lstm"])
def test_run_final_test_pipeline_rejects_cached_actuals_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    """Reject schedule-valid cached values differing from source"""
    dataset = pipeline_dataset()
    protocol = pipeline_protocol()
    directory = checkpoint_directory(tmp_path)
    fingerprint = build_final_fingerprint(DATASET_DIGEST, protocol)
    if family == "arima":
        result = FakeARIMABuilder()(dataset, "AAPL", protocol)
        result.loc[0, "actual_return"] = cast(float, result.loc[0, "actual_return"]) + 0.01
        save_final_arima(directory, fingerprint, "AAPL", result)
        message = "cached ARIMA.*actual_return"
    else:
        seed = dataset_seed_result(dataset)
        predictions = seed.predictions.copy(deep=True)
        predictions.loc[0, "actual_return"] = (
            cast(float, predictions.loc[0, "actual_return"]) + 0.01
        )
        save_final_seed_result(
            directory,
            fingerprint,
            "lstm",
            replace(seed, predictions=predictions),
        )
        message = "cached seed.*actual_return"
    builder = FakeARIMABuilder()
    trainer = FakeFinalTrainer()
    install_arima_builder(monkeypatch, builder)

    with pytest.raises(ValueError, match=message):
        run_final_test_pipeline(
            dataset,
            DATASET_DIGEST,
            directory,
            cast(LSTMTrainer, trainer),
            protocol,
        )

    assert builder.calls == []
    assert trainer.fit_names == []


@pytest.mark.parametrize("case", ["missing-leading", "missing-interior", "before-schedule"])
def test_run_final_test_pipeline_rejects_cached_arima_schedule_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Reject checksum-valid incomplete or premature ARIMA caches"""
    dataset = pipeline_dataset()
    protocol = pipeline_protocol()
    directory = checkpoint_directory(tmp_path)
    builder = FakeARIMABuilder()
    predictions = builder(dataset, "AAPL", protocol)
    if case == "missing-leading":
        predictions = predictions.iloc[1:].reset_index(drop=True)
    elif case == "missing-interior":
        predictions = predictions.drop(predictions.index[2]).reset_index(drop=True)
    else:
        source = (
            dataset.loc[dataset["ticker"].eq("AAPL")].sort_values(["date", "target_date"]).iloc[[0]]
        )
        early = source.loc[:, ["date", "target_date", "ticker", "split"]].copy()
        early["model"] = "arima"
        early["actual_return"] = source["target_return"].to_numpy(dtype="float64")
        early["predicted_return"] = 0.0
        predictions = pd.concat(
            [early.loc[:, PREDICTION_COLUMNS], predictions],
            ignore_index=True,
        )
    fingerprint = build_final_fingerprint(DATASET_DIGEST, protocol)
    save_final_arima(directory, fingerprint, "AAPL", predictions)
    training_builder = FakeARIMABuilder()
    trainer = FakeFinalTrainer()
    install_arima_builder(monkeypatch, training_builder)

    with pytest.raises(ValueError, match="cached ARIMA.*AAPL.*schedule"):
        run_final_test_pipeline(
            dataset,
            DATASET_DIGEST,
            directory,
            cast(LSTMTrainer, trainer),
            protocol,
        )

    assert training_builder.calls == []
    assert trainer.fit_names == []


@pytest.mark.parametrize("case", ["missing-interior", "extra-row"])
def test_run_final_test_pipeline_rejects_cached_seed_schedule_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Reject checksum-valid malformed seed schedules before fits"""
    dataset = pipeline_dataset()
    protocol = pipeline_protocol()
    directory = checkpoint_directory(tmp_path)
    result = seed_result()
    predictions = result.predictions.copy(deep=True)
    middle = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "target_date": [pd.Timestamp("2020-01-03")],
            "ticker": ["AAPL"],
            "split": ["test"],
            "model": ["lstm_seed_00"],
            "actual_return": [0.01],
            "predicted_return": [0.0],
        }
    ).loc[:, PREDICTION_COLUMNS]
    predictions = pd.concat(
        [predictions.iloc[[0]], middle, predictions.iloc[[1]]], ignore_index=True
    )
    if case == "missing-interior":
        predictions = predictions.drop(predictions.index[1]).reset_index(drop=True)
    else:
        extra = middle.copy()
        extra["date"] = pd.Timestamp("2020-01-03")
        extra["target_date"] = pd.Timestamp("2020-06-01")
        predictions = pd.concat([predictions, extra], ignore_index=True)
    result = replace(result, predictions=predictions)
    fingerprint = build_final_fingerprint(DATASET_DIGEST, protocol)
    save_final_seed_result(directory, fingerprint, "lstm", result)
    builder = FakeARIMABuilder()
    trainer = FakeFinalTrainer()
    install_arima_builder(monkeypatch, builder)

    with pytest.raises(ValueError, match="cached seed.*schedule"):
        run_final_test_pipeline(
            dataset,
            DATASET_DIGEST,
            directory,
            cast(LSTMTrainer, trainer),
            protocol,
        )

    assert builder.calls == []
    assert trainer.fit_names == []


@pytest.mark.parametrize("stage", ["arima", "lstm", "residual_lstm"])
def test_run_final_test_pipeline_preserves_completed_units_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """Keep completed units and omit final table after interruption"""
    dataset = pipeline_dataset()
    protocol = pipeline_protocol()
    directory = checkpoint_directory(tmp_path)
    builder = FakeARIMABuilder(fail_ticker="JPM" if stage == "arima" else None)
    trainer = FakeFinalTrainer(fail_family=stage if stage != "arima" else None)
    install_arima_builder(monkeypatch, builder)

    with pytest.raises((RuntimeError, ValueError), match="controlled"):
        run_final_test_pipeline(
            dataset,
            DATASET_DIGEST,
            directory,
            cast(LSTMTrainer, trainer),
            protocol,
        )

    fingerprint = build_final_fingerprint(DATASET_DIGEST, protocol)
    assert load_final_predictions(directory, fingerprint) is None
    arima_cache = load_final_arima(directory, fingerprint)
    lstm_cache = load_final_seed_results(directory, fingerprint, "lstm")
    residual_cache = load_final_seed_results(directory, fingerprint, "residual_lstm")
    if stage == "arima":
        assert set(arima_cache) == {"AAPL"}
        assert lstm_cache == residual_cache == {}
    elif stage == "lstm":
        assert set(arima_cache) == set(CORE_TICKERS)
        assert lstm_cache == residual_cache == {}
    else:
        assert set(arima_cache) == set(CORE_TICKERS)
        assert len(lstm_cache) == 40
        assert residual_cache == {}
