"""Load reproducible processed-data snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

DATASET_NAME = "next_trading_day_returns.csv"
MANIFEST_NAME = "next_trading_day_returns.manifest.json"
REQUIRED_DATASET_COLUMNS = (
    "date",
    "target_date",
    "ticker",
    "adj_close",
    "log_return",
    "target_return",
    "split",
)


@dataclass(frozen=True)
class ProcessedSnapshot:
    """Verified model dataset and its provenance metadata."""

    dataset: pd.DataFrame
    metadata: Mapping[str, object]


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    """Return a manifest section as a mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest section {name!r} must be an object")
    return cast(Mapping[str, object], value)


def _as_string(value: object, name: str) -> str:
    """Return a manifest field as a string."""
    if not isinstance(value, str):
        raise ValueError(f"Manifest field {name!r} must be a string")
    return value


def _sha256(path: Path) -> str:
    """Calculate a file digest independently."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_processed_snapshot(
    processed_dir: Path = Path("data/processed"),
) -> ProcessedSnapshot:
    """Load model data after verifying its manifest."""
    manifest_path = processed_dir / MANIFEST_NAME
    manifest_value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _as_mapping(manifest_value, "root")
    files = _as_mapping(manifest.get("files"), "files")
    record_name = f"files.{DATASET_NAME}"
    dataset_record = _as_mapping(files.get(DATASET_NAME), record_name)

    dataset_path = processed_dir / DATASET_NAME
    expected_digest = _as_string(
        dataset_record.get("sha256"),
        f"{record_name}.sha256",
    )
    if _sha256(dataset_path) != expected_digest:
        raise ValueError(f"{DATASET_NAME} SHA-256 does not match manifest")

    expected_bytes = dataset_record.get("bytes")
    if (
        type(expected_bytes) is not int
        or expected_bytes < 0
        or dataset_path.stat().st_size != expected_bytes
    ):
        raise ValueError(f"{DATASET_NAME} byte count does not match manifest")

    dataset = pd.read_csv(dataset_path)
    missing_columns = set(REQUIRED_DATASET_COLUMNS) - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"{DATASET_NAME} missing columns: {sorted(missing_columns)}")

    expected_rows = dataset_record.get("rows")
    if type(expected_rows) is not int or expected_rows < 0 or len(dataset) != expected_rows:
        raise ValueError(f"{DATASET_NAME} row count does not match manifest")

    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["target_date"] = pd.to_datetime(dataset["target_date"])
    dataset = dataset.sort_values(["ticker", "target_date"]).reset_index(drop=True)
    return ProcessedSnapshot(dataset=dataset, metadata=manifest)
