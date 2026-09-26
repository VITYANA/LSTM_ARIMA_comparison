"""Tests for loading reproducible processed-data snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from evaluation.loading import load_processed_snapshot

DATASET_NAME = "next_trading_day_returns.csv"
MANIFEST_NAME = "next_trading_day_returns.manifest.json"
DATASET_COLUMNS = [
    "date",
    "target_date",
    "ticker",
    "adj_close",
    "log_return",
    "target_return",
    "split",
]
Manifest = dict[str, object]


@pytest.fixture
def model_dataset() -> pd.DataFrame:
    """Return shuffled model rows across research splits"""
    return pd.DataFrame(
        [
            ["2016-01-05", "2016-01-06", "BBB", 51.0, 0.02, -0.01, "validation"],
            ["2010-01-04", "2010-01-05", "AAA", 100.0, 0.01, 0.02, "train"],
            ["2020-01-02", "2020-01-03", "AAA", 120.0, -0.01, 0.03, "test"],
            ["2016-01-04", "2016-01-05", "BBB", 50.0, -0.02, 0.02, "validation"],
        ],
        columns=DATASET_COLUMNS,
    )


def build_manifest(dataset_path: Path, rows: int) -> Manifest:
    """Build a valid processed snapshot manifest"""
    return {
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "source": {
            "raw_manifest_path": "data/raw/manifest.json",
            "raw_manifest_sha256": "raw-digest",
        },
        "target": "next-trading-day log return",
        "split_basis": "target_date",
        "split_config": {
            "train_start": "2006-01-03",
            "train_end": "2015-12-31",
            "validation_start": "2016-01-04",
            "validation_end": "2019-12-31",
            "test_start": "2020-01-02",
            "test_end": "2025-12-31",
        },
        "split_rows": {"train": 1, "validation": 2, "test": 1},
        "software": {"python": "3.11.9", "numpy": "2.4.6", "pandas": "2.3.3"},
        "files": {
            DATASET_NAME: {
                "path": f"data/processed/{DATASET_NAME}",
                "rows": rows,
                "bytes": dataset_path.stat().st_size,
                "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            }
        },
    }


def write_manifest(processed_dir: Path, manifest: object) -> None:
    """Replace fixture manifest with supplied content"""
    (processed_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def write_snapshot(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Manifest]:
    """Create a complete processed snapshot fixture"""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    dataset_path = processed_dir / DATASET_NAME
    frame.to_csv(dataset_path, index=False)
    manifest = build_manifest(dataset_path, len(frame))
    write_manifest(processed_dir, manifest)
    return processed_dir, manifest


def set_manifest_value(manifest: Manifest, path: tuple[str, ...], value: object) -> None:
    """Replace one nested fixture manifest value"""
    current = manifest
    for key in path[:-1]:
        nested = current[key]
        assert isinstance(nested, dict)
        current = cast(Manifest, nested)
    current[path[-1]] = value


def test_load_processed_snapshot_verifies_parses_and_sorts(
    tmp_path: Path,
    model_dataset: pd.DataFrame,
) -> None:
    """Verify integrity dates ordering and metadata"""
    processed_dir, manifest = write_snapshot(tmp_path, model_dataset)

    snapshot = load_processed_snapshot(processed_dir)

    assert snapshot.metadata == manifest
    assert snapshot.dataset["date"].dtype == "datetime64[ns]"
    assert snapshot.dataset["target_date"].dtype == "datetime64[ns]"
    assert snapshot.dataset[["ticker", "target_date"]].values.tolist() == [
        ["AAA", pd.Timestamp("2010-01-05")],
        ["AAA", pd.Timestamp("2020-01-03")],
        ["BBB", pd.Timestamp("2016-01-05")],
        ["BBB", pd.Timestamp("2016-01-06")],
    ]


@pytest.mark.parametrize("invalid_manifest", [None, [], "manifest"])
def test_load_processed_snapshot_rejects_non_mapping_root(
    tmp_path: Path,
    model_dataset: pd.DataFrame,
    invalid_manifest: object,
) -> None:
    """Reject non-object manifest root values"""
    processed_dir, _ = write_snapshot(tmp_path, model_dataset)
    write_manifest(processed_dir, invalid_manifest)

    with pytest.raises(ValueError, match="section 'root'.*object"):
        load_processed_snapshot(processed_dir)


@pytest.mark.parametrize(
    ("path", "section"),
    [
        (("files",), "files"),
        (("files", DATASET_NAME), f"files.{DATASET_NAME}"),
    ],
)
def test_load_processed_snapshot_rejects_non_mapping_sections(
    tmp_path: Path,
    model_dataset: pd.DataFrame,
    path: tuple[str, ...],
    section: str,
) -> None:
    """Reject malformed nested manifest sections"""
    processed_dir, manifest = write_snapshot(tmp_path, model_dataset)
    set_manifest_value(manifest, path, "invalid")
    write_manifest(processed_dir, manifest)

    with pytest.raises(ValueError, match=rf"section '{section}'.*object"):
        load_processed_snapshot(processed_dir)


@pytest.mark.parametrize("invalid_sha256", [None, 123, ["digest"]])
def test_load_processed_snapshot_rejects_non_string_digest(
    tmp_path: Path,
    model_dataset: pd.DataFrame,
    invalid_sha256: object,
) -> None:
    """Reject non-string manifest digest values"""
    processed_dir, manifest = write_snapshot(tmp_path, model_dataset)
    set_manifest_value(
        manifest,
        ("files", DATASET_NAME, "sha256"),
        invalid_sha256,
    )
    write_manifest(processed_dir, manifest)

    with pytest.raises(ValueError, match=r"field '.*sha256'.*string"):
        load_processed_snapshot(processed_dir)


def test_load_processed_snapshot_rejects_changed_dataset(
    tmp_path: Path,
    model_dataset: pd.DataFrame,
) -> None:
    """Reject dataset changed after manifest creation"""
    processed_dir, _ = write_snapshot(tmp_path, model_dataset)
    with (processed_dir / DATASET_NAME).open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="SHA-256"):
        load_processed_snapshot(processed_dir)


@pytest.mark.parametrize("invalid_bytes", [None, "100", -1, True, 999])
def test_load_processed_snapshot_rejects_invalid_byte_count(
    tmp_path: Path,
    model_dataset: pd.DataFrame,
    invalid_bytes: object,
) -> None:
    """Reject invalid or inconsistent byte counts"""
    processed_dir, manifest = write_snapshot(tmp_path, model_dataset)
    set_manifest_value(manifest, ("files", DATASET_NAME, "bytes"), invalid_bytes)
    write_manifest(processed_dir, manifest)

    with pytest.raises(ValueError, match="byte count"):
        load_processed_snapshot(processed_dir)


@pytest.mark.parametrize("invalid_rows", [None, "4", -1, True, 999])
def test_load_processed_snapshot_rejects_invalid_row_count(
    tmp_path: Path,
    model_dataset: pd.DataFrame,
    invalid_rows: object,
) -> None:
    """Reject invalid or inconsistent row counts"""
    processed_dir, manifest = write_snapshot(tmp_path, model_dataset)
    set_manifest_value(manifest, ("files", DATASET_NAME, "rows"), invalid_rows)
    write_manifest(processed_dir, manifest)

    with pytest.raises(ValueError, match="row count"):
        load_processed_snapshot(processed_dir)


@pytest.mark.parametrize("missing_column", DATASET_COLUMNS)
def test_load_processed_snapshot_rejects_missing_columns(
    tmp_path: Path,
    model_dataset: pd.DataFrame,
    missing_column: str,
) -> None:
    """Reject snapshots missing required model columns"""
    processed_dir, _ = write_snapshot(
        tmp_path,
        model_dataset.drop(columns=missing_column),
    )

    with pytest.raises(ValueError, match=rf"missing columns.*{missing_column}"):
        load_processed_snapshot(processed_dir)
