"""Tests for loading reproducible raw-data snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from analysis.loading import load_raw_snapshot

PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
]
Manifest = dict[str, object]


@pytest.fixture
def prices() -> pd.DataFrame:
    """Return two small ticker histories with aligned dates"""
    return pd.DataFrame(
        [
            ["2020-01-02", "AAA", 10, 12, 9, 11, 11, 100, 0.0, 0.0],
            ["2020-01-03", "AAA", 11, 13, 10, 12, 12, 110, 0.2, 0.0],
            ["2020-01-02", "BBB", 20, 22, 19, 21, 21, 200, 0.0, 0.0],
            ["2020-01-03", "BBB", 21, 23, 20, 22, 22, 210, 0.0, 2.0],
        ],
        columns=PRICE_COLUMNS,
    )


def build_manifest(prices_path: Path, rows: int) -> Manifest:
    """Build a valid manifest for a fixture CSV"""
    return {
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "config": {
            "tickers": ["AAA", "BBB"],
            "start": "2020-01-01",
            "end": "2020-01-04",
            "interval": "1d",
            "source": "test source",
        },
        "files": {
            "prices.csv": {
                "path": "data/raw/prices.csv",
                "rows": rows,
                "bytes": prices_path.stat().st_size,
                "sha256": hashlib.sha256(prices_path.read_bytes()).hexdigest(),
            }
        },
    }


def write_snapshot(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Manifest]:
    """Create a minimal snapshot fixture and manifest"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    prices_path = raw_dir / "prices.csv"
    frame.to_csv(prices_path, index=False)
    manifest = build_manifest(prices_path, len(frame))
    write_manifest(raw_dir, manifest)
    return raw_dir, manifest


def write_manifest(raw_dir: Path, manifest: object) -> None:
    """Replace the fixture manifest with supplied content"""
    (raw_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def set_manifest_value(manifest: Manifest, path: tuple[str, ...], value: object) -> None:
    """Replace one nested fixture-manifest value"""
    current = manifest
    for key in path[:-1]:
        nested = current[key]
        assert isinstance(nested, dict)
        current = cast(Manifest, nested)
    current[path[-1]] = value


def test_load_raw_snapshot_reads_verified_prices(tmp_path: Path, prices: pd.DataFrame) -> None:
    """Loads verified CSV and parses chronological dates"""
    raw_dir, _ = write_snapshot(tmp_path, prices.sample(frac=1, random_state=1))

    snapshot = load_raw_snapshot(raw_dir)

    assert snapshot.metadata.tickers == ("AAA", "BBB")
    assert snapshot.metadata.source == "test source"
    assert snapshot.prices["date"].dtype == "datetime64[ns]"
    assert snapshot.prices[["ticker", "date"]].values.tolist() == [
        ["AAA", pd.Timestamp("2020-01-02")],
        ["AAA", pd.Timestamp("2020-01-03")],
        ["BBB", pd.Timestamp("2020-01-02")],
        ["BBB", pd.Timestamp("2020-01-03")],
    ]


def test_load_raw_snapshot_rejects_changed_csv(tmp_path: Path, prices: pd.DataFrame) -> None:
    """Rejects a CSV changed after manifest creation"""
    raw_dir, _ = write_snapshot(tmp_path, prices)
    with (raw_dir / "prices.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="SHA-256"):
        load_raw_snapshot(raw_dir)


def test_load_raw_snapshot_rejects_wrong_row_count(tmp_path: Path, prices: pd.DataFrame) -> None:
    """Rejects row count inconsistent with manifest"""
    raw_dir, manifest = write_snapshot(tmp_path, prices)
    set_manifest_value(manifest, ("files", "prices.csv", "rows"), 999)
    write_manifest(raw_dir, manifest)

    with pytest.raises(ValueError, match="row count"):
        load_raw_snapshot(raw_dir)


def test_load_raw_snapshot_rejects_missing_columns(tmp_path: Path, prices: pd.DataFrame) -> None:
    """Rejects snapshots without required market columns"""
    raw_dir, _ = write_snapshot(tmp_path, prices.drop(columns="adj_close"))

    with pytest.raises(ValueError, match="missing columns.*adj_close"):
        load_raw_snapshot(raw_dir)


@pytest.mark.parametrize("invalid_manifest", [None, [], "manifest"])
def test_load_raw_snapshot_rejects_non_mapping_root(
    tmp_path: Path, prices: pd.DataFrame, invalid_manifest: object
) -> None:
    """Rejects non-object manifest root values"""
    raw_dir, _ = write_snapshot(tmp_path, prices)
    write_manifest(raw_dir, invalid_manifest)

    with pytest.raises(ValueError, match="section 'root'.*object"):
        load_raw_snapshot(raw_dir)


@pytest.mark.parametrize(
    ("path", "section"),
    [
        (("config",), "config"),
        (("files",), "files"),
        (("files", "prices.csv"), "files.prices.csv"),
    ],
)
def test_load_raw_snapshot_rejects_non_mapping_sections(
    tmp_path: Path,
    prices: pd.DataFrame,
    path: tuple[str, ...],
    section: str,
) -> None:
    """Rejects malformed nested manifest sections"""
    raw_dir, manifest = write_snapshot(tmp_path, prices)
    set_manifest_value(manifest, path, "invalid")
    write_manifest(raw_dir, manifest)

    with pytest.raises(ValueError, match=rf"section '{section}'.*object"):
        load_raw_snapshot(raw_dir)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (("created_at_utc",), "created_at_utc"),
        (("files", "prices.csv", "sha256"), "files.prices.csv.sha256"),
    ],
)
def test_load_raw_snapshot_rejects_non_string_fields(
    tmp_path: Path,
    prices: pd.DataFrame,
    path: tuple[str, ...],
    field: str,
) -> None:
    """Rejects non-string manifest field values"""
    raw_dir, manifest = write_snapshot(tmp_path, prices)
    set_manifest_value(manifest, path, 123)
    write_manifest(raw_dir, manifest)

    with pytest.raises(ValueError, match=rf"field '{field}'.*string"):
        load_raw_snapshot(raw_dir)


@pytest.mark.parametrize("invalid_bytes", ["100", -1])
def test_load_raw_snapshot_rejects_wrong_byte_count(
    tmp_path: Path, prices: pd.DataFrame, invalid_bytes: object
) -> None:
    """Rejects invalid or inconsistent byte counts"""
    raw_dir, manifest = write_snapshot(tmp_path, prices)
    set_manifest_value(manifest, ("files", "prices.csv", "bytes"), invalid_bytes)
    write_manifest(raw_dir, manifest)

    with pytest.raises(ValueError, match="byte count"):
        load_raw_snapshot(raw_dir)


@pytest.mark.parametrize("invalid_tickers", ["AAA", ["AAA", 123]])
def test_load_raw_snapshot_rejects_invalid_ticker_lists(
    tmp_path: Path, prices: pd.DataFrame, invalid_tickers: object
) -> None:
    """Rejects malformed ticker collections in config"""
    raw_dir, manifest = write_snapshot(tmp_path, prices)
    set_manifest_value(manifest, ("config", "tickers"), invalid_tickers)
    write_manifest(raw_dir, manifest)

    with pytest.raises(ValueError, match="config.tickers.*string list"):
        load_raw_snapshot(raw_dir)
