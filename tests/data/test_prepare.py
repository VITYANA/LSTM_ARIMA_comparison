"""Tests for building reproducible processed-data snapshots."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from analysis.loading import RawSnapshot, SnapshotMetadata
from data import prepare as prepare_module
from data.prepare import build_processed_snapshot, main, parse_args


def file_digest(path: Path) -> str:
    """Calculate an independent SHA-256 fixture digest"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, pd.DataFrame]:
    """Provide deterministic raw provenance and prepared rows"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "manifest.json").write_text('{"snapshot": "test"}\n', encoding="utf-8")
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-01-04", "2010-01-05"]),
            "ticker": ["AAA", "AAA"],
            "adj_close": [100.0, 101.0],
        }
    )
    dataset = pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-01-04", "2016-01-04", "2020-01-02"]),
            "target_date": pd.to_datetime(["2010-01-05", "2016-01-05", "2020-01-03"]),
            "ticker": ["AAA", "AAA", "AAA"],
            "adj_close": [100.0, 110.0, 120.0],
            "log_return": [0.01, 0.02, -0.01],
            "target_return": [0.02, -0.01, 0.03],
            "split": ["train", "validation", "test"],
        }
    )
    metadata = SnapshotMetadata(
        created_at_utc="2026-01-01T00:00:00+00:00",
        tickers=("AAA",),
        start="2010-01-04",
        end="2020-01-04",
        interval="1d",
        source="test source",
    )

    def fake_load_raw_snapshot(path: Path) -> RawSnapshot:
        assert path == raw_dir
        return RawSnapshot(prices=prices, metadata=metadata)

    def fake_build_model_dataset(frame: pd.DataFrame) -> pd.DataFrame:
        pd.testing.assert_frame_equal(frame, prices)
        return dataset.copy()

    monkeypatch.setattr(prepare_module, "load_raw_snapshot", fake_load_raw_snapshot)
    monkeypatch.setattr(prepare_module, "build_model_dataset", fake_build_model_dataset)
    return raw_dir, dataset


def test_build_processed_snapshot_writes_verified_dataset(
    tmp_path: Path,
    prepared_source: tuple[Path, pd.DataFrame],
) -> None:
    """Write deterministic model data and provenance manifest"""
    raw_dir, dataset = prepared_source
    output_dir = tmp_path / "processed"

    manifest_path = build_processed_snapshot(raw_dir, output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = output_dir / "next_trading_day_returns.csv"
    stored = pd.read_csv(dataset_path, parse_dates=["date", "target_date"])

    pd.testing.assert_frame_equal(stored, dataset, check_dtype=False)
    assert manifest_path == output_dir / "next_trading_day_returns.manifest.json"
    assert manifest["source"] == {
        "raw_manifest_path": (raw_dir / "manifest.json").as_posix(),
        "raw_manifest_sha256": file_digest(raw_dir / "manifest.json"),
    }
    assert manifest["target"] == "next-trading-day log return"
    assert manifest["split_basis"] == "target_date"
    assert manifest["files"]["next_trading_day_returns.csv"] == {
        "path": dataset_path.as_posix(),
        "rows": 3,
        "bytes": dataset_path.stat().st_size,
        "sha256": file_digest(dataset_path),
    }
    assert manifest["split_rows"] == {"train": 1, "validation": 1, "test": 1}


def test_build_processed_snapshot_requires_force_for_existing_files(
    tmp_path: Path,
    prepared_source: tuple[Path, pd.DataFrame],
) -> None:
    """Prevent accidental replacement of processed snapshots"""
    raw_dir, dataset = prepared_source
    output_dir = tmp_path / "processed"
    dataset_path = output_dir / "next_trading_day_returns.csv"
    build_processed_snapshot(raw_dir, output_dir)
    dataset_path.write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Use --force"):
        build_processed_snapshot(raw_dir, output_dir)

    assert dataset_path.read_text(encoding="utf-8") == "stale\n"
    manifest_path = build_processed_snapshot(raw_dir, output_dir, force=True)
    stored = pd.read_csv(dataset_path, parse_dates=["date", "target_date"])
    pd.testing.assert_frame_equal(stored, dataset, check_dtype=False)
    assert manifest_path.exists()


def test_parse_args_uses_defaults() -> None:
    """Return documented processed-snapshot defaults"""
    args = parse_args([])

    assert args.raw_dir == Path("data/raw")
    assert args.output_dir == Path("data/processed")
    assert args.force is False


def test_parse_args_accepts_custom_paths_and_force(tmp_path: Path) -> None:
    """Parse custom input output and replacement options"""
    args = parse_args(
        [
            "--raw-dir",
            str(tmp_path / "source"),
            "--output-dir",
            str(tmp_path / "result"),
            "--force",
        ]
    )

    assert args.raw_dir == tmp_path / "source"
    assert args.output_dir == tmp_path / "result"
    assert args.force is True


def test_main_forwards_arguments_and_reports_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forward CLI options and print created manifest"""
    expected = tmp_path / "processed" / "next_trading_day_returns.manifest.json"
    captured: dict[str, object] = {}

    def fake_build(raw_dir: Path, output_dir: Path, force: bool) -> Path:
        captured.update(raw_dir=raw_dir, output_dir=output_dir, force=force)
        return expected

    monkeypatch.setattr(prepare_module, "build_processed_snapshot", fake_build)

    result = main(
        [
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "processed"),
            "--force",
        ]
    )

    assert result == 0
    assert captured == {
        "raw_dir": tmp_path / "raw",
        "output_dir": tmp_path / "processed",
        "force": True,
    }
    assert str(expected) in capsys.readouterr().out


def test_module_entrypoint_displays_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Execute module entrypoint and display CLI help"""
    monkeypatch.setattr(sys, "argv", ["data.prepare", "--help"])

    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(SystemExit) as error,
    ):
        runpy.run_module("data.prepare", run_name="__main__")

    assert error.value.code == 0
    assert "Build and freeze" in capsys.readouterr().out
