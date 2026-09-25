"""Build and freeze the processed model-ready dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from analysis.loading import load_raw_snapshot
from analysis.preparation import DEFAULT_SPLIT_CONFIG, build_model_dataset
from data.data import (
    ensure_outputs_are_available,
    file_record,
    package_version,
    sha256,
    write_csv,
)

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DATASET_NAME = "next_trading_day_returns.csv"
MANIFEST_NAME = "next_trading_day_returns.manifest.json"


def build_processed_snapshot(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    force: bool = False,
) -> Path:
    """Create the model dataset and a provenance manifest."""
    dataset_path = output_dir / DATASET_NAME
    manifest_path = output_dir / MANIFEST_NAME
    ensure_outputs_are_available((dataset_path, manifest_path), force)

    snapshot = load_raw_snapshot(raw_dir)
    dataset = build_model_dataset(snapshot.prices)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(dataset, dataset_path)

    split_rows = {
        split: int(dataset["split"].eq(split).sum()) for split in ("train", "validation", "test")
    }
    raw_manifest_path = raw_dir / "manifest.json"
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "raw_manifest_path": raw_manifest_path.as_posix(),
            "raw_manifest_sha256": sha256(raw_manifest_path),
        },
        "target": "next-trading-day log return",
        "split_basis": "target_date",
        "split_config": asdict(DEFAULT_SPLIT_CONFIG),
        "split_rows": split_rows,
        "software": {
            "python": sys.version.split()[0],
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
        },
        "files": {DATASET_NAME: file_record(dataset_path, len(dataset))},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse processed-snapshot command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--force", action="store_true", help="Replace an existing snapshot")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run processed-data preparation from the command line."""
    args = parse_args(argv)
    manifest_path = build_processed_snapshot(args.raw_dir, args.output_dir, args.force)
    print(f"Processed snapshot created: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
